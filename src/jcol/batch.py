"""Headless annotation using the same scheduler and durable columns as the browser."""

from __future__ import annotations

import asyncio
import math
import os
import warnings
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import polars as pl

from .codebook import Codebook
from .core import Cache, Jev, resolve_backend
from .engine import Engine
from .evaluation import evaluate
from .pool import LocalPool
from .project import Project
from .tables import append_columns, identity, read_table, row_texts, select_inputs, write_table


@dataclass
class AnnotationResult:
    table: pl.DataFrame
    metadata: dict
    evaluation: dict

    @property
    def complete(self) -> bool:
        return self.metadata["complete"]

    def write(self, path: str | Path) -> None:
        write_table(self.table, path)


def annotate(table: pl.DataFrame | str | Path, codebook: Codebook | dict | str | Path, **kwargs) -> AnnotationResult:
    """Apply a codebook synchronously. In notebooks with a running loop, await annotate_async instead."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(annotate_async(table, codebook, **kwargs))
    raise RuntimeError("an event loop is already running; use await jcol.annotate_async(...)")


async def annotate_async(table: pl.DataFrame | str | Path, codebook: Codebook | dict | str | Path, *,
                         project: str | Path | None = None, api: str | None = None, model: str | None = None,
                         budget: float = 2.0, concurrency: int = 32, max_chars: int | None = None,
                         cache: bool = True, progress: Callable[[dict], None] | None = None) -> AnnotationResult:
    """Return all source columns plus values/confidences; preserve partial results on budget or API failure.

    `project` checkpoints each successful cell and refuses changed data, inputs, codebook or model settings.
    `budget` is checked before scheduling calls; calls already in flight may exceed it. It resets per invocation.
    """
    validate_settings(budget, concurrency, max_chars)
    df, book, inputs, texts, truncated = prepare(table, codebook, max_chars=max_chars)
    if truncated:
        warnings.warn(f"{truncated} rows will be truncated to {max_chars} characters", stacklevel=2)
    backend, key = resolve_backend(api)
    model = model or os.environ.get("JEV_MODEL") or backend.model
    endpoint = os.environ.get("JEV_URL") or backend.url
    context = identity(df, inputs, model=model, endpoint=endpoint, max_chars=max_chars, codebook=book.to_dict())
    store = Project(project, context) if project is not None else None
    answer_cache = None
    engine = None
    try:
        answer_cache = Cache() if cache else None
        pool = LocalPool(Jev(key, backend, model=model, timeout=12), concurrency, warmup=False)
        engine = Engine(texts, pool, model=model, cache=answer_cache, budget=budget, max_chars=max_chars,
                        endpoint=endpoint, project=store)
        existing = {c.spec.name for c in engine.columns.values()}
        for variable in book.variables:
            if variable.spec.name not in existing:
                engine.commit_spec(variable.spec)
        if any(len(c.values) < df.height for c in engine.columns.values()):
            await engine.start()
            while not engine.idle.is_set():
                try:
                    await asyncio.wait_for(engine.idle.wait(), timeout=1)
                except asyncio.TimeoutError:
                    if progress:
                        progress({"filled": sum(len(c.values) for c in engine.columns.values()),
                                  "total": df.height * len(book.variables), "stats": engine.stats()})
        output = append_columns(df, engine.columns.values())
        counts = {c.spec.name: len(c.values) for c in engine.columns.values()}
        metadata = {"identity": context, "stats": engine.stats(), "filled": counts,
                    "complete": all(n == df.height for n in counts.values()),
                    "fatal": engine.fatal, "truncated_rows": truncated, "project": str(project) if project else None}
        return AnnotationResult(output, metadata, evaluate(output, book))
    finally:
        if engine:
            await engine.close()
        if answer_cache:
            answer_cache.db.close()
        if store:
            store.close()


def validate_settings(budget: float, concurrency: int, max_chars: int | None) -> None:
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if max_chars is not None and (not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0):
        raise ValueError("max_chars must be a positive integer or None")


def prepare(table, codebook, *, max_chars=None):
    """Validate the same input contract for preflight and paid runs, without auth or writes."""
    validate_settings(0, 1, max_chars)
    book = Codebook.load(codebook)
    df = table if isinstance(table, pl.DataFrame) else read_table(table)
    inputs = select_inputs(df, book.inputs)
    outputs = {n for v in book.variables for n in (v.spec.name, v.spec.name + "__confidence")}
    if conflicts := outputs.intersection(df.columns):
        raise ValueError(f"output columns already exist: {', '.join(sorted(conflicts))}")
    # Validate declared labels before starting a paid run, even if no predictions exist yet.
    blank = df.with_columns([pl.lit(None).alias(v.spec.name) for v in book.variables])
    evaluate(blank, book)
    texts = row_texts(df, inputs)
    truncated = sum(len(t) > max_chars for t in texts) if max_chars is not None else 0
    return df, book, inputs, texts, truncated

"""jcol: open a table in the browser and add columns to it by describing them.

    jcol data/complaints-5k.parquet --text narrative

The server holds the API key, the data and the scheduler. The page is one static file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

import polars as pl
import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from . import __version__
from .core import Cache, Jev, JevFatal, resolve_backend
from .engine import Engine
from .pool import LocalPool, ProcessPool
from .project import Project
from .tables import append_columns, identity, read_table, row_texts, select_inputs

STATIC = Path(__file__).parent / "static"
PREVIEW_CHARS = 320


def load_table(path: Path, text: str | list[str] | None, limit: int | None):
    if not path.exists():
        raise SystemExit(f"jcol: no such file: {path}")
    try:
        df = read_table(path, limit)
        inputs = select_inputs(df, text)
    except ValueError as exc:
        raise SystemExit(f"jcol: {exc}") from exc
    return df, inputs[0] if len(inputs) == 1 else inputs


def build(df: pl.DataFrame, text, *, source: str, budget: float, api: str | None, model: str | None,
          cache: bool, workers: int, per_worker: int, project: Path | None = None,
          max_chars: int | None = 4000) -> Starlette:
    if not math.isfinite(budget) or budget < 0 or workers < 0 or per_worker <= 0:
        raise ValueError("budget and workers must be nonnegative; per-worker must be positive")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max-chars must be positive, or 0 to send full rows")
    inputs = select_inputs(df, text)
    texts = row_texts(df, inputs)
    backend, key = resolve_backend(api)
    model = model or os.environ.get("JEV_MODEL") or backend.model
    endpoint = os.environ.get("JEV_URL") or backend.url
    store = Project(project, identity(df, inputs, model=model, endpoint=endpoint, max_chars=max_chars)) if project else None
    pool = (ProcessPool(backend.name, key, model, workers, per_worker) if workers > 0
            else LocalPool(Jev(key, backend, model=model, timeout=12), per_worker))
    engine = Engine(texts, pool, model=model, cache=Cache() if cache else None, budget=budget,
                    max_chars=max_chars, project=store, endpoint=endpoint)
    fields = [c for c in df.columns if c not in inputs]
    table = {
        "version": __version__, "source": source, "n": df.height, "text": ", ".join(inputs), "inputs": list(inputs),
        "fields": fields, "order": engine.order,
        "previews": [t[:PREVIEW_CHARS] for t in engine.texts],
        "values": {c: df[c].cast(pl.String).fill_null("").to_list() for c in fields},
    }

    async def index(request):
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})  # the page changes with the package

    async def init(request):
        return JSONResponse(table | {"stats": engine.stats()})

    async def row(request):
        i = int(request.path_params["i"])
        if not 0 <= i < df.height:
            return JSONResponse({"error": "no such row"}, status_code=404)
        return JSONResponse({k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                             for k, v in df.row(i, named=True).items()})

    async def export(request):
        try:
            payload = await request.json()
            rows = payload.get("rows", list(range(df.height)))
            if not isinstance(rows, list) or any(type(r) is not int or not 0 <= r < df.height for r in rows):
                raise ValueError("rows must be valid zero-based row indices")
            columns, used = [], set(df.columns)
            for col in engine.columns.values():
                if not col.committed:
                    continue
                name = col.spec.name
                while name in used or name + "__confidence" in used:
                    name = "jcol_" + col.id + "__" + name
                used.update((name, name + "__confidence"))
                columns.append(replace(col, spec=replace(col.spec, name=name)))
            output = append_columns(df, columns)[rows]
            return Response(output.write_csv(), media_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="jcol-export.csv"'})
        except (ValueError, TypeError, AttributeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def ws(socket: WebSocket):
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        engine.listeners.add(queue)
        await socket.send_text(json.dumps(engine.snapshot()))

        async def pump():
            while True:
                await socket.send_text(json.dumps(await queue.get()))

        sender = asyncio.ensure_future(pump())
        try:
            while True:
                m = json.loads(await socket.receive_text())
                kind = m.get("type")
                if kind == "preview":
                    engine.preview(m.get("header", ""), [int(r) for r in m.get("rows", [])][:80])
                elif kind == "commit":
                    engine.visible = [int(r) for r in m.get("rows", [])][:80]
                    engine.commit(m.get("header", ""))
                elif kind == "view":
                    engine.view([int(r) for r in m.get("rows", [])][:80])
                elif kind == "remove":
                    engine.remove(str(m.get("id")))
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            engine.listeners.discard(queue)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            await engine.start()
            print(f"jcol: ready ({workers or 1} worker{'s' if workers > 1 else ''})", file=sys.stderr)
            yield
        finally:
            await engine.close()
            if engine.cache:
                engine.cache.db.close()
            if store:
                store.close()

    return Starlette(routes=[Route("/", index), Route("/api/init", init), Route("/api/row/{i}", row),
                             Route("/api/export", export, methods=["POST"]),
                             WebSocketRoute("/ws", ws)], lifespan=lifespan)


def cli(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="jcol browse", description="Open a table in the browser and add columns by describing them.",
                                 epilog="Batch: jcol run --help. Offline validation: jcol evaluate --help.")
    ap.add_argument("file", help="a .parquet, .csv or .tsv table")
    ap.add_argument("--text", nargs="+", metavar="COLUMN", help="input columns Jev reads (default: the one with the most text)")
    ap.add_argument("--limit", type=int, metavar="N", help="use only the first N rows")
    ap.add_argument("--budget", type=float, default=2.0, metavar="DOLLARS", help="stop spending at this much (default 2.00)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--api", choices=("typesafe", "openrouter"), help="which API to call (default: whichever has a key)")
    ap.add_argument("--model", metavar="ID")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the answer cache")
    ap.add_argument("--workers", type=int, default=16, metavar="N",
                    help="worker processes; one process tops out near 200 calls a second (default 16; 0 runs in-process)")
    ap.add_argument("--per-worker", type=int, default=64, metavar="N", help="background calls in flight per worker (default 64)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser window")
    save = ap.add_mutually_exclusive_group()
    save.add_argument("--project", type=Path, help="saved columns (default: FILE.jcol.sqlite)")
    save.add_argument("--no-project", action="store_true", help="keep columns only in memory")
    ap.add_argument("--max-chars", type=int, default=4000, help="characters sent per row (default 4000; 0 sends full rows)")
    ap.add_argument("--version", action="version", version=f"jcol {__version__}")
    a = ap.parse_args(argv)
    df, text = load_table(Path(a.file), a.text, a.limit)
    try:
        project = None if a.no_project else (a.project or Path(str(a.file) + ".jcol.sqlite"))
        if project and project.resolve() == Path(a.file).resolve():
            raise ValueError("project and source table paths must be distinct")
        app = build(df, text, source=Path(a.file).name, budget=a.budget, api=a.api, model=a.model,
                    cache=not a.no_cache, workers=a.workers, per_worker=a.per_worker, project=project,
                    max_chars=None if a.max_chars == 0 else a.max_chars)
    except (JevFatal, ValueError, OSError) as e:
        raise SystemExit(f"jcol: {e}")
    truncated = sum(len(t) > a.max_chars for t in row_texts(df, select_inputs(df, text))) if a.max_chars else 0
    if truncated:
        print(f"jcol: warning: {truncated} rows truncated to {a.max_chars} characters; use --max-chars 0 for full rows", file=sys.stderr)
    url = f"http://127.0.0.1:{a.port}"
    print(f"jcol: {df.height:,} rows of {Path(a.file).name}; reading column {text!r}; budget ${a.budget:.2f}\njcol: {url}",
          file=sys.stderr)
    if not a.no_open:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    cli()

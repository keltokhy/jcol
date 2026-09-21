"""The scheduler behind the table: what gets judged, in what order, and how results reach the browser.

Two lanes. The hot lane judges the rows on screen the moment they are needed, with no queue and with slow
calls re-sent, because a person is looking at them. The background lane fills everything else in a fixed
random order, so the rows finished so far are always a random sample of the table and a running estimate
computed from them is honest.

Every call carries one row and every column that still needs that row, because extra questions in a call
cost almost no time. Identical texts are asked once: complaint data is full of form letters.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from .core import Cache
from .spec import Spec, parse

FLUSH_EVERY = 0.04  # seconds between batches sent to the browser


@dataclass
class Column:
    id: str
    header: str
    spec: Spec
    committed: bool
    values: dict[int, tuple] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    seconds: float | None = None

    def describe(self) -> dict:
        return {"id": self.id, "header": self.header, "kind": self.spec.kind, "name": self.spec.name,
                "options": list(self.spec.options), "committed": self.committed, "seconds": self.seconds}


class Engine:
    def __init__(self, texts: list[str], pool, *, model: str, cache: Cache | None = None, budget: float = 2.0,
                 max_chars: int = 4000, seed: int = 70):
        self.texts, self.pool, self.model, self.cache = texts, pool, model, cache
        self.budget, self.max_chars, self.n = budget, max_chars, len(texts)
        self.order = list(range(self.n))
        random.Random(seed).shuffle(self.order)
        self.columns: dict[str, Column] = {}
        self.visible: list[int] = []
        self.listeners: set[asyncio.Queue] = set()
        self.errors = self.cached = 0
        self.over_budget = False
        self._waiting: dict[str, list[tuple[int, str]]] = {}  # cache key -> the (row, column) cells awaiting it
        self._asked: set[tuple[int, str]] = set()
        self._jobs: dict[int, tuple[list[tuple[str, str]], bool]] = {}  # job id -> ([(column id, cache key)], hot)
        self._job_ids = self._col_ids = 0
        self._preview_id: str | None = None
        self._outbox: list[list] = []
        self._slots = asyncio.Semaphore(pool.capacity)
        self._hot_pending = 0
        self._wake, self._restart = asyncio.Event(), False
        self._tasks: set[asyncio.Task] = set()
        pool.on_result = self._on_result

    # ---- lifecycle -------------------------------------------------------------------------------------

    async def start(self) -> None:
        await self.pool.start()
        for coro in (self._background(), self._flusher()):
            task = asyncio.ensure_future(coro)
            self._tasks.add(task)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.pool.close()

    # ---- what the browser asks for -------------------------------------------------------------------

    def preview(self, header: str, rows: list[int]) -> str | None:
        """Judge the rows on screen for a header that is still being typed. Returns the ghost column's id."""
        spec = parse(header)
        if self._preview_id:
            self.columns.pop(self._preview_id, None)
            self._preview_id = None
        if spec is None:
            self._send({"type": "preview", "column": None})
            return None
        col = self._new_column(header, spec, committed=False)
        self._preview_id = col.id
        self._send({"type": "preview", "column": col.describe()})
        self.visible = rows
        self._hot(rows)
        return col.id

    def commit(self, header: str) -> str | None:
        """Turn a header into a real column: visible rows now, the rest in the background."""
        spec = parse(header)
        if spec is None:
            return None
        if self._preview_id:
            self.columns.pop(self._preview_id, None)
            self._preview_id = None
            self._send({"type": "preview", "column": None})
        col = self._new_column(header, spec, committed=True)
        self._send({"type": "column", "column": col.describe()})
        self._hot(self.visible)
        self._restart = True
        self._wake.set()
        return col.id

    def view(self, rows: list[int]) -> None:
        self.visible = rows
        self._hot(rows)

    def remove(self, col_id: str) -> None:
        if self.columns.pop(col_id, None):
            self._send({"type": "removed", "id": col_id})

    def snapshot(self) -> dict:
        """Everything a freshly connected browser needs to redraw the current state."""
        cols = [c for c in self.columns.values() if c.committed]
        return {"type": "snapshot", "columns": [c.describe() for c in cols],
                "cells": [[c.id, r, v, p] for c in cols for r, (v, p) in c.values.items()], "stats": self.stats()}

    def stats(self) -> dict:
        t = self.pool.totals()
        return {"calls": t["calls"], "cached": self.cached, "hedges": t["hedges"], "tokens": t["tokens"],
                "dollars": round(t["cost"], 5), "errors": self.errors, "inflight": len(self._jobs),
                "budget": self.budget, "over_budget": self.over_budget, "model": t["model"] or self.model}

    # ---- the two lanes -------------------------------------------------------------------------------

    def _new_column(self, header: str, spec: Spec, *, committed: bool) -> Column:
        self._col_ids += 1
        col = Column(f"{'c' if committed else 'p'}{self._col_ids}", header, spec, committed)
        self.columns[col.id] = col
        return col

    def _needs(self, row: int, committed_only: bool = False) -> list[Column]:
        return [c for c in self.columns.values() if row not in c.values and (row, c.id) not in self._asked
                and (c.committed or not committed_only)]

    def _hot(self, rows: list[int]) -> None:
        if not self.over_budget:
            for row in rows:
                if 0 <= row < self.n and (cols := self._needs(row)):
                    self._ask(row, cols, hot=True)

    async def _background(self) -> None:
        sweeps = 0
        while True:
            await self._wake.wait()
            self._wake.clear()
            pos = 0
            while pos < self.n:
                if self._restart:
                    self._restart, pos = False, 0
                row = self.order[pos]
                pos += 1
                if not self._needs(row, committed_only=True):
                    continue
                if self.pool.totals()["cost"] >= self.budget:
                    self.over_budget = True
                    self._send({"type": "budget", "budget": self.budget})
                    break
                while self._hot_pending:  # someone is waiting on their screen; do not queue more behind it
                    await asyncio.sleep(0.01)
                await self._slots.acquire()
                cols = self._needs(row, committed_only=True)  # the wait may have been long; ask only what is still missing
                if not cols or not self._ask(row, cols, hot=False):
                    self._slots.release()
            # a failed call leaves its cell empty; once the stragglers land, sweep again for what is still missing
            while self._jobs:
                await asyncio.sleep(0.05)
            missing = any(c.committed and len(c.values) < self.n for c in self.columns.values())
            if missing and not self.over_budget and sweeps < 2:
                sweeps += 1
                self._wake.set()
            else:
                sweeps = 0

    def _ask(self, row: int, cols: list[Column], *, hot: bool) -> bool:
        """Fill what the cache knows, join calls already in the air, and send the rest. True if a call went out."""
        state = self.texts[row][:self.max_chars]
        send: list[tuple[Column, str]] = []
        for c in cols:
            key = Cache.key(self.model, state, c.spec.question)
            if self.cache and (hit := self.cache.get(key)) is not None:
                self.cached += 1
                self._fill(c, row, hit)
                continue
            self._asked.add((row, c.id))
            waiters = self._waiting.setdefault(key, [])
            waiters.append((row, c.id))
            if len(waiters) == 1:
                send.append((c, key))
            else:
                self.cached += 1  # the same text and question are already in the air
        if not send:
            return False
        self._job_ids += 1
        self._jobs[self._job_ids] = ([(c.id, key) for c, key in send], hot)
        self._hot_pending += hot
        self.pool.submit((self._job_ids, state, {c.id: c.spec.question for c, _ in send}, hot))
        return True

    def _on_result(self, job_id: int, answers: dict | None, error: str | None) -> None:
        asked, hot = self._jobs.pop(job_id, ([], False))
        if hot:
            self._hot_pending -= 1
        else:
            self._slots.release()
        if error:
            self.errors += 1
            if error.startswith("fatal: "):
                self.over_budget = True
                self._send({"type": "fatal", "message": error[7:]})
        for col_id, key in asked:
            answer = None if answers is None else answers.get(col_id)
            if answer is not None and self.cache:
                self.cache.put(key, answer)
            for row, cid in self._waiting.pop(key, []):
                self._asked.discard((row, cid))
                if answer is not None and (c := self.columns.get(cid)):
                    self._fill(c, row, answer)

    def _fill(self, c: Column, row: int, answer: dict) -> None:
        value, p = c.spec.cell(answer)
        c.values[row] = (value, p)
        self._outbox.append([c.id, row, value, p])
        if c.committed and c.seconds is None and len(c.values) == self.n:
            c.seconds = round(time.perf_counter() - c.started, 1)
            self._send({"type": "done", "id": c.id, "seconds": c.seconds})

    # ---- to the browser ------------------------------------------------------------------------------

    def _send(self, message: dict) -> None:
        for q in self.listeners:
            q.put_nowait(message)

    async def _flusher(self) -> None:
        last_stats = 0.0
        while True:
            await asyncio.sleep(FLUSH_EVERY)
            now = time.perf_counter()
            if self._outbox or (self._jobs and now - last_stats > 0.25):
                cells, self._outbox = self._outbox, []
                self._send({"type": "cells", "cells": cells, "stats": self.stats()})
                last_stats = now

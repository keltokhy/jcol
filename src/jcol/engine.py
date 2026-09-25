"""The scheduler behind the table: what gets judged, in what order, and how results reach the browser.

Two lanes. The hot lane judges the rows on screen the moment they are needed, with no queue and with slow
calls re-sent, because a person is looking at them. The background lane fills everything else in a fixed
random order, so the rows finished so far are always a random sample of the table and a running estimate
computed from them is honest.

Every call carries one row and every column that still needs that row, because extra questions in a call
cost almost no time. Identical texts are asked once: complaint data is full of form letters. The runtime's
client does the asking: the cache, sharing a call already in the air, the budget, re-sending slow calls,
and sending from worker processes when it has them.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from jevkit_runtime import Client, JevBudgetExceeded, JevError, JevFatal, Noul
from .spec import Spec, parse

FLUSH_EVERY = 0.04  # seconds between batches sent to the browser
HEDGE_AFTER = 0.35  # seconds; measured: trims a screenful's worst case from about 1.4 s to 0.7 s
WARM_UP = {"q": Noul("This is a greeting.")}


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
    """`capacity` bounds the background lane's rows in the air; the hot lane never waits for it."""

    def __init__(self, texts: list[str], client: Client, *, capacity: int = 64, max_chars: int | None = 4000,
                 seed: int = 70, project=None, warm_up: bool = False):
        self.texts, self.client, self.project = texts, client, project
        self.model, self.endpoint = client.backend.model, client.backend.url
        self.max_chars, self.n, self.warm_up = max_chars, len(texts), warm_up
        self.order = list(range(self.n))
        random.Random(seed).shuffle(self.order)
        self.columns: dict[str, Column] = {}
        self.visible: list[int] = []
        self.listeners: set[asyncio.Queue] = set()
        self.errors = self.cached = 0
        self.over_budget = False
        self.fatal: str | None = None
        self.idle = asyncio.Event()
        self.idle.set()
        self._asked: set[tuple[int, str]] = set()
        self._inflight = 0
        self._col_ids = 0
        self._preview_id: str | None = None
        self._outbox: list[list] = []
        self._slots = asyncio.Semaphore(capacity)
        self._hot_pending = 0
        self._wake, self._restart = asyncio.Event(), False
        self._tasks: set[asyncio.Task] = set()
        if project:
            self.columns = {c.id: c for c in project.columns()}
            self._col_ids = max((int(cid[1:]) for cid in self.columns), default=0)

    @property
    def budget(self) -> float:
        return self.client.budget.limit

    # ---- lifecycle -------------------------------------------------------------------------------------

    async def start(self) -> None:
        await self.client.start()
        if self.warm_up:  # opens the connections, and under a limit teaches the budget the price
            try:
                await self.client.ask("warm up", WARM_UP)
            except (JevError, JevFatal, JevBudgetExceeded):
                pass
        for coro in (self._background(), self._flusher()):
            task = asyncio.ensure_future(coro)
            self._tasks.add(task)
        if any(c.committed and len(c.values) < self.n for c in self.columns.values()):
            self.idle.clear()
            self._wake.set()

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.client.close()

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
        return self.commit_spec(spec, header=header)

    def commit_spec(self, spec: Spec, *, header: str | None = None) -> str:
        """Commit an explicit codebook definition without the interactive header grammar."""
        if self._preview_id:
            self.columns.pop(self._preview_id, None)
            self._preview_id = None
            self._send({"type": "preview", "column": None})
        col = self._new_column(header or spec.name, spec, committed=True)
        if self.project:
            self.project.add(col)
        if self.n == 0:
            col.seconds = 0.0
            if self.project:
                self.project.done(col)
        self._send({"type": "column", "column": col.describe()})
        self._hot(self.visible)
        self._restart = True
        self.idle.clear()
        self._wake.set()
        return col.id

    def view(self, rows: list[int]) -> None:
        self.visible = rows
        self._hot(rows)

    def remove(self, col_id: str) -> None:
        if self.columns.pop(col_id, None):
            if self.project:
                self.project.remove(col_id)
            self._send({"type": "removed", "id": col_id})

    def snapshot(self) -> dict:
        """Everything a freshly connected browser needs to redraw the current state."""
        cols = [c for c in self.columns.values() if c.committed]
        return {"type": "snapshot", "columns": [c.describe() for c in cols],
                "cells": [[c.id, r, v, p] for c in cols for r, (v, p) in c.values.items()], "stats": self.stats()}

    def stats(self) -> dict:
        m = self.client.meter
        return {"calls": m.calls, "cached": self.cached, "hedges": m.hedges, "tokens": m.input_tokens,
                "dollars": round(m.cost, 5), "errors": self.errors, "inflight": self._inflight,
                "budget": None if self.client.budget.unlimited else self.budget, "over_budget": self.over_budget,
                "model": m.model or self.model}

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
            while pos < self.n and not self.over_budget:
                if self._restart:
                    self._restart, pos = False, 0
                row = self.order[pos]
                pos += 1
                if not self._needs(row, committed_only=True):
                    continue
                while self._hot_pending:  # someone is waiting on their screen; do not queue more behind it
                    await asyncio.sleep(0.01)
                await self._slots.acquire()
                if self.over_budget:
                    self._slots.release()
                    break
                cols = self._needs(row, committed_only=True)  # the wait may have been long; ask only what is still missing
                if not cols or not self._ask(row, cols, hot=False):
                    self._slots.release()
            # a failed call leaves its cell empty; once the stragglers land, sweep again for what is still missing
            while self._inflight:
                await asyncio.sleep(0.05)
            missing = any(c.committed and len(c.values) < self.n for c in self.columns.values())
            if missing and not self.over_budget and sweeps < 2:
                sweeps += 1
                self._wake.set()
            else:
                sweeps = 0
                if not self._wake.is_set():
                    self.idle.set()

    def _ask(self, row: int, cols: list[Column], *, hot: bool) -> bool:
        """Fill what the cache knows now, and send the rest. True if a call went out."""
        state = self.texts[row][:self.max_chars]
        plan = self.client.plan(state, {c.id: c.spec.question for c in cols})
        for c in cols:
            if c.id in plan.hits and self._fill(c, row, plan.hits[c.id]):
                self.cached += 1
        if plan.complete:
            return False
        waiting = [c for c in cols if c.id in plan.misses]
        for c in waiting:
            self._asked.add((row, c.id))
        self._inflight += 1
        self._hot_pending += hot
        task = asyncio.ensure_future(self._answer(row, waiting, plan, hot))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _answer(self, row: int, waiting: list[Column], plan, hot: bool) -> None:
        answers = None
        try:
            answers = await self.client.send(plan, priority=hot, hedge_after=HEDGE_AFTER if hot else None)
        except JevBudgetExceeded:
            if not self.over_budget:
                self.over_budget = True
                self._send({"type": "budget", "budget": self.budget})
        except JevFatal as e:
            self.errors += 1
            self.over_budget, self.fatal = True, str(e)
            self._send({"type": "fatal", "message": str(e)})
        except JevError:
            self.errors += 1
        finally:
            self._inflight -= 1
            if hot:
                self._hot_pending -= 1
            else:
                self._slots.release()
            for c in waiting:
                self._asked.discard((row, c.id))
        if answers is None:
            return
        for c in waiting:
            if c.id in answers and (col := self.columns.get(c.id)) is not None:
                if answers.origins[c.id].get("source") == "shared":
                    self.cached += 1  # the same text and question were already in the air
                self._fill(col, row, answers[c.id])

    def _fill(self, c: Column, row: int, answer: dict) -> bool:
        try:
            value, p = c.spec.cell(answer)
        except (KeyError, TypeError, ValueError, OverflowError):
            self.errors += 1
            return False
        if self.project and c.committed:
            self.project.fill(c, row, answer)
        c.values[row] = (value, p)
        self._outbox.append([c.id, row, value, p])
        if c.committed and c.seconds is None and len(c.values) == self.n:
            c.seconds = round(time.perf_counter() - c.started, 1)
            if self.project:
                self.project.done(c)
            self._send({"type": "done", "id": c.id, "seconds": c.seconds})
        return True

    # ---- to the browser ------------------------------------------------------------------------------

    def _send(self, message: dict) -> None:
        for q in self.listeners:
            q.put_nowait(message)

    async def _flusher(self) -> None:
        last_stats = 0.0
        while True:
            await asyncio.sleep(FLUSH_EVERY)
            now = time.perf_counter()
            if self._outbox or (self._inflight and now - last_stats > 0.25):
                cells, self._outbox = self._outbox, []
                self._send({"type": "cells", "cells": cells, "stats": self.stats()})
                last_stats = now

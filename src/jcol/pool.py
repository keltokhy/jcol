"""Where calls actually run.

One Python process tops out near 200 calls a second however many are in flight: the HTTP/2 framing and JSON
work is single-threaded. The API itself takes far more (measured 2026-09-18: 1 process 140 rows/s,
12 processes 824, 24 processes 916), so the table shards its calls across worker processes. Each worker has
its own event loop and connection. Hot jobs, for rows on screen, travel on their own queue and never wait
for a free slot.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .core import BACKENDS, Jev, JevError, JevFatal

HEDGE_AFTER = 0.35  # seconds; measured: trims a screenful's worst case from about 1.4 s to 0.7 s
Job = tuple  # (job_id, state, questions, hot)


def _totals(jev: Jev) -> dict:
    m = jev.meter
    return {"calls": m.calls, "hedges": m.hedges, "retries": m.retries, "tokens": m.input_tokens, "cost": m.cost,
            "model": m.model or jev.model}


class LocalPool:
    """Everything in this process. Fine for tests and small tables."""

    def __init__(self, jev: Jev, per_worker: int = 128, *, warmup: bool = True):
        self.jev, self.sem, self.on_result = jev, asyncio.Semaphore(per_worker), None
        self.warmup = warmup
        self.capacity = per_worker
        self._tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        if not self.warmup:
            return
        try:
            await self.jev.ask("warm up", {"q": {"type": "noul", "instructions": "This is a greeting."}})
        except (JevError, JevFatal):
            pass

    def submit(self, job: Job) -> None:
        task = asyncio.ensure_future(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, job: Job) -> None:
        job_id, state, questions, hot = job
        if not hot:
            await self.sem.acquire()
        try:
            answers = await self.jev.ask(state, questions, hedge_after=HEDGE_AFTER if hot else None)
            self.on_result(job_id, answers, None)
        except JevError as e:
            self.on_result(job_id, None, str(e))
        except JevFatal as e:
            self.on_result(job_id, None, "fatal: " + str(e))
        finally:
            if not hot:
                self.sem.release()

    def totals(self) -> dict:
        return _totals(self.jev)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.jev.close()


def _worker(wid: int, backend_name: str, key: str, model: str | None, per_worker: int, hot_q, bg_q, out_q) -> None:
    async def main() -> None:
        jev = Jev(key, BACKENDS[backend_name], model=model, timeout=12)
        try:
            await jev.ask("warm up", {"q": {"type": "noul", "instructions": "This is a greeting."}})
        except (JevError, JevFatal):
            pass
        out_q.put(("ready", wid, None, None, _totals(jev)))
        loop, sem, threads = asyncio.get_running_loop(), asyncio.Semaphore(per_worker), ThreadPoolExecutor(2)
        tasks: set[asyncio.Task] = set()

        async def handle(job: Job, limited: bool) -> None:
            job_id, state, questions, hot = job
            try:
                answers = await jev.ask(state, questions, hedge_after=HEDGE_AFTER if hot else None)
                out_q.put(("result", wid, job_id, answers, _totals(jev)))
            except JevError as e:
                out_q.put(("error", wid, job_id, str(e), _totals(jev)))
            except JevFatal as e:
                out_q.put(("error", wid, job_id, "fatal: " + str(e), _totals(jev)))
            finally:
                if limited:
                    sem.release()

        async def pull(q, limited: bool) -> None:
            while True:
                if limited:
                    await sem.acquire()  # a saturated worker stops taking background jobs; an idle one picks them up
                job = await loop.run_in_executor(threads, q.get)
                if job is None:
                    return
                task = asyncio.ensure_future(handle(job, limited))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        await asyncio.gather(pull(hot_q, False), pull(bg_q, True))
        await jev.close()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


class ProcessPool:
    def __init__(self, backend_name: str, key: str, model: str | None, workers: int = 16, per_worker: int = 64):
        self.ctx = mp.get_context("spawn")
        self.hot_q, self.bg_q, self.out_q = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        self.workers, self.capacity, self.on_result = workers, workers * per_worker, None
        self._totals: dict[int, dict] = {}
        self._procs = [self.ctx.Process(target=_worker, daemon=True,
                                        args=(w, backend_name, key, model, per_worker, self.hot_q, self.bg_q, self.out_q))
                       for w in range(workers)]

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        ready = asyncio.Event()
        seen: set[int] = set()

        def deliver(kind, wid, job_id, payload, totals):
            self._totals[wid] = totals
            if kind == "ready":
                seen.add(wid)
                if len(seen) == self.workers:
                    ready.set()
            elif kind == "result":
                self.on_result(job_id, payload, None)
            else:
                self.on_result(job_id, None, payload)

        def pump() -> None:
            while True:
                item = self.out_q.get()
                if item is None:
                    return
                loop.call_soon_threadsafe(deliver, *item)

        for p in self._procs:
            p.start()
        threading.Thread(target=pump, daemon=True).start()
        await ready.wait()

    def submit(self, job: Job) -> None:
        (self.hot_q if job[3] else self.bg_q).put(job)

    def totals(self) -> dict:
        out = {"calls": 0, "hedges": 0, "retries": 0, "tokens": 0, "cost": 0.0, "model": ""}
        for t in self._totals.values():
            for k in ("calls", "hedges", "retries", "tokens", "cost"):
                out[k] += t[k]
            out["model"] = t["model"] or out["model"]
        return out

    async def close(self) -> None:
        for _ in self._procs:
            self.hot_q.put(None)
            self.bg_q.put(None)
        self.out_q.put(None)
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

"""The scheduler against a fake pool and, once, against the real in-process pool over a fake API. No network."""

import asyncio
import random

import pytest

from jevkit_runtime import AnswerStore, Backend, Client, answer_key
from jcol.core import PROVIDERS
from jcol.engine import Engine
from jcol.pool import LocalPool, ProcessPool

from fakes import FakeAPI, FakePool

BACKEND = Backend("openrouter", PROVIDERS["openrouter"].url, "jev-test", key="test-key")
JOINT = Backend("diffusiongemma", PROVIDERS["diffusiongemma"].url, "openjev-latest", joint_reads=True)

TEXTS = ["this is fraud about my mortgage!!", "a late fee on my credit card", "fraud again, a credit card!",
         "nothing much", "a late fee on my credit card", "my mortgage servicer lost a payment!"]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 20))


async def until(queue: asyncio.Queue, kind: str) -> dict:
    while (message := await queue.get())["type"] != kind:
        pass
    return message


async def started(texts, pool, **kw) -> tuple[Engine, asyncio.Queue]:
    engine = Engine(texts, pool, **({"backend": BACKEND} | kw))
    queue: asyncio.Queue = asyncio.Queue()
    engine.listeners.add(queue)
    await engine.start()
    return engine, queue


def test_the_background_order_is_a_fixed_shuffle():
    a, b = Engine(["x"] * 50, FakePool(), backend=BACKEND), Engine(["x"] * 50, FakePool(), backend=BACKEND)
    expected = list(range(50))
    random.Random(70).shuffle(expected)
    assert a.order == b.order == expected
    assert Engine(["x"] * 50, FakePool(), backend=BACKEND, seed=1).order != expected


def test_a_committed_column_fills_every_row_and_identical_texts_are_asked_once():
    async def go():
        pool = FakePool()
        engine, queue = await started(TEXTS, pool)
        try:
            col = engine.commit("alleges fraud?")
            column = await until(queue, "column")
            done = await until(queue, "done")
            return engine, pool, col, column, done
        finally:
            await engine.close()

    engine, pool, col, column, done = run(go())
    assert col == "c1" and done["id"] == "c1"
    assert column["column"] | {"seconds": None} == {"id": "c1", "header": "alleges fraud?", "kind": "noul",
                                                    "name": "alleges fraud", "options": [], "committed": True,
                                                    "seconds": None}
    values = engine.columns["c1"].values
    assert [values[r][0] for r in range(6)] == [0.9, 0.1, 0.9, 0.1, 0.1, 0.1]
    assert len(pool.jobs) == 5 and engine.cached == 1  # rows 1 and 4 are the same text
    assert all(not hot for *_, hot in pool.jobs)  # nothing was on screen
    assert sorted(state for _, state, *_ in pool.jobs) == sorted(set(TEXTS))


def test_a_preview_judges_only_the_rows_on_screen_and_a_new_one_replaces_it():
    async def go():
        pool = FakePool()
        engine, queue = await started(TEXTS, pool)
        try:
            first = engine.preview("alleges fraud", [0, 1, 99])
            await until(queue, "cells")
            filled = dict(engine.columns[first].values)
            second = engine.preview("al", [0, 1])  # too short to ask
            return engine, pool, first, filled, second, [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            await engine.close()

    engine, pool, first, filled, second, rest = run(go())
    assert first == "p1" and second is None and not engine.columns
    assert filled == {0: (0.9, 0.9), 1: (0.1, 0.9)}  # row 99 does not exist
    assert [(state, hot) for _, state, _, hot in pool.jobs] == [(TEXTS[0], True), (TEXTS[1], True)]
    assert rest[-1] == {"type": "preview", "column": None}
    assert engine.snapshot()["columns"] == []


def test_the_snapshot_holds_committed_columns_and_not_the_preview():
    async def go():
        pool = FakePool()
        engine, queue = await started(TEXTS[:1], pool)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            engine.visible = [0]
            engine.commit("product: mortgage, credit card, other")
            engine.preview("tone: calm < upset < furious", [0])
            await until(queue, "done")
            await until(queue, "cells")
            return engine, pool
        finally:
            await engine.close()

    engine, pool = run(go())
    assert [sorted(questions) for _, _, questions, _ in pool.jobs] == [["c1"], ["c2"], ["p3"]]
    assert engine.columns["c2"].values[0] == ("mortgage", 0.8)
    assert engine.columns["p3"].values[0] == (2.0, 0.7)
    snapshot = engine.snapshot()
    assert [c["id"] for c in snapshot["columns"]] == ["c1", "c2"]  # a preview is not part of the table
    assert sorted(snapshot["cells"]) == [["c1", 0, 0.9, 0.9], ["c2", 0, "mortgage", 0.8]]


def test_columns_added_together_share_their_calls():
    async def go():
        pool = FakePool()
        engine, queue = await started(TEXTS, pool)
        try:
            engine.commit("alleges fraud?")
            engine.commit("product: mortgage, credit card, other")
            await until(queue, "done")
            await until(queue, "done")
            return pool
        finally:
            await engine.close()

    pool = run(go())
    assert len(pool.jobs) == 5
    assert all(sorted(questions) == ["c1", "c2"] for _, _, questions, _ in pool.jobs)


def test_text_is_cut_to_max_chars_before_it_is_sent():
    async def go():
        pool = FakePool()
        engine, queue = await started(["0123456789"], pool, max_chars=4)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            return pool
        finally:
            await engine.close()

    assert [state for _, state, *_ in run(go()).jobs] == ["0123"]


def test_a_failed_call_is_swept_up_again():
    async def go():
        pool = FakePool(fail={TEXTS[3]: 1})
        engine, queue = await started(TEXTS, pool)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            return engine, pool
        finally:
            await engine.close()

    engine, pool = run(go())
    assert engine.errors == 1 and len(engine.columns["c1"].values) == 6
    assert [state for _, state, *_ in pool.jobs].count(TEXTS[3]) == 2


def test_a_fatal_error_reaches_the_page_and_stops_the_hot_lane():
    async def go():
        pool = FakePool(fail={text: "fatal: openrouter said 402: no credits" for text in TEXTS})
        engine, queue = await started(TEXTS, pool)
        try:
            engine.commit("alleges fraud?")
            message = await until(queue, "fatal")
            await asyncio.sleep(0.2)
            before = len(pool.jobs)
            engine.view([0, 1, 2])  # nothing more is asked, even for rows on screen
            return engine, message, before, len(pool.jobs)
        finally:
            await engine.close()

    engine, message, before, after = run(go())
    assert message == {"type": "fatal", "message": "openrouter said 402: no credits"}
    assert engine.over_budget and before == after and not engine.columns["c1"].values


def test_the_background_stops_at_the_budget():
    async def go():
        pool = FakePool(capacity=1, cost=0.5)
        engine, queue = await started(TEXTS, pool, budget=1.0)
        try:
            engine.commit("alleges fraud?")
            message = await until(queue, "budget")
            await asyncio.sleep(0.2)
            return engine, pool, message
        finally:
            await engine.close()

    engine, pool, message = run(go())
    assert message == {"type": "budget", "budget": 1.0}
    assert engine.over_budget and engine.stats()["over_budget"]
    assert pool.spent >= 1.0 and 0 < len(engine.columns["c1"].values) < len(TEXTS)


def test_answers_come_back_from_the_cache(tmp_path):
    async def fill(cache):
        pool = FakePool()
        engine, queue = await started(TEXTS, pool, cache=cache)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            return engine, pool
        finally:
            await engine.close()

    cache = AnswerStore(tmp_path / "answers.sqlite")
    try:
        first, paid = run(fill(cache))
        again, free = run(fill(cache))
    finally:
        cache.db.close()
    assert len(paid.jobs) == 5 and not free.jobs
    assert again.cached == 6 and again.columns["c1"].values == first.columns["c1"].values


def test_a_joint_read_comes_back_from_the_cache_only_as_the_same_call(tmp_path):
    """DiffusionGemma answers each question in the light of the others in its call. An answer given beside
    another column is not the answer the question would get alone, so the cache serves it only to the same call."""
    async def fill(cache, headers):
        pool = FakePool()
        engine, queue = await started(TEXTS, pool, backend=JOINT, cache=cache)
        try:
            for header in headers:
                engine.commit(header)
            for _ in headers:
                await until(queue, "done")
            return pool
        finally:
            await engine.close()

    both = ["alleges fraud?", "product: mortgage, credit card, other"]
    cache = AnswerStore(tmp_path / "answers.sqlite")
    try:
        first = run(fill(cache, both))
        again = run(fill(cache, both))
        alone = run(fill(cache, both[:1]))
    finally:
        cache.db.close()
    assert len(first.jobs) == 5 and all(sorted(questions) == ["c1", "c2"] for _, _, questions, _ in first.jobs)
    assert not again.jobs
    assert len(alone.jobs) == 5 and all(list(questions) == ["c1"] for _, _, questions, _ in alone.jobs)


def test_cached_answers_say_who_gave_them(tmp_path):
    api, cache = FakeAPI(), AnswerStore(tmp_path / "answers.sqlite")

    async def go():
        pool = LocalPool(Client(BACKEND, transport=api.transport), per_worker=4, warmup=False)
        engine, queue = await started(TEXTS, pool, cache=cache)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            return engine
        finally:
            await engine.close()

    try:
        question = run(go()).columns["c1"].spec.question
        entries = [cache.entry(answer_key(BACKEND, text, question)) for text in set(TEXTS)]
    finally:
        cache.db.close()
    assert len(entries) == 5
    for entry in entries:
        assert entry.metadata["provider"] == "openrouter" and entry.metadata["requested_model"] == "jev-test"
        assert entry.metadata["resolved_model"] == "typesafe/jev-test" and "source" not in entry.metadata


def test_removing_a_column_tells_the_page():
    async def go():
        engine, queue = await started(TEXTS, FakePool())
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            engine.remove("c1")
            engine.remove("c1")  # already gone
            await asyncio.sleep(0.1)
            return engine, [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            await engine.close()

    engine, messages = run(go())
    assert [m for m in messages if m["type"] == "removed"] == [{"type": "removed", "id": "c1"}]
    assert not engine.columns and engine.snapshot()["cells"] == []


def test_the_in_process_pool_over_a_fake_api(tmp_path):
    api, cache = FakeAPI(), AnswerStore(tmp_path / "answers.sqlite")

    async def go():
        pool = LocalPool(Client(BACKEND, transport=api.transport), per_worker=4)
        engine, queue = await started(TEXTS, pool, cache=cache)
        try:
            engine.visible = [0, 1]
            engine.commit("tone: calm < upset < furious")
            await until(queue, "done")
            return engine
        finally:
            await engine.close()

    try:
        engine = run(go())
    finally:
        cache.db.close()
    assert api.bodies[0]["state"] == "warm up" and len(api.rows) == 5
    assert all(b["model"] == "jev-test" and list(b["questions"]) == ["c1"] for b in api.rows)
    assert [engine.columns["c1"].values[r] for r in range(6)] == [(2.0, 0.7), (0.0, 0.7), (1.0, 0.7), (0.0, 0.7),
                                                                  (0.0, 0.7), (1.0, 0.7)]
    stats = engine.stats()
    assert (stats["calls"], stats["model"], stats["inflight"], stats["errors"]) == (6, "typesafe/jev-test", 0, 0)
    assert stats["dollars"] == round(6 * 0.0001, 5)


def test_worker_totals_add_up_without_starting_a_process():
    pool = ProcessPool(BACKEND, workers=2, per_worker=8)
    assert pool.capacity == 16 and not any(p.is_alive() for p in pool._procs)
    pool._totals = {0: {"calls": 3, "hedges": 1, "retries": 0, "tokens": 900, "cost": 0.01, "model": "typesafe/jev-test"},
                    1: {"calls": 2, "hedges": 0, "retries": 1, "tokens": 600, "cost": 0.02, "model": ""}}
    assert pool.totals() == {"calls": 5, "hedges": 1, "retries": 1, "tokens": 1500, "cost": pytest.approx(0.03),
                             "model": "typesafe/jev-test"}

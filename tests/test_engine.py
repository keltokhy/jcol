"""The scheduler over the runtime's client and a fake API. No network."""

import asyncio
import random

from jevkit_runtime import AnswerStore, Backend, Budget, Client, answer_key
from jcol.core import PROVIDERS
from jcol.engine import Engine

from fakes import FakeAPI

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


async def started(texts, api, *, backend=BACKEND, budget=None, store=None, attempts=4, **kw):
    client = Client(backend, transport=api.transport, attempts=attempts, budget=budget or Budget(), store=store)
    engine = Engine(texts, client, **kw)
    queue: asyncio.Queue = asyncio.Queue()
    engine.listeners.add(queue)
    await engine.start()
    return engine, queue


def test_the_background_order_is_a_fixed_shuffle():
    client = Client(BACKEND)
    a, b = Engine(["x"] * 50, client), Engine(["x"] * 50, client)
    expected = list(range(50))
    random.Random(70).shuffle(expected)
    assert a.order == b.order == expected
    assert Engine(["x"] * 50, client, seed=1).order != expected


def test_a_committed_column_fills_every_row_and_identical_texts_are_asked_once():
    api = FakeAPI(delay=0.01)

    async def go():
        engine, queue = await started(TEXTS, api)
        try:
            col = engine.commit("alleges fraud?")
            column = await until(queue, "column")
            done = await until(queue, "done")
            return engine, col, column, done
        finally:
            await engine.close()

    engine, col, column, done = run(go())
    assert col == "c1" and done["id"] == "c1"
    assert column["column"] | {"seconds": None} == {"id": "c1", "header": "alleges fraud?", "kind": "noul",
                                                    "name": "alleges fraud", "options": [], "committed": True,
                                                    "seconds": None}
    values = engine.columns["c1"].values
    assert [values[r][0] for r in range(6)] == [0.9, 0.1, 0.9, 0.1, 0.1, 0.1]
    assert len(api.rows) == 5 and engine.cached == 1  # rows 1 and 4 are the same text, asked once
    assert sorted(b["state"] for b in api.rows) == sorted(set(TEXTS))


def test_a_preview_judges_only_the_rows_on_screen_and_a_new_one_replaces_it():
    api = FakeAPI()

    async def go():
        engine, queue = await started(TEXTS, api)
        try:
            first = engine.preview("alleges fraud", [0, 1, 99])
            while len(engine.columns[first].values) < 2:
                await until(queue, "cells")
            filled = dict(engine.columns[first].values)
            second = engine.preview("al", [0, 1])  # too short to ask
            await asyncio.sleep(0.05)
            return engine, first, filled, second, [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            await engine.close()

    engine, first, filled, second, rest = run(go())
    assert first == "p1" and second is None and not engine.columns
    assert filled == {0: (0.9, 0.9), 1: (0.1, 0.9)}  # row 99 does not exist
    assert sorted(b["state"] for b in api.rows) == sorted(TEXTS[:2])
    assert {"type": "preview", "column": None} in rest
    assert engine.snapshot()["columns"] == []


def test_the_snapshot_holds_committed_columns_and_not_the_preview():
    api = FakeAPI()

    async def go():
        engine, queue = await started(TEXTS[:1], api)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            engine.visible = [0]
            engine.commit("product: mortgage, credit card, other")
            engine.preview("tone: calm < upset < furious", [0])
            await until(queue, "done")
            while 0 not in engine.columns["p3"].values:
                await until(queue, "cells")
            return engine
        finally:
            await engine.close()

    engine = run(go())
    assert sorted(sorted(b["questions"]) for b in api.rows) == [["c1"], ["c2"], ["p3"]]
    assert engine.columns["c2"].values[0] == ("mortgage", 0.8)
    assert engine.columns["p3"].values[0] == (2.0, 0.7)
    snapshot = engine.snapshot()
    assert [c["id"] for c in snapshot["columns"]] == ["c1", "c2"]  # a preview is not part of the table
    assert sorted(snapshot["cells"]) == [["c1", 0, 0.9, 0.9], ["c2", 0, "mortgage", 0.8]]


def test_columns_added_together_share_their_calls():
    api = FakeAPI(delay=0.01)

    async def go():
        engine, queue = await started(TEXTS, api)
        try:
            engine.commit("alleges fraud?")
            engine.commit("product: mortgage, credit card, other")
            await until(queue, "done")
            await until(queue, "done")
        finally:
            await engine.close()

    run(go())
    assert len(api.rows) == 5
    assert all(sorted(b["questions"]) == ["c1", "c2"] for b in api.rows)


def test_text_is_cut_to_max_chars_before_it_is_sent():
    api = FakeAPI()

    async def go():
        engine, queue = await started(["0123456789"], api, max_chars=4)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
        finally:
            await engine.close()

    run(go())
    assert [b["state"] for b in api.rows] == ["0123"]


def test_a_failed_call_is_swept_up_again():
    api = FakeAPI(fail={TEXTS[3]: 1})

    async def go():
        engine, queue = await started(TEXTS, api, attempts=1)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            return engine
        finally:
            await engine.close()

    engine = run(go())
    assert engine.errors == 1 and len(engine.columns["c1"].values) == 6
    assert [b["state"] for b in api.rows].count(TEXTS[3]) == 2


def test_a_fatal_error_reaches_the_page_and_stops_the_hot_lane():
    api = FakeAPI(status=402)

    async def go():
        engine, queue = await started(TEXTS, api)
        try:
            engine.commit("alleges fraud?")
            message = await until(queue, "fatal")
            await asyncio.sleep(0.2)
            before = len(api.rows)
            engine.view([0, 1, 2])  # nothing more is asked, even for rows on screen
            await asyncio.sleep(0.1)
            return engine, message, before, len(api.rows)
        finally:
            await engine.close()

    engine, message, before, after = run(go())
    assert message == {"type": "fatal", "message": "openrouter said 402: scripted 402"}
    assert engine.over_budget and before == after and not engine.columns["c1"].values


def test_the_background_stops_at_the_budget():
    api = FakeAPI(cost=0.5)

    async def go():
        engine, queue = await started(TEXTS, api, budget=Budget(1.2), capacity=1)
        try:
            engine.commit("alleges fraud?")
            message = await until(queue, "budget")
            await asyncio.sleep(0.2)
            return engine, message
        finally:
            await engine.close()

    engine, message = run(go())
    # The first call learns the price; a second fits beside it and a third would pass the limit.
    assert message == {"type": "budget", "budget": 1.2}
    assert engine.over_budget and engine.stats()["over_budget"]
    assert engine.client.budget.spent == 1.0 and len(engine.columns["c1"].values) == 2


def test_answers_come_back_from_the_cache(tmp_path):
    async def fill(api, store):
        engine, queue = await started(TEXTS, api, store=store)
        try:
            engine.commit("alleges fraud?")
            await until(queue, "done")
            return engine
        finally:
            await engine.close()

    store = AnswerStore(tmp_path / "answers.sqlite")
    paid, free = FakeAPI(delay=0.01), FakeAPI()
    try:
        first = run(fill(paid, store))
        again = run(fill(free, store))
    finally:
        store.close()
    assert len(paid.rows) == 5 and not free.rows
    assert again.cached == 6 and again.columns["c1"].values == first.columns["c1"].values


def test_a_joint_read_comes_back_from_the_cache_only_as_the_same_call(tmp_path):
    """DiffusionGemma answers each question in the light of the others in its call, so the cache serves an
    answer only to the same call."""
    async def fill(store, headers):
        api = FakeAPI(delay=0.01)
        engine, queue = await started(TEXTS, api, backend=JOINT, store=store)
        try:
            for header in headers:
                engine.commit(header)
            for _ in headers:
                await until(queue, "done")
            return api
        finally:
            await engine.close()

    both = ["alleges fraud?", "product: mortgage, credit card, other"]
    store = AnswerStore(tmp_path / "answers.sqlite")
    try:
        first = run(fill(store, both))
        again = run(fill(store, both))
        alone = run(fill(store, both[:1]))
    finally:
        store.close()
    assert len(first.rows) == 5 and all(sorted(b["questions"]) == ["c1", "c2"] for b in first.rows)
    assert not again.rows
    assert len(alone.rows) == 5 and all(list(b["questions"]) == ["c1"] for b in alone.rows)


def test_cached_answers_say_who_gave_them_and_the_stats_come_from_the_meter(tmp_path):
    api, store = FakeAPI(), AnswerStore(tmp_path / "answers.sqlite")

    async def go():
        engine, queue = await started(TEXTS, api, store=store)
        try:
            engine.visible = [0, 1]
            engine.commit("tone: calm < upset < furious")
            await until(queue, "done")
            return engine
        finally:
            await engine.close()

    try:
        engine = run(go())
        question = engine.columns["c1"].spec.question
        entries = [store.entry(answer_key(BACKEND, text, question)) for text in set(TEXTS)]
    finally:
        store.close()
    assert len(entries) == 5
    for entry in entries:
        assert entry.metadata["provider"] == "openrouter" and entry.metadata["requested_model"] == "jev-test"
        assert entry.metadata["resolved_model"] == "typesafe/jev-test"
    assert [engine.columns["c1"].values[r] for r in range(6)] == [(2.0, 0.7), (0.0, 0.7), (1.0, 0.7), (0.0, 0.7),
                                                                  (0.0, 0.7), (1.0, 0.7)]
    stats = engine.stats()
    assert (stats["calls"], stats["model"], stats["inflight"], stats["errors"]) == (5, "typesafe/jev-test", 0, 0)
    assert stats["dollars"] == round(5 * 0.0001, 5)


def test_removing_a_column_tells_the_page():
    async def go():
        engine, queue = await started(TEXTS, FakeAPI())
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

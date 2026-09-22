"""The client against a fake Decisions endpoint: keys, the cache, retries, shared and re-sent calls. No network."""

import asyncio
import json
import socket

import httpx
import pytest

from jcol import core
from jcol.core import BACKENDS, Cache, Jev, JevError, JevFatal, cache_path, resolve_backend

from fakes import FakeAPI

Q = {"q": {"type": "noul", "instructions": 'The text fits this description: "alleges fraud"'}}


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 20))


def test_the_tests_cannot_reach_the_network():
    with pytest.raises(RuntimeError, match="tests must stay offline"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)  # refused by conftest before a packet leaves


# ---- keys ------------------------------------------------------------------------------------------------

def test_the_key_comes_from_the_environment(monkeypatch):
    backend, key = resolve_backend()
    assert (backend.name, key) == ("openrouter", "test-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", " ts-key\n")
    backend, key = resolve_backend()
    assert (backend.name, key) == ("typesafe", "ts-key")  # with both, TypeSafe's own API
    assert resolve_backend("openrouter")[0].name == "openrouter"
    monkeypatch.setenv("JEV_API", "openrouter")
    assert resolve_backend()[0].name == "openrouter"


def test_the_key_comes_from_a_file(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    (tmp_path / "config" / "jev").mkdir(parents=True)
    (tmp_path / "config" / "jev" / "openrouter.key").write_text("file-key\n")
    backend, key = resolve_backend()
    assert (backend.name, key) == ("openrouter", "file-key")
    assert backend.key_file == tmp_path / "config" / "jev" / "openrouter.key"


def test_no_key_is_fatal_and_says_what_to_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(JevFatal, match="TYPESAFE_API_KEY or OPENROUTER_API_KEY"):
        resolve_backend()
    with pytest.raises(JevFatal, match="no key for typesafe"):
        resolve_backend("typesafe")
    with pytest.raises(JevFatal, match="unknown API"):
        resolve_backend("gateway")


# ---- the cache -------------------------------------------------------------------------------------------

def test_cache_keys_are_exact():
    q = Q["q"]
    key = Cache.key("jev-latest", "some text", q)
    assert key == Cache.key("jev-latest", "some text", dict(reversed(list(q.items()))))
    assert len({key, Cache.key("jev-1.13", "some text", q), Cache.key("jev-latest", "some text ", q),
                Cache.key("jev-latest", "some text", q | {"instructions": "something else"})}) == 4
    assert Cache.key("m", "t", q, endpoint="https://one.test") != Cache.key("m", "t", q, endpoint="https://two.test")


def test_the_cache_lives_under_xdg_cache_home_and_round_trips(tmp_path):
    assert cache_path() == tmp_path / "cache" / "jev" / "answers.sqlite"
    cache = Cache()
    try:
        assert cache.get("k") is None
        cache.put("k", {"noul": 0.25})
        cache.put("k", {"noul": 0.75})
        assert cache.get("k") == {"noul": 0.75}
    finally:
        cache.db.close()
    again = Cache(tmp_path / "cache" / "jev" / "answers.sqlite")
    try:
        assert again.get("k") == {"noul": 0.75}
    finally:
        again.db.close()


# ---- calls -----------------------------------------------------------------------------------------------

def test_a_call_sends_the_model_state_and_questions_and_is_metered():
    seen = []

    async def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"model": "typesafe/jev-test", "answers": {"q": {"noul": 0.9}},
                                         "usage": {"input_tokens": 300, "cost": 0.0001}})

    async def go():
        jev = Jev("test-key", transport=httpx.MockTransport(handler))
        try:
            assert await jev.ask("this is fraud", Q) == {"q": {"noul": 0.9}}
            return jev.meter
        finally:
            await jev.close()

    meter = run(go())
    (request,) = seen
    assert str(request.url) == BACKENDS["openrouter"].url
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {"model": "~typesafe/jev-latest", "state": "this is fraud", "questions": Q}
    assert (meter.calls, meter.input_tokens, meter.model) == (1, 300, "typesafe/jev-test")
    assert meter.cost == pytest.approx(0.0001)


def test_tokens_are_priced_when_the_api_reports_no_cost():
    api = FakeAPI(report_cost=False)

    async def go():
        jev = Jev("test-key", "typesafe", transport=api.transport)
        try:
            await jev.ask("text", Q)
            return jev.meter.cost
        finally:
            await jev.close()

    assert run(go()) == pytest.approx(300 * core.PRICE_PER_MTOK / 1e6)
    assert api.bodies[0]["model"] == "jev-latest"


def test_cached_answers_are_not_asked_again(tmp_path):
    api, cache = FakeAPI(), Cache(tmp_path / "answers.sqlite")
    other = {"r": {"type": "noul", "instructions": 'The text fits this description: "asks for a refund"'}}

    async def go():
        jev = Jev("test-key", cache=cache, transport=api.transport)
        try:
            first = await jev.ask("this is fraud", Q)
            assert await jev.ask("this is fraud", Q) == first
            await jev.ask("this is fraud", Q | other)  # only the new question travels
            return jev.meter
        finally:
            await jev.close()

    try:
        meter = run(go())
    finally:
        cache.db.close()
    assert [list(b["questions"]) for b in api.bodies] == [["q"], ["r"]]
    assert (meter.calls, meter.cached) == (2, 1)


def test_identical_requests_in_the_air_share_one_call():
    api = FakeAPI(delay=0.05)

    async def go():
        jev = Jev("test-key", transport=api.transport)
        try:
            return await asyncio.gather(*(jev.ask("this is fraud", Q) for _ in range(5))), jev.meter
        finally:
            await jev.close()

    results, meter = run(go())
    assert results == [{"q": {"noul": 0.9}}] * 5
    assert len(api.bodies) == 1 and (meter.calls, meter.cached) == (1, 4)


def test_a_retryable_status_is_retried():
    api = FakeAPI(script=[503])

    async def go():
        jev = Jev("test-key", transport=api.transport)
        try:
            return await jev.ask("this is fraud", Q), jev.meter.retries
        finally:
            await jev.close()

    assert run(go()) == ({"q": {"noul": 0.9}}, 1)
    assert len(api.bodies) == 2


@pytest.mark.parametrize("status, error", [(401, JevFatal), (402, JevFatal), (400, JevError)])
def test_other_statuses_are_not_retried(status, error):
    api = FakeAPI(script=[status])

    async def go():
        jev = Jev("test-key", transport=api.transport)
        try:
            await jev.ask("text", Q)
        finally:
            await jev.close()

    with pytest.raises(error, match=f"scripted {status}"):
        run(go())
    assert len(api.bodies) == 1


def test_an_answer_that_is_missing_is_an_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"answers": {}}))

    async def go():
        jev = Jev("test-key", transport=transport)
        try:
            await jev.ask("text", Q)
        finally:
            await jev.close()

    with pytest.raises(JevError, match="no answer returned for question 'q'"):
        run(go())


def test_a_slow_call_is_sent_again_and_the_first_answer_wins():
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            await asyncio.sleep(5)  # the straggler; cancelled when the client closes
        return httpx.Response(200, json={"answers": {"q": {"noul": 0.9}}, "usage": {"cost": 0.0001}})

    async def go():
        jev = Jev("test-key", transport=httpx.MockTransport(handler))
        try:
            quick = await jev.ask("this is fraud", Q, hedge_after=0.05)
            hedges = jev.meter.hedges
            await jev.ask("another text", Q, hedge_after=0.5)  # answered in time, so not re-sent
            return quick, hedges, jev.meter.hedges
        finally:
            for task in list(jev._flights.values()):
                task.cancel()
            await jev.close()

    assert run(go()) == ({"q": {"noul": 0.9}}, 1, 1)
    assert len(calls) == 3

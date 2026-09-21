"""Stand-ins for the Decisions endpoint and for the pool of workers. No network, no key."""

import asyncio
import json

import httpx


def answer(state: str, q: dict) -> dict:
    """A text is a yes when it contains 'fraud'; its category is the first option it mentions, else the last;
    its position on a scale is the number of exclamation marks in it."""
    if q["type"] == "noul":
        return {"noul": 0.9 if "fraud" in state else 0.1}
    if q["type"] == "choice":
        options = list(q["criteria"])
        label = next((o for o in options if o in state), options[-1])
        return {"choice": label, "probabilities": {o: 0.8 if o == label else 0.2 / (len(options) - 1) for o in options}}
    return {"score": float(min(state.count("!"), len(q["criteria"]) - 1)), "confidence": 0.7}


class FakeAPI:
    """An httpx transport handler. `script` is a list of status codes to return before answering normally."""

    def __init__(self, *, cost=0.0001, report_cost=True, script=(), delay=0.0):
        self.bodies, self.cost, self.report_cost, self.script, self.delay = [], cost, report_cost, list(script), delay

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.script and (status := self.script.pop(0)) != 200:
            return httpx.Response(status, json={"error": {"message": f"scripted {status}"}})
        usage = {"input_tokens": 300} | ({"cost": self.cost} if self.report_cost else {})
        found = {qid: answer(body["state"], q) for qid, q in body["questions"].items()}
        return httpx.Response(200, json={"model": "typesafe/jev-test", "answers": found, "usage": usage})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def rows(self) -> list[dict]:
        """The requests that carried a row of the table, which leaves out a pool's warm-up call."""
        return [b for b in self.bodies if b["state"] != "warm up"]


class FakePool:
    """What the engine needs from a pool, answered on the next turn of the event loop.

    `fail` maps a text to how many times a call about it should fail before it succeeds, or to an error
    message that ends the run."""

    def __init__(self, *, capacity=64, cost=0.0, fail=None):
        self.capacity, self.on_result, self.cost, self.fail = capacity, None, cost, dict(fail or {})
        self.jobs, self.spent = [], 0.0

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def submit(self, job) -> None:
        self.jobs.append(job)
        asyncio.get_running_loop().call_soon(self._finish, job)

    def _finish(self, job) -> None:
        job_id, state, questions, hot = job
        self.spent += self.cost
        pending = self.fail.get(state)
        if isinstance(pending, str):
            return self.on_result(job_id, None, pending)
        if pending:
            self.fail[state] = pending - 1
            return self.on_result(job_id, None, "gave up after 12s (HTTP 503)")
        self.on_result(job_id, {qid: answer(state, q) for qid, q in questions.items()}, None)

    def totals(self) -> dict:
        return {"calls": len(self.jobs), "hedges": 0, "retries": 0, "tokens": 0, "cost": self.spent, "model": ""}

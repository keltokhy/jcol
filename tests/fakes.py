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
    """An httpx transport handler. `script` is a list of status codes to return before answering normally;
    `fail` maps a text to how many times a call about it fails with 503 first; `status` refuses every call."""

    def __init__(self, *, cost=0.0001, report_cost=True, script=(), delay=0.0, fail=None, status=None):
        self.bodies, self.cost, self.report_cost, self.script, self.delay = [], cost, report_cost, list(script), delay
        self.fail, self.status = dict(fail or {}), status

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.status is not None:
            return httpx.Response(self.status, json={"error": {"message": f"scripted {self.status}"}})
        if self.script and (status := self.script.pop(0)) != 200:
            return httpx.Response(status, json={"error": {"message": f"scripted {status}"}})
        if self.fail.get(body["state"]):
            self.fail[body["state"]] -= 1
            return httpx.Response(503, json={"error": {"message": "scripted 503"}})
        usage = {"input_tokens": 300} | ({"cost": self.cost} if self.report_cost else {})
        found = {qid: answer(body["state"], q) for qid, q in body["questions"].items()}
        return httpx.Response(200, json={"model": "typesafe/jev-test", "answers": found, "usage": usage})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def rows(self) -> list[dict]:
        """The requests that carried a row of the table, which leaves out a warm-up call."""
        return [b for b in self.bodies if b["state"] != "warm up"]

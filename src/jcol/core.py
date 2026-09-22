"""jcol compatibility adapter for the shared JevKit implementation.

Prompts, cache identity, answer reuse, and budget policy retain their existing contracts.
Transport, configuration, storage, validation, and usage parsing come from jevkit_core.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from jevkit_core import (
    FATAL,
    RETRYABLE,
    PRICE_PER_MTOK,
    AnswerCache,
    Backend as _Backend,
    DecisionClient,
    JevBudgetExceeded,
    JevError,
    JevFatal,
    Meter as _Meter,
    backend_catalog,
    cache_path,
    config_dir,
    digest,
    parse_usage,
    resolve_backend as _resolve_backend,
)


Backend = _Backend


BACKENDS = backend_catalog("typesafe", "openrouter")


def resolve_backend(name: str | None = None):
    return _resolve_backend(BACKENDS, name)


class Cache(AnswerCache):
    """Keep the existing cache identity while sharing its storage implementation."""

    metadata = False

    @staticmethod
    def key(model: str, state, question: dict, *, endpoint: str | None = None) -> str:
        return digest([endpoint, model, state, question])


@dataclass
class Meter(_Meter):
    hedges: int = 0


class Jev(DecisionClient):
    def __init__(
        self,
        key: str,
        backend: Backend | str = "openrouter",
        *,
        model: str | None = None,
        timeout: float = 15.0,
        attempts: int = 4,
        concurrency: int = 32,
        cache: Cache | None = None,
        transport=None,
    ):
        backend = BACKENDS[backend] if isinstance(backend, str) else backend
        meter = Meter()
        super().__init__(
            key,
            backend,
            model=model,
            timeout=timeout,
            attempts=attempts,
            concurrency=concurrency,
            cache=cache,
            transport=transport,
            meter=meter,
            http2=True,
        )

    async def ask(
        self, state, questions: dict[str, dict], *, hedge_after: float | None = None
    ) -> dict[str, dict]:
        """Answer every question about one state. Only questions missing from the cache are sent.

        With `hedge_after`, a call still unanswered after that many seconds is sent a second time and the
        first answer wins. It trims the slow tail of a screenful at the price of a few duplicate calls.
        """
        keys = {
            qid: Cache.key(self.model, state, q, endpoint=self.url)
            for qid, q in questions.items()
        }
        answers = {}
        if self.cache:
            for qid, k in keys.items():
                if (hit := self.cache.get(k)) is not None:
                    answers[qid] = hit
        misses = {qid: q for qid, q in questions.items() if qid not in answers}
        if not misses:
            self.meter.cached += 1
            return answers

        task, _ = self.share_request(
            (keys[qid] for qid in misses), lambda: self._call(state, misses)
        )
        by_key = await (
            task
            if hedge_after is None
            else self._hedged(task, state, misses, hedge_after)
        )
        return answers | {qid: by_key[keys[qid]] for qid in misses}

    async def _hedged(
        self, first: asyncio.Task, state, questions: dict[str, dict], after: float
    ) -> dict[str, dict]:
        done, _ = await asyncio.wait({first}, timeout=after)
        if done:
            return first.result()
        second = asyncio.ensure_future(self._call(state, questions))
        self.meter.hedges += 1
        pending = {first, second}
        error: Exception | None = None
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if t.exception() is None:
                    if second in pending:
                        second.cancel()  # never cancel `first`: other callers may be sharing it
                    return t.result()
                error = t.exception()
        raise error

    def _record(
        self, state, questions: dict, data: dict, seconds: float
    ) -> dict[str, dict]:
        usage = parse_usage(data.get("usage"), price_per_mtok=PRICE_PER_MTOK)
        self.meter.record(usage, seconds, model=data.get("model") or self.model)
        if not isinstance(data["answers"], dict):
            raise JevError("answers must be an object")
        out = {}
        for qid, q in questions.items():
            if qid not in data["answers"]:
                raise JevError(f"no answer returned for question {qid!r}")
            k = Cache.key(self.model, state, q, endpoint=self.url)
            out[k] = data["answers"][qid]
            if self.cache:
                self.cache.put(k, out[k])
        return out


__all__ = [
    "BACKENDS",
    "Backend",
    "Cache",
    "Meter",
    "Jev",
    "JevError",
    "JevFatal",
    "JevBudgetExceeded",
    "PRICE_PER_MTOK",
    "RETRYABLE",
    "FATAL",
    "cache_path",
    "config_dir",
    "resolve_backend",
]

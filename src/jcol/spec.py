"""A column header is a tiny language with three shapes, one per Jev question type.

    alleges fraud?                                   yes/no      -> a probability
    product: mortgage, credit card, other            categories  -> a label and its probability
    tone: calm < upset < furious                     a scale     -> a position on it
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Spec:
    kind: str  # noul, choice or score
    name: str
    options: tuple[str, ...] = field(default=())
    definition: str = ""
    descriptions: tuple[str, ...] = field(default=())

    @property
    def question(self) -> dict:
        if self.kind == "noul":
            return {"type": "noul", "instructions": self.definition or f'The text fits this description: "{self.name}"'}
        if self.kind == "choice":
            return {"type": "choice", "instructions": self.definition or f"Choose the {self.name} that best describes the text.",
                    "criteria": dict(zip(self.options, self.descriptions or self.options))}
        return {"type": "score", "instructions": self.definition or f"Rate the text on this scale: {self.name}.",
                "criteria": list(self.options)}

    def cell(self, answer: dict) -> tuple[float | str, float]:
        """(value, confidence) for the table. Yes/no columns show the probability itself."""
        if not isinstance(answer, dict):
            raise ValueError("answer must be an object")
        if self.kind == "noul":
            p = float(answer["noul"])
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("binary probability must be between 0 and 1")
            return round(p, 3), round(max(p, 1 - p), 3)
        if self.kind == "choice":
            label = answer["choice"]
            if label not in self.options:
                raise ValueError(f"unknown category: {label!r}")
            probabilities = answer.get("probabilities") or {}
            if not isinstance(probabilities, dict):
                raise ValueError("probabilities must be an object")
            p = probabilities.get(label, answer.get("confidence") or 0.0)
            if not math.isfinite(float(p)) or not 0 <= float(p) <= 1:
                raise ValueError("confidence must be between 0 and 1")
            return label, round(float(p), 3)
        if not math.isfinite(float(answer["score"])) or not 0 <= float(answer["score"]) <= len(self.options) - 1:
            raise ValueError("score is outside its scale")
        if not math.isfinite(float(answer.get("confidence") or 0.0)) or not 0 <= float(answer.get("confidence") or 0.0) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return round(float(answer["score"]), 2), round(float(answer.get("confidence") or 0.0), 3)


def parse(header: str) -> Spec | None:
    """The Spec a header describes, or None while it is still too short or malformed to ask."""
    h = " ".join(header.split())
    if ":" not in h:
        name = h.rstrip("?").strip()
        return Spec("noul", name) if len(name) >= 3 else None
    name, rest = (part.strip() for part in h.split(":", 1))
    if not name or not rest:
        return None
    if "<" in rest:
        levels = tuple(x.strip() for x in rest.split("<") if x.strip())
        return Spec("score", name, levels) if 2 <= len(levels) <= 10 and len(set(levels)) == len(levels) else None
    options = tuple(dict.fromkeys(x.strip() for x in rest.split(",") if x.strip()))
    return Spec("choice", name, options) if 2 <= len(options) <= 255 else None

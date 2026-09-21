"""A column header is a tiny language with three shapes, one per Jev question type.

    alleges fraud?                                   yes/no      -> a probability
    product: mortgage, credit card, other            categories  -> a label and its probability
    tone: calm < upset < furious                     a scale     -> a position on it
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    kind: str  # noul, choice or score
    name: str
    options: tuple[str, ...] = field(default=())

    @property
    def question(self) -> dict:
        if self.kind == "noul":
            return {"type": "noul", "instructions": f'The text fits this description: "{self.name}"'}
        if self.kind == "choice":
            return {"type": "choice", "instructions": f"Choose the {self.name} that best describes the text.",
                    "criteria": {o: o for o in self.options}}
        return {"type": "score", "instructions": f"Rate the text on this scale: {self.name}.",
                "criteria": list(self.options)}

    def cell(self, answer: dict) -> tuple[float | str, float]:
        """(value, confidence) for the table. Yes/no columns show the probability itself."""
        if self.kind == "noul":
            p = float(answer["noul"])
            return round(p, 3), round(max(p, 1 - p), 3)
        if self.kind == "choice":
            label = answer["choice"]
            p = (answer.get("probabilities") or {}).get(label, answer.get("confidence") or 0.0)
            return label, round(float(p), 3)
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

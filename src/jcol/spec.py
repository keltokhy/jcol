"""A column header is a tiny language with three shapes, one per Jev question type.

    alleges fraud?                                   yes/no      -> a probability
    product: mortgage, credit card, other            categories  -> a label and its probability
    tone: calm < upset < furious                     a scale     -> a position on it
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jevkit_runtime import Choice, Noul, Question, Score


@dataclass(frozen=True)
class Spec:
    kind: str  # noul, choice or score
    name: str
    options: tuple[str, ...] = field(default=())
    definition: str = ""
    descriptions: tuple[str, ...] = field(default=())

    @property
    def question(self) -> Question:
        """The runtime question this column asks: it validates every answer, options and bounds included."""
        if self.kind == "noul":
            return Noul(self.definition or f'The text fits this description: "{self.name}"')
        if self.kind == "choice":
            return Choice(self.definition or f"Choose the {self.name} that best describes the text.",
                          dict(zip(self.options, self.descriptions or self.options)))
        return Score(self.definition or f"Rate the text on this scale: {self.name}.", self.options)

    def cell(self, answer: dict) -> tuple[float | str, float]:
        """(value, confidence) for the table. Yes/no columns show the probability itself."""
        question = self.question
        question.validate(answer)
        confidence = question.confidence(answer)
        value = question.value(answer)
        if self.kind == "noul":
            return round(value, 3), round(confidence, 3)
        if self.kind == "choice":
            return value, round(confidence or 0.0, 3)
        return round(value, 2), round(confidence or 0.0, 3)


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

"""Versioned codebooks: definitions and explicit input fields, independent of the browser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .spec import Spec


@dataclass(frozen=True)
class Variable:
    spec: Spec
    gold: str | None = None


@dataclass(frozen=True)
class Codebook:
    inputs: tuple[str, ...]
    variables: tuple[Variable, ...]

    @classmethod
    def load(cls, value: Codebook | dict | str | Path) -> Codebook:
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).read_text())
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("codebook must be an object with version: 1")
        _keys(value, {"version", "inputs", "columns"}, "codebook")
        inputs = _names(value.get("inputs"), "inputs", 1, 255)
        columns = value.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError("columns must be a nonempty list")
        variables = []
        for column in columns:
            if not isinstance(column, dict):
                raise ValueError("each column must be an object")
            _keys(column, {"name", "kind", "definition", "options", "gold"}, "column")
            name, definition = column.get("name"), column.get("definition")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("each column needs a nonempty name")
            if not isinstance(definition, str) or not definition.strip():
                raise ValueError(f"{name}: a nonempty definition is required")
            raw_kind = column.get("kind")
            kind = {"binary": "noul", "category": "choice", "scale": "score"}.get(raw_kind) if isinstance(raw_kind, str) else None
            if kind is None:
                raise ValueError(f"{name}: kind must be binary, category or scale")
            options, descriptions = (), ()
            if kind != "noul":
                raw = column.get("options")
                if kind == "choice" and isinstance(raw, dict):
                    descriptions = tuple(raw.values())
                    if any(not isinstance(d, str) or not d.strip() for d in descriptions):
                        raise ValueError(f"{name}: option definitions must be nonempty strings")
                    raw = list(raw)
                options = _names(raw, f"{name} options", 2, 10 if kind == "score" else 255)
            elif "options" in column:
                raise ValueError(f"{name}: binary columns do not take options")
            gold = column.get("gold")
            if gold is not None and (not isinstance(gold, str) or not gold.strip()):
                raise ValueError(f"{name}: gold must name a label column")
            if gold in inputs:
                raise ValueError(f"{name}: gold column {gold!r} must not be a model input")
            variables.append(Variable(Spec(kind, name, options, definition, descriptions), gold))
        names = [v.spec.name for v in variables]
        outputs = [n for name in names for n in (name, name + "__confidence")]
        if len(set(outputs)) != len(outputs):
            raise ValueError("column names and generated confidence names must be unique")
        return cls(inputs, tuple(variables))

    def to_dict(self) -> dict:
        columns = []
        for variable in self.variables:
            spec = variable.spec
            column = {"name": spec.name, "kind": {"noul": "binary", "choice": "category", "score": "scale"}[spec.kind],
                      "definition": spec.definition}
            if spec.options:
                column["options"] = dict(zip(spec.options, spec.descriptions)) if spec.descriptions else list(spec.options)
            if variable.gold is not None:
                column["gold"] = variable.gold
            columns.append(column)
        return {"version": 1, "inputs": list(self.inputs), "columns": columns}


def _keys(value: dict, allowed: set, where: str) -> None:
    if extra := value.keys() - allowed:
        raise ValueError(f"unknown {where} fields: {', '.join(sorted(extra))}")


def _names(value, where: str, minimum: int, maximum: int) -> tuple[str, ...]:
    if (not isinstance(value, list) or not minimum <= len(value) <= maximum
            or any(not isinstance(v, str) or not v.strip() for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{where} must contain {minimum} to {maximum} distinct nonempty names")
    return tuple(value)

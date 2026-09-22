"""Offline validation against user-supplied labels, with explicit coverage."""

from __future__ import annotations

import math

import polars as pl

from .codebook import Codebook


def evaluate(table: pl.DataFrame, codebook: Codebook | dict | str, *, threshold: float = 0.5) -> dict:
    book = Codebook.load(codebook)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    report = {"rows": table.height, "threshold": threshold, "columns": {},
              "interpretation": "Agreement with supplied labels; label quality and representativeness are not established."}
    for variable in book.variables:
        if variable.gold is None:
            continue
        spec = variable.spec
        for name in (spec.name, variable.gold):
            if name not in table.columns:
                raise ValueError(f"evaluation needs column {name!r}")
        labeled, pairs, disagreements = 0, [], []
        for row, (gold, prediction) in enumerate(zip(table[variable.gold], table[spec.name])):
            if _missing(gold):
                continue
            labeled += 1
            expected = _label(gold, spec, gold=True)
            if _missing(prediction):
                continue
            observed = _label(prediction, spec, gold=False)
            predicted = int(observed >= threshold) if spec.kind == "noul" else observed
            pairs.append((expected, predicted, observed))
            if expected != predicted:
                disagreements.append({"row_index": row, "expected": expected, "predicted": predicted})
        n = len(pairs)
        metrics = {"gold": variable.gold, "labeled": labeled, "evaluated": n, "missing_predictions": labeled - n,
                   "coverage": n / labeled if labeled else None, "disagreements": disagreements}
        if spec.kind == "score":
            metrics["mae"] = sum(abs(g - p) for g, p, _ in pairs) / n if n else None
        else:
            labels = [0, 1] if spec.kind == "noul" else list(spec.options)
            confusion = {str(g): {str(p): 0 for p in labels} for g in labels}
            for g, p, _ in pairs:
                confusion[str(g)][str(p)] += 1
            classes = {}
            for label in labels:
                true_positive = sum(g == label and p == label for g, p, _ in pairs)
                support = sum(g == label for g, _, _ in pairs)
                predicted_count = sum(p == label for _, p, _ in pairs)
                classes[str(label)] = {"support": support,
                                       "precision": true_positive / predicted_count if predicted_count else 0.0,
                                       "recall": true_positive / support if support else 0.0,
                                       "f1": 2 * true_positive / (support + predicted_count) if support + predicted_count else 0.0}
            metrics.update(accuracy=sum(g == p for g, p, _ in pairs) / n if n else None,
                           macro_f1=sum(c["f1"] for c in classes.values()) / len(classes) if n else None,
                           classes=classes, confusion=confusion)
            if spec.kind == "noul":
                metrics["brier_score"] = sum((g - probability) ** 2 for g, _, probability in pairs) / n if n else None
        report["columns"][spec.name] = metrics
    return report


def _missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, float) and math.isnan(value))


def _label(value, spec, *, gold: bool):
    if spec.kind == "choice":
        if value not in spec.options:
            raise ValueError(f"{spec.name}: unknown {'gold' if gold else 'predicted'} category {value!r}")
        return value
    if spec.kind == "noul" and gold:
        lookup = {"true": 1, "yes": 1, "1": 1, "1.0": 1, "false": 0, "no": 0, "0": 0, "0.0": 0}
        if str(value).strip().lower() not in lookup:
            raise ValueError(f"{spec.name}: gold binary labels must be true/false, yes/no or 1/0")
        return lookup[str(value).strip().lower()]
    if spec.kind == "score" and value in spec.options:
        return float(spec.options.index(value))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{spec.name}: expected a numeric value, got {value!r}") from exc
    maximum = 1 if spec.kind == "noul" else len(spec.options) - 1
    if not math.isfinite(number) or not 0 <= number <= maximum:
        raise ValueError(f"{spec.name}: value must be between 0 and {maximum}")
    return number

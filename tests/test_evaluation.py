import polars as pl
import pytest

from jcol import Codebook, evaluate


def book(kind="binary", **kw):
    return {"version": 1, "inputs": ["text"], "columns": [
        {"name": "prediction", "kind": kind, "definition": "Apply this rule.", "gold": "gold", **kw}]}


def test_known_binary_metrics_confusion_and_coverage():
    df = pl.DataFrame({"gold": [1, 0, 1, 0, 1, None], "prediction": [0.8, 0.9, 0.2, 0.1, None, 0.3]})
    metrics = evaluate(df, book())["columns"]["prediction"]
    assert (metrics["labeled"], metrics["evaluated"], metrics["coverage"], metrics["accuracy"]) == (5, 4, 0.8, 0.5)
    assert metrics["macro_f1"] == 0.5
    assert metrics["brier_score"] == pytest.approx(0.375)
    assert metrics["confusion"] == {"0": {"0": 1, "1": 1}, "1": {"0": 1, "1": 1}}
    assert [r["row_index"] for r in metrics["disagreements"]] == [1, 2]


def test_categories_and_scale_metrics():
    category = evaluate(pl.DataFrame({"gold": ["a", "b", "b"], "prediction": ["a", "a", "b"]}),
                        book("category", options=["a", "b"]))["columns"]["prediction"]
    assert category["accuracy"] == pytest.approx(2 / 3)
    assert category["macro_f1"] == pytest.approx(2 / 3)
    scale = evaluate(pl.DataFrame({"gold": ["calm", "furious"], "prediction": [0.5, 1.5]}),
                     book("scale", options=["calm", "upset", "furious"]))["columns"]["prediction"]
    assert scale["mae"] == 0.5


@pytest.mark.parametrize("change", [
    {"version": 2}, {"inputs": []}, {"inputs": ["gold"]}, {"typo": True},
    {"columns": []}, {"columns": [{"name": "x", "kind": "binary"}]},
    {"columns": [{"name": "x", "kind": "category", "definition": "Choose.", "options": ["a", "a"]}]}
])
def test_malformed_codebooks_are_rejected(change):
    with pytest.raises(ValueError):
        Codebook.load(book() | change)


def test_round_trip_and_output_name_collisions():
    source = book("category", options={"yes": "An explicit yes", "no": "No evidence"})
    assert Codebook.load(source).to_dict() == source
    source["columns"].append(source["columns"][0] | {"name": "prediction__confidence"})
    with pytest.raises(ValueError, match="unique"):
        Codebook.load(source)


def test_invalid_prediction_is_not_silently_discarded():
    with pytest.raises(ValueError, match="between"):
        evaluate(pl.DataFrame({"gold": [1], "prediction": [float("inf")]}), book())

"""End-to-end codebooks, resumable runs, exports and CLI behavior over a fake API."""

import asyncio
import json
import sys

import polars as pl
import pytest

import jcol
from jcol import batch
from jcol.cli import cli
from jcol.core import Jev
from jcol.project import Project
from jcol.tables import identity

from fakes import FakeAPI


BOOK = {"version": 1, "inputs": ["narrative", "context"], "columns": [
    {"name": "fraud_flag", "kind": "binary", "definition": "Does the row allege fraud?", "gold": "gold_fraud"},
    {"name": "product_code", "kind": "category", "definition": "Identify the product discussed.",
     "options": {"mortgage": "A home loan", "credit card": "A revolving card account", "other": "Anything else"}, "gold": "gold_product"},
    {"name": "tone_code", "kind": "scale", "definition": "Rate the tone.", "options": ["calm", "upset", "furious"]}
]}


@pytest.fixture
def df():
    return pl.DataFrame({"id": [8, 2, 3], "narrative": ["fraud!!", None, "nothing"],
                         "context": ["mortgage", "credit card", ""], "gold_fraud": [1, 0, 0],
                         "gold_product": ["mortgage", "credit card", "other"]})


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    monkeypatch.setattr(batch, "Jev", lambda backend, **kw: Jev(backend, transport=fake.transport, **kw))
    return fake


def test_codebook_definitions_multiple_fields_full_outputs_and_label_report(df, api, tmp_path):
    result = jcol.annotate(df, BOOK, cache=False)
    assert result.complete
    assert result.table.select(df.columns).equals(df)
    assert result.table["fraud_flag"].to_list() == [0.9, 0.1, 0.1]
    assert result.table["product_code"].to_list() == ["mortgage", "credit card", "other"]
    assert result.table["tone_code"].to_list() == [2.0, 0.0, 0.0]
    assert len(api.rows) == 3 and all(len(body["questions"]) == 3 for body in api.rows)
    for body in api.rows:
        state = json.loads(body["state"])
        assert set(state) == {"narrative", "context"}
        assert body["questions"]["c1"]["instructions"] == BOOK["columns"][0]["definition"]
        assert body["questions"]["c2"]["criteria"] == BOOK["columns"][1]["options"]
    metrics = result.evaluation["columns"]["fraud_flag"]
    assert (metrics["accuracy"], metrics["coverage"], metrics["evaluated"]) == (1.0, 1.0, 3)
    for suffix in ("parquet", "csv", "tsv"):
        path = tmp_path / f"out.{suffix}"
        result.write(path)
        assert path.exists()
    assert pl.read_parquet(tmp_path / "out.parquet").equals(result.table)


def test_saved_columns_survive_restart_without_cache_or_new_calls(df, api, tmp_path):
    path = tmp_path / "run.jcol.sqlite"
    first = jcol.annotate(df, BOOK, project=path, cache=False)
    api.bodies.clear()
    second = jcol.annotate(df, BOOK, project=path, cache=False)
    assert second.table.equals(first.table)
    assert second.complete and not api.bodies  # no warm-up either
    assert second.metadata["stats"]["calls"] == 0


def test_project_survives_dataframe_to_parquet_round_trip(df, api, tmp_path):
    # Slicing and nulls can leave different Arrow buffer layouts after a disk round trip.
    table = df[[2, 0, 1]]
    project = tmp_path / "roundtrip.jcol.sqlite"
    first = jcol.annotate(table, BOOK, project=project, cache=False)
    path = tmp_path / "roundtrip.parquet"
    table.write_parquet(path)
    api.bodies.clear()
    second = jcol.annotate(path, BOOK, project=project, cache=False)
    assert second.table.equals(first.table) and not api.bodies


def test_fingerprint_is_independent_of_buffers_on_the_shipped_sample(tmp_path):
    from pathlib import Path

    sample = pl.read_parquet(Path(__file__).parents[1] / "data/complaints-5k.parquet").sample(n=100, seed=70, shuffle=True)
    path = tmp_path / "sample.parquet"
    sample.write_parquet(path)
    assert identity(sample, ("narrative",)) == identity(pl.read_parquet(path), ("narrative",))


def test_fingerprint_retains_nanoseconds_and_binary_data(tmp_path):
    first = pl.DataFrame({"text": ["a"], "time": [1], "bytes": [b"\x00\xff"]}).with_columns(pl.col("time").cast(pl.Datetime("ns")))
    second = first.with_columns(pl.lit(2).cast(pl.Datetime("ns")).alias("time"))
    assert identity(first, ("text",)) != identity(second, ("text",))
    first = first.with_columns(pl.struct("time").alias("nested"))
    second = second.with_columns(pl.struct("time").alias("nested")).with_columns(first["time"])
    assert identity(first, ("text",)) != identity(second, ("text",))
    first.write_parquet(tmp_path / "temporal.parquet")
    assert identity(first, ("text",)) == identity(pl.read_parquet(tmp_path / "temporal.parquet"), ("text",))


def test_partial_budget_run_resumes_only_missing_cells(df, api, tmp_path):
    path = tmp_path / "run.jcol.sqlite"
    first = jcol.annotate(df, BOOK, project=path, budget=0.0001, concurrency=1, cache=False)
    assert not first.complete
    assert len(api.rows) == 1
    assert first.evaluation["columns"]["fraud_flag"]["coverage"] == pytest.approx(1 / 3)
    old_states = {body["state"] for body in api.rows}
    api.bodies.clear()
    second = jcol.annotate(df, BOOK, project=path, concurrency=1, cache=False)
    assert second.complete and len(api.rows) == 2
    assert not old_states.intersection(body["state"] for body in api.rows)


def test_project_rejects_changed_rows_codebook_model_or_truncation(df, api, tmp_path):
    path = tmp_path / "run.jcol.sqlite"
    jcol.annotate(df, BOOK, project=path, cache=False)
    calls = len(api.bodies)
    changed = json.loads(json.dumps(BOOK))
    changed["columns"][0]["definition"] += " Only explicit allegations."
    for table, book, options in [(df.reverse(), BOOK, {}), (df, changed, {}),
                                 (df, BOOK, {"model": "different"}), (df, BOOK, {"max_chars": 10})]:
        with pytest.raises(ValueError, match="does not match"):
            jcol.annotate(table, book, project=path, cache=False, **options)
    assert len(api.bodies) == calls


def test_project_is_exclusive_and_reopens_after_close(df, tmp_path):
    path = tmp_path / "project.sqlite"
    context = identity(df, ("narrative",))
    project = Project(path, context)
    try:
        with pytest.raises(ValueError, match="already open"):
            Project(path, context)
    finally:
        project.close()
    Project(path, context).close()


def test_full_text_is_default_and_truncation_is_explicit(api):
    table = pl.DataFrame({"narrative": ["x" * 4500 + "fraud"]})
    book = {"version": 1, "inputs": ["narrative"], "columns": [BOOK["columns"][0] | {"gold": None}]}
    assert jcol.annotate(table, book, cache=False).table["fraud_flag"][0] == 0.9
    with pytest.warns(UserWarning, match="1 rows will be truncated"):
        result = jcol.annotate(table, book, cache=False, max_chars=4000)
    assert result.table["fraud_flag"][0] == 0.1
    assert result.table["narrative"][0] == table["narrative"][0]
    assert result.metadata["truncated_rows"] == 1


def test_empty_input_finishes_with_typed_output_and_no_calls(df, api):
    result = jcol.annotate(df.head(0), BOOK, cache=False)
    assert result.complete and result.table.height == 0 and not api.bodies
    assert result.table.schema["product_code"] == pl.String
    assert result.evaluation["columns"]["fraud_flag"]["accuracy"] is None


@pytest.mark.parametrize("options", [{"budget": -1}, {"budget": float("nan")}, {"concurrency": 0}, {"max_chars": 0}])
def test_invalid_settings_fail_before_calls(df, api, options):
    with pytest.raises(ValueError):
        jcol.annotate(df, BOOK, **options)
    assert not api.bodies


def test_existing_output_or_invalid_gold_fails_before_calls(df, api):
    with pytest.raises(ValueError, match="already exist"):
        jcol.annotate(df.with_columns(pl.lit(1).alias("fraud_flag")), BOOK)
    with pytest.raises(ValueError, match="gold binary labels"):
        jcol.annotate(df.with_columns(pl.lit(2).alias("gold_fraud")), BOOK)
    assert not api.bodies


def test_async_notebook_entry_point(df, api):
    async def go():
        with pytest.raises(RuntimeError, match="annotate_async"):
            jcol.annotate(df, BOOK)
        return await jcol.annotate_async(df, BOOK, cache=False)
    assert asyncio.run(go()).complete


def test_cli_run_and_offline_evaluate(df, api, tmp_path, monkeypatch):
    source, book, out, report = [tmp_path / name for name in ("source.parquet", "book.json", "out.parquet", "report.json")]
    df.write_parquet(source)
    book.write_text(json.dumps(BOOK))
    monkeypatch.setattr(sys, "argv", ["jcol", "run", str(source), "--codebook", str(book), "--output", str(out), "--report", str(report)])
    cli()
    assert json.loads(report.read_text())["run"]["complete"]
    assert (tmp_path / "out.parquet.jcol.sqlite").exists()
    api.bodies.clear()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr(sys, "argv", ["jcol", "evaluate", str(out), "--codebook", str(book), "--report", str(report)])
    cli()
    assert json.loads(report.read_text())["columns"]["product_code"]["accuracy"] == 1
    assert not api.bodies


def test_cli_partial_exit_is_nonzero_and_keeps_output(df, api, tmp_path, monkeypatch):
    source, book, out = [tmp_path / name for name in ("source.csv", "book.json", "out.csv")]
    df.write_csv(source)
    book.write_text(json.dumps(BOOK))
    monkeypatch.setattr(sys, "argv", ["jcol", "run", str(source), "--codebook", str(book), "--output", str(out), "--budget", "0"])
    with pytest.raises(SystemExit) as stop:
        cli()
    assert stop.value.code == 2 and out.exists()
    assert not api.bodies


def test_cli_refuses_to_overwrite_source(df, api, tmp_path, monkeypatch):
    source, book = tmp_path / "source.csv", tmp_path / "book.json"
    df.write_csv(source)
    original = source.read_bytes()
    book.write_text(json.dumps(BOOK))
    monkeypatch.setattr(sys, "argv", ["jcol", "run", str(source), "--codebook", str(book), "--output", str(source)])
    with pytest.raises(SystemExit) as stop:
        cli()
    assert stop.value.code == 1 and source.read_bytes() == original and not api.bodies


def test_fatal_failure_returns_partial_results_without_hanging(df, api):
    api.script = [402]
    result = jcol.annotate(df, BOOK, concurrency=1, cache=False)
    assert not result.complete and "402" in result.metadata["fatal"]
    assert len(api.bodies) == 1


def test_malformed_answers_finish_with_missing_cells_instead_of_hanging(df, monkeypatch):
    import httpx

    async def malformed(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {k: {"noul": 2} for k in body["questions"]}})

    monkeypatch.setattr(batch, "Jev", lambda backend, **kw: Jev(backend, transport=httpx.MockTransport(malformed), **kw))

    async def go():
        return await asyncio.wait_for(jcol.annotate_async(df, BOOK, cache=False), timeout=5)

    result = asyncio.run(go())
    assert not result.complete and result.metadata["stats"]["errors"] > 0


def test_committed_cells_survive_abrupt_process_exit(tmp_path):
    import subprocess

    path = tmp_path / "crash.jcol.sqlite"
    code = '''
import os, sys
from jcol.project import Project
from jcol.engine import Column
from jcol.spec import Spec
project = Project(sys.argv[1], {"test": 1})
column = Column("c1", "alleges fraud?", Spec("noul", "alleges fraud"), True)
project.add(column)
project.fill(column, 0, {"noul": 0.9})
os._exit(0)
'''
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)
    reopened = Project(path, {"test": 1})
    try:
        assert reopened.columns()[0].values == {0: (0.9, 0.9)}
    finally:
        reopened.close()


@pytest.mark.parametrize("usage", [[], "invalid", {"cost": "0.1"}, {"cost": -1},
                                   {"input_tokens": "300"}, {"input_tokens": -1}, {"cost": True},
                                   {"cost": 10 ** 400}, {"input_tokens": 10 ** 400}])
def test_invalid_usage_stops_batch_without_hanging_or_sending_more_rows(df, monkeypatch, usage):
    import httpx

    calls = []

    async def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {qid: {"noul": 0.5} for qid in body["questions"]}, "usage": usage})

    monkeypatch.setattr(batch, "Jev", lambda backend, **kw: Jev(backend, transport=httpx.MockTransport(handler), **kw))

    async def go():
        return await asyncio.wait_for(jcol.annotate_async(df, BOOK, cache=False, concurrency=1), timeout=2)

    result = asyncio.run(go())
    assert not result.complete and "usage metadata" in result.metadata["fatal"]
    assert len(calls) == 1

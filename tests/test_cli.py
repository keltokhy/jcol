"""Public shell contracts: streams, offline operations, failure recovery and output safety."""

import io
import json
import os
import sys

import polars as pl
import pytest

from jcol import batch
from jcol.cli import cli
from jevkit_runtime import Client
from jcol.project import Project
from jcol.tables import identity, write_table
from fakes import FakeAPI


BOOK = {"version": 1, "inputs": ["text"], "columns": [
    {"name": "fraud", "kind": "binary", "definition": "Does this allege fraud?"}]}


@pytest.fixture
def source(tmp_path):
    source, book = tmp_path / "input.csv", tmp_path / "book.json"
    source.write_text('id,text\n1,"fraud, café"\n2,ordinary\n3,other\n', encoding="utf-8")
    book.write_text(json.dumps(BOOK))
    return source, book


@pytest.fixture
def api(monkeypatch):
    api = FakeAPI()
    monkeypatch.setattr(batch, "Client", lambda backend, **kw: Client(backend, transport=api.transport, **kw))
    return api


def invoke(*args):
    cli([str(a) for a in args])


def failure(*args, code=1):
    with pytest.raises(SystemExit) as stop:
        invoke(*args)
    assert stop.value.code == code


def test_root_help_does_not_launch_browser(capsys):
    invoke()
    output = capsys.readouterr()
    for command in ("run", "validate", "inspect", "status", "export", "doctor", "browse", "init"):
        assert command in output.out
    assert not output.err
    failure("rnu", "--json")
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "usage_error"


@pytest.mark.parametrize("args", [("--json", "doctor"), ("doctor", "--json")])
def test_doctor_no_key_is_parseable_and_offline(args, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    invoke(*args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["schema_version"] == 1
    assert not payload["data"]["ready"]
    assert not payload["data"]["reachability"]["checked"]


def test_doctor_reports_key_source_without_key_or_url_credentials(monkeypatch, capsys):
    monkeypatch.setenv("JEV_URL", "https://user:password@example.com/api?token=secret")
    invoke("doctor", "--json")
    output = capsys.readouterr().out
    assert "password" not in output and "secret" not in output and "test-key" not in output
    data = json.loads(output)["data"]
    assert data["credentials"][1]["source"] == "env" and data["ready"]
    assert data["endpoint"] == "https://example.com/api"


def test_init_emits_reusable_codebook_and_refuses_accidental_overwrite(tmp_path, capsys):
    args = ("init", "--input", "subject", "body", "--name", "topic", "--definition", "Classify topic",
            "--kind", "category", "--option", "billing", "--option", "other")
    invoke(*args)
    book = json.loads(capsys.readouterr().out)
    assert book["inputs"] == ["subject", "body"] and book["columns"][0]["options"] == ["billing", "other"]
    path = tmp_path / "book.json"
    invoke(*args, "-o", path)
    capsys.readouterr()
    before = path.read_bytes()
    failure(*args, "-o", path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("format,content", [("csv", "text,id\nhello,1\n"), ("tsv", "text\tid\nhello\t1\n"),
                                          ("jsonl", '{"text":"hello","id":1}\n')])
def test_inspect_stdin(format, content, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(content))
    invoke("inspect", "-", "--input-format", format, "--json")
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["rows"] == 1 and data["columns"][0] == {"name": "text", "type": "String", "nulls": 0}


def test_stdin_requires_explicit_format(capsys):
    failure("inspect", "-", "--json")
    assert "--input-format" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_argument_errors_do_not_contaminate_table_stdout(capsys):
    failure("run", "missing.csv", "-o", "-", "--json", "--unknown")
    output = capsys.readouterr()
    assert not output.out and json.loads(output.err)["error"]["code"] == "usage_error"


def test_escaped_credentials_are_redacted_before_json_encoding(monkeypatch, capsys):
    key = 'sensitive-"key\\suffix'
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setenv("JEV_API", key)
    failure("doctor", "--json")
    data = json.loads(capsys.readouterr().out)
    assert key not in data["error"]["message"] and "[redacted]" in data["error"]["message"]


def test_validate_and_dry_run_need_no_key_and_write_nothing(source, tmp_path, monkeypatch, capsys):
    src, book = source
    monkeypatch.delenv("OPENROUTER_API_KEY")
    before = set(tmp_path.iterdir())
    invoke("validate", src, "--codebook", book, "--json", "--max-chars", 4)
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["cells"] == 3 and data["truncated_rows"] == 3
    invoke("run", src, "--codebook", book, "--dry-run", "-o", tmp_path / "out.csv", "--report", tmp_path / "report.json")
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert set(tmp_path.iterdir()) == before


def test_validation_and_paid_run_reject_same_output_collision(source, tmp_path, api, capsys):
    src, book = source
    pl.read_csv(src).with_columns(pl.lit(1).alias("fraud")).write_csv(src)
    for command in ("validate", "run"):
        args = ("-o", tmp_path / "out.csv") if command == "run" else ()
        failure(command, src, "--codebook", book, *args, "--json")
        assert "already exist" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert not api.bodies


@pytest.mark.parametrize("format", ["csv", "tsv", "jsonl"])
def test_run_stdout_is_only_table_data(source, format, api, capsys):
    src, book = source
    invoke("run", src, "--codebook", book, "-o", "-", "--output-format", format,
           "--no-project", "--no-cache", "--quiet", "--json")
    output = capsys.readouterr()
    table = (pl.read_ndjson(io.BytesIO(output.out.encode())) if format == "jsonl"
             else pl.read_csv(io.StringIO(output.out), separator="\t" if format == "tsv" else ","))
    assert table.height == 3 and table["fraud"].to_list() == [0.9, 0.1, 0.1]
    assert json.loads(output.err)["data"]["run"]["complete"]


def test_stdout_requires_explicit_persistence_choice(source, api, capsys):
    src, book = source
    failure("run", src, "--codebook", book, "-o", "-", "--json")
    output = capsys.readouterr()
    assert not output.out and "--project" in json.loads(output.err)["error"]["message"]
    assert not api.bodies


def test_partial_resume_status_and_offline_export(source, tmp_path, api, monkeypatch, capsys):
    src, book = source
    output, project = tmp_path / "coded.csv", tmp_path / "saved.sqlite"
    args = ("run", src, "--codebook", book, "-o", output, "--project", project, "--no-cache", "--concurrency", 1)
    failure(*args, "--budget", 0.0001, code=2)
    capsys.readouterr()
    invoke("status", project, "--json")
    state = json.loads(capsys.readouterr().out)["data"]
    assert not state["complete"] and state["columns"][0]["filled"] == 1
    first = set(b["state"] for b in api.bodies)
    api.bodies.clear()
    invoke(*args)
    capsys.readouterr()
    assert len(api.bodies) == 2 and not first.intersection(b["state"] for b in api.bodies)
    api.bodies.clear()
    invoke(*args)
    assert not api.bodies
    capsys.readouterr()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    exported = tmp_path / "recovered.parquet"
    invoke("export", project, "--source", src, "-o", exported, "--json")
    assert json.loads(capsys.readouterr().out)["data"]["complete"]
    assert pl.read_parquet(exported).equals(pl.read_csv(output))
    assert not api.bodies
    pl.read_csv(src).reverse().write_csv(src)
    failure("export", project, "--source", src, "-o", tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()


def test_dry_run_checks_saved_identity_without_calls(source, tmp_path, api, capsys):
    src, book = source
    out = tmp_path / "out.csv"
    invoke("run", src, "--codebook", book, "-o", out)
    capsys.readouterr()
    api.bodies.clear()
    invoke("run", src, "--codebook", book, "-o", out, "--dry-run")
    assert json.loads(capsys.readouterr().out)["resume"]
    failure("run", src, "--codebook", book, "-o", out, "--dry-run", "--model", "changed")
    assert not api.bodies


def test_read_active_project_and_export_before_columns_are_committed(source, tmp_path, capsys):
    src, book = source
    df = pl.read_csv(src)
    path = tmp_path / "project.sqlite"
    project = Project(path, identity(df, ("text",), codebook=BOOK))
    try:
        invoke("status", path)
        data = json.loads(capsys.readouterr().out)
        assert not data["complete"] and data["columns"][0]["missing"] == 3
        invoke("export", path, "--source", src, "-o", tmp_path / "out.csv")
        assert pl.read_csv(tmp_path / "out.csv")["fraud"].null_count() == 3
    finally:
        project.close()


@pytest.mark.parametrize("suffix", ["", ".lock", "-wal", "-shm"])
def test_cannot_overwrite_project_or_sqlite_sidecars(source, tmp_path, api, suffix):
    src, book = source
    path = tmp_path / "saved.sqlite"
    failure("run", src, "--codebook", book, "--project", path, "-o", str(path) + suffix, "--output-format", "csv")
    assert not api.bodies and not path.exists()


def test_hardlink_to_source_is_rejected_before_calls(source, tmp_path, api):
    src, book = source
    link = tmp_path / "output.csv"
    os.link(src, link)
    original = src.read_bytes()
    failure("run", src, "--codebook", book, "-o", link)
    assert src.read_bytes() == original and not api.bodies


def test_failed_serialization_preserves_previous_file(tmp_path):
    output = tmp_path / "keep.csv"
    output.write_text("previous complete output")
    with pytest.raises(pl.exceptions.PolarsError):
        write_table(pl.DataFrame({"nested": [[1, 2]]}), output)
    assert output.read_text() == "previous complete output"
    assert list(tmp_path.iterdir()) == [output]


def test_corrupt_inputs_return_json_errors_without_traceback(tmp_path, capsys):
    invalid = tmp_path / "invalid.parquet"
    invalid.write_text("not parquet")
    failure("inspect", invalid, "--json")
    output = capsys.readouterr()
    assert not json.loads(output.out)["ok"] and "Traceback" not in output.err
    failure("status", invalid, "--json")
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_missing_project_is_not_created(tmp_path, capsys):
    path = tmp_path / "missing.sqlite"
    failure("status", path, "--json")
    assert not path.exists()
    assert "no such project" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_interrupt_exit_and_machine_error_redaction(source, monkeypatch, capsys):
    import importlib
    module = importlib.import_module("jcol.cli")
    src, book = source

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "annotate", interrupt)
    failure("run", src, "--codebook", book, "-o", "-", "--no-project", "--json", code=130)
    output = capsys.readouterr()
    assert not output.out and json.loads(output.err)["error"]["code"] == "interrupted"
    monkeypatch.setenv("JEV_API", "test-key")
    failure("doctor", "--json")
    output = capsys.readouterr().out
    assert "test-key" not in output and "[redacted]" in output

"""The server and the command line, in-process over a fake API. Nothing listens on a port and nothing is opened."""

import sys

import polars as pl
import pytest
from starlette.testclient import TestClient

import jcol
from jcol import app as jcol_app
from jcol.core import Jev

from fakes import FakeAPI

ROWS = {"id": [1, 2, 3, 4], "product": ["mortgage", "credit card", "credit card", None],
        "narrative": ["this is fraud about my mortgage!!", "a late fee on my credit card",
                      "fraud again, a credit card!", None]}


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "complaints.csv"
    pl.DataFrame(ROWS).write_csv(path)
    return path


@pytest.fixture
def api(monkeypatch):
    """Whatever `build` constructs talks to a fake; a real server or browser window is an error."""
    fake = FakeAPI()
    monkeypatch.setattr(jcol_app, "Jev", lambda key, backend, **kw: Jev(key, backend, transport=fake.transport, **kw))

    def refuse(*args, **kw):
        raise AssertionError("a test tried to start worker processes")

    monkeypatch.setattr(jcol_app, "ProcessPool", refuse)
    return fake


@pytest.fixture
def no_server(monkeypatch):
    served = []
    monkeypatch.setattr(jcol_app.uvicorn, "run", lambda app, **kw: served.append(kw))
    monkeypatch.setattr(jcol_app.webbrowser, "open", lambda url: served.append(url))
    return served


# ---- reading a table -------------------------------------------------------------------------------------

def test_the_text_column_defaults_to_the_one_with_the_most_text(table):
    df, text = jcol_app.load_table(table, None, None)
    assert text == "narrative" and df.height == 4
    assert df["narrative"].to_list()[3] is None  # preserve source nulls; serialize missing inputs as empty text
    df, text = jcol_app.load_table(table, "product", 2)
    assert text == "product" and df.height == 2


def test_parquet_and_tsv_are_read_too(tmp_path):
    pl.DataFrame(ROWS).write_parquet(tmp_path / "t.parquet")
    pl.DataFrame(ROWS).write_csv(tmp_path / "t.tsv", separator="\t")
    for name in ("t.parquet", "t.tsv"):
        df, text = jcol_app.load_table(tmp_path / name, None, None)
        assert (text, df.columns) == ("narrative", ["id", "product", "narrative"])


def test_a_table_that_cannot_be_used_is_refused(table, tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    pl.DataFrame({"a": [1, 2]}).write_csv(tmp_path / "numbers.csv")
    for path, column, message in [(tmp_path / "missing.csv", None, "no such file"),
                                  (tmp_path / "notes.txt", None, "table format must be"),
                                  (tmp_path / "numbers.csv", None, "no text column"),
                                  (table, "story", "no column 'story'")]:
        with pytest.raises(SystemExit, match=message):
            jcol_app.load_table(path, column, None)


# ---- the server ------------------------------------------------------------------------------------------

def serve(table, **kw):
    df, text = jcol_app.load_table(table, None, None)
    options = dict(source=table.name, budget=2.0, api=None, model=None, cache=False, workers=0, per_worker=8)
    return TestClient(jcol_app.build(df, text, **(options | kw)))


def test_the_page_and_the_table_are_served(table, api):
    with serve(table) as client:
        page = client.get("/")
        assert page.status_code == 200 and "<title>jcol</title>" in page.text
        assert page.headers["cache-control"] == "no-store"
        init = client.get("/api/init").json()
        assert client.get("/api/row/1").json() == {"id": 2, "product": "credit card",
                                                   "narrative": "a late fee on my credit card"}
        assert client.get("/api/row/4").status_code == 404
    assert (init["version"], init["source"], init["n"], init["text"]) == (jcol.__version__, "complaints.csv", 4, "narrative")
    assert init["fields"] == ["id", "product"] and sorted(init["order"]) == [0, 1, 2, 3]
    assert init["values"] == {"id": ["1", "2", "3", "4"], "product": ["mortgage", "credit card", "credit card", ""]}
    assert init["previews"] == ROWS["narrative"][:3] + [""]
    assert init["stats"]["budget"] == 2.0 and init["stats"]["model"] == "typesafe/jev-test"  # what the API said answered


def test_long_texts_reach_the_page_as_previews(tmp_path, api):
    path = tmp_path / "long.csv"
    pl.DataFrame({"narrative": ["x" * 1000]}).write_csv(path)
    with serve(path) as client:
        assert client.get("/api/init").json()["previews"] == ["x" * jcol_app.PREVIEW_CHARS]
        assert client.get("/api/row/0").json() == {"narrative": "x" * 1000}


def test_a_column_is_previewed_committed_filled_and_removed_over_the_socket(table, api):
    def until(ws, kind):
        while (message := ws.receive_json())["type"] != kind:
            pass
        return message

    def cells(ws, column, count):  # a column's cells can arrive over several batches
        found = []
        while len(found) < count:
            found += [c for c in until(ws, "cells")["cells"] if c[0] == column]
        return sorted(found)

    with serve(table) as client, client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "snapshot", "columns": [], "cells": [],
                                     "stats": client.get("/api/init").json()["stats"]}
        ws.send_json({"type": "preview", "header": "alleges fraud", "rows": [0, 1]})
        ghost = until(ws, "preview")["column"]
        assert (ghost["id"], ghost["kind"], ghost["committed"]) == ("p1", "noul", False)
        assert cells(ws, "p1", 2) == [["p1", 0, 0.9, 0.9], ["p1", 1, 0.1, 0.9]]

        ws.send_json({"type": "commit", "header": "product: mortgage, credit card, other", "rows": [0, 1]})
        column = until(ws, "column")["column"]
        assert (column["id"], column["name"], column["options"]) == ("c2", "product", ["mortgage", "credit card", "other"])
        assert until(ws, "done")["id"] == "c2"

        with client.websocket_connect("/ws") as late:  # a page that connects later is caught up
            snapshot = late.receive_json()
        assert [c["id"] for c in snapshot["columns"]] == ["c2"]
        assert sorted(snapshot["cells"]) == [["c2", 0, "mortgage", 0.8], ["c2", 1, "credit card", 0.8],
                                             ["c2", 2, "credit card", 0.8], ["c2", 3, "other", 0.8]]
        ws.send_json({"type": "remove", "id": "c2"})
        assert until(ws, "removed") == {"type": "removed", "id": "c2"}

    assert api.bodies[0]["state"] == "warm up"
    assert {b["model"] for b in api.bodies} == {"~typesafe/jev-latest"}
    assert all(set(b) == {"model", "state", "questions"} for b in api.bodies)


# ---- the command -----------------------------------------------------------------------------------------

def test_help_and_version(monkeypatch, capsys):
    for flag in ("--help", "--version"):
        monkeypatch.setattr(sys, "argv", ["jcol", flag])
        with pytest.raises(SystemExit) as stop:
            jcol_app.cli()
        assert stop.value.code == 0
    out = capsys.readouterr().out
    assert "Open a table in the browser and add columns by describing them." in out
    assert f"jcol {jcol.__version__}" in out
    for option in ("--text", "--limit", "--budget", "--port", "--api", "--model", "--no-cache", "--workers",
                   "--per-worker", "--no-open"):
        assert option in out


def test_the_command_says_what_it_read_and_serves_on_localhost(table, api, no_server, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["jcol", str(table), "--workers", "0", "--budget", "0.5", "--port", "9999"])
    jcol_app.cli()
    assert capsys.readouterr().err == ("jcol: 4 rows of complaints.csv; reading column 'narrative'; budget $0.50\n"
                                       "jcol: http://127.0.0.1:9999\n")
    assert no_server == ["http://127.0.0.1:9999", {"host": "127.0.0.1", "port": 9999, "log_level": "warning"}]
    assert not api.bodies  # nothing is asked until the server starts


def test_without_a_key_the_command_stops_before_it_serves(table, api, no_server, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr(sys, "argv", ["jcol", str(table), "--workers", "0", "--no-open"])
    with pytest.raises(SystemExit, match="jcol: no API key. Set TYPESAFE_API_KEY or OPENROUTER_API_KEY"):
        jcol_app.cli()
    assert not api.bodies and not no_server


def test_a_missing_file_is_reported(monkeypatch, tmp_path, no_server):
    monkeypatch.setattr(sys, "argv", ["jcol", str(tmp_path / "nope.parquet")])
    with pytest.raises(SystemExit, match="jcol: no such file"):
        jcol_app.cli()
    assert not no_server


def test_restart_restores_saved_columns_and_removal_is_durable(table, api, tmp_path):
    project = tmp_path / "table.jcol.sqlite"
    with serve(table, project=project) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "commit", "header": "alleges fraud?", "rows": []})
        while ws.receive_json()["type"] != "done":
            pass
    rows = len(api.rows)
    with serve(table, project=project) as client, client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["columns"][0]["header"] == "alleges fraud?"
        assert len(snapshot["cells"]) == 4 and len(api.rows) == rows
        ws.send_json({"type": "remove", "id": "c1"})
        while ws.receive_json()["type"] != "removed":
            pass
    with serve(table, project=project) as client, client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["columns"] == []


def test_export_preserves_long_text_types_nulls_and_requested_order(tmp_path, api):
    import io

    df = pl.DataFrame({"narrative": ["x" * 5000, None, "fraud"], "id": [1, 2, 3]})
    path = tmp_path / "table.parquet"
    df.write_parquet(path)
    with serve(path) as client:
        response = client.post("/api/export", json={"rows": [2, 0, 1]})
        exported = pl.read_csv(io.StringIO(response.text))
        assert response.status_code == 200 and exported.equals(df[[2, 0, 1]])
        empty = pl.read_csv(io.StringIO(client.post("/api/export", json={"rows": []}).text))
        assert empty.height == 0 and empty.columns == df.columns
        assert client.post("/api/export", json={"rows": [-1]}).status_code == 400


def test_multiple_browser_inputs_are_named_and_exports_resolve_source_name_collisions(table, api):
    import io

    df, inputs = jcol_app.load_table(table, ["narrative", "product"], None)
    app = jcol_app.build(df, inputs, source=table.name, budget=2, api=None, model=None,
                         cache=False, workers=0, per_worker=8)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.get("/api/init").json()["inputs"] == ["narrative", "product"]
        ws.send_json({"type": "commit", "header": "product: mortgage, credit card, other", "rows": []})
        while ws.receive_json()["type"] != "done":
            pass
        output = pl.read_csv(io.StringIO(client.post("/api/export", json={}).text))
        assert output.select(df.columns).equals(df)
        assert "jcol_c1__product" in output.columns
        assert output["jcol_c1__product"][0] == "mortgage"

"""Durable columns and raw answers, bound to exact data and inference settings."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .spec import Spec


class Project:
    def __init__(self, path: str | Path, identity: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A separate connection holds a writer lock while the data connection commits
        # each result. The OS releases this lock after a crash; no stale PID files.
        self.lock = sqlite3.connect(str(self.path) + ".lock", timeout=0, check_same_thread=False)
        try:
            self.lock.execute("CREATE TABLE IF NOT EXISTS owner (id INTEGER)")
            self.lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            self.lock.close()
            if "locked" not in str(exc):
                raise
            raise ValueError(f"project is already open: {self.path}") from exc
        try:
            self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS columns (id TEXT PRIMARY KEY, header TEXT NOT NULL, "
                            "spec TEXT NOT NULL, seconds REAL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS cells (col TEXT, row INTEGER, answer TEXT NOT NULL, "
                            "PRIMARY KEY (col, row))")
            encoded = json.dumps(identity, sort_keys=True)
            found = self.db.execute("SELECT value FROM meta WHERE key = 'identity'").fetchone()
            if found and found[0] != encoded:
                raise ValueError("project does not match this table or inference settings; use a new project path")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('identity', ?)", (encoded,))
        except BaseException:
            self.close()
            raise

    def columns(self):
        return _columns(self.db)

    def add(self, column) -> None:
        self.db.execute("INSERT INTO columns VALUES (?, ?, ?, ?)",
                        (column.id, column.header, json.dumps(asdict(column.spec)), column.seconds))

    def fill(self, column, row: int, answer: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO cells VALUES (?, ?, ?)",
                        (column.id, row, json.dumps(answer, allow_nan=False)))

    def done(self, column) -> None:
        self.db.execute("UPDATE columns SET seconds = ? WHERE id = ?", (column.seconds, column.id))

    def remove(self, cid: str) -> None:
        self.db.execute("BEGIN")
        try:
            self.db.execute("DELETE FROM cells WHERE col = ?", (cid,))
            self.db.execute("DELETE FROM columns WHERE id = ?", (cid,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def close(self) -> None:
        if hasattr(self, "db"):
            self.db.close()
        self.lock.close()


def _columns(db, *, values=True):
    from .engine import Column

    columns = []
    for cid, header, encoded, seconds in db.execute("SELECT id, header, spec, seconds FROM columns ORDER BY rowid"):
        fields = json.loads(encoded)
        fields["options"] = tuple(fields["options"])
        fields["descriptions"] = tuple(fields.get("descriptions", []))
        column = Column(cid, header, Spec(**fields), True, seconds=seconds)
        if values:
            for row, answer in db.execute("SELECT row, answer FROM cells WHERE col = ?", (cid,)):
                column.values[row] = column.spec.cell(json.loads(answer))
        columns.append(column)
    return columns


def read_project(path: str | Path, *, values: bool = False):
    """Read a consistent snapshot, including while a writer is running; never create a project."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"no such project: {path}")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
    try:
        db.execute("BEGIN")
        row = db.execute("SELECT value FROM meta WHERE key = 'identity'").fetchone()
        context = json.loads(row[0]) if row else None
        if (not isinstance(context, dict) or context.get("format") != 2
                or not isinstance(context.get("rows"), int) or context["rows"] < 0
                or not isinstance(context.get("inputs"), list)
                or not context["inputs"] or any(not isinstance(name, str) for name in context["inputs"])):
            raise ValueError("unsupported or invalid project identity")
        columns = _columns(db, values=values)
        counts = dict(db.execute("SELECT col, COUNT(*) FROM cells GROUP BY col"))
        if "codebook" in context:
            # A process can stop between creating the project and committing all variables.
            from .codebook import Codebook
            from .engine import Column

            existing = {c.spec.name for c in columns}
            for i, variable in enumerate(Codebook.load(context["codebook"]).variables):
                if variable.spec.name not in existing:
                    columns.append(Column(f"missing-{i}", variable.spec.name, variable.spec, True))
        return context, columns, counts
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid project data") from exc
    finally:
        db.close()

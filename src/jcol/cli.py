"""Terminal-first codebook annotation, composable streams and offline recovery."""

from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import polars as pl

from . import __version__
from .batch import annotate, prepare, validate_settings
from .codebook import Codebook
from jevkit_runtime import JevError, JevFatal, Settings
from .core import PROVIDERS
from .evaluation import evaluate
from .project import read_project
from .tables import FORMATS, append_columns, atomic_path, identity, read_table, table_format, write_table


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def _common(parser):
    # Suppressed defaults let global flags work before or after a subcommand.
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit a versioned JSON envelope (table streams remain raw)")
    parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS, help="suppress progress and summaries")


def _input(parser):
    parser.add_argument("file", help="source table, or - for stdin")
    parser.add_argument("--input-format", choices=FORMATS, help="required for stdin; otherwise inferred from extension")


def _output(parser, *, required=False):
    parser.add_argument("-o", "--output", required=required, help="destination table, or - for stdout")
    parser.add_argument("--output-format", choices=FORMATS, help="inferred from extension; stdout defaults to jsonl")


def _backend(parser):
    parser.add_argument("--api", choices=tuple(PROVIDERS), help="provider (or JEV_API; otherwise first configured key)")
    parser.add_argument("--model", help="model ID (or JEV_MODEL; otherwise provider default)")


def parser() -> Parser:
    root = Parser(prog="jcol", description="Turn natural-language codebooks into columns. Validate, run, resume and export.",
                  epilog="Start: jcol init --help. Offline checks: jcol doctor; jcol validate --help.")
    root.add_argument("--version", action="version", version=f"jcol {__version__}")
    _common(root)
    commands = root.add_subparsers(dest="command", metavar="COMMAND")

    def command(name, description):
        child = commands.add_parser(name, help=description, description=description)
        _common(child)
        return child

    p = command("init", "Create a valid codebook from a definition; no model calls.")
    p.add_argument("--input", nargs="+", required=True, metavar="FIELD", help="fields the model may read")
    p.add_argument("--name", required=True, help="output column name")
    p.add_argument("--definition", required=True, help="natural-language coding instructions")
    p.add_argument("--kind", choices=("binary", "category", "scale"), default="binary")
    p.add_argument("--option", action="append", help="repeat for category labels or ordered scale levels")
    p.add_argument("--gold", help="optional reviewed-label field, excluded from model inputs")
    p.add_argument("-o", "--output", default="-", help="codebook JSON path (default: stdout)")
    p.add_argument("--force", action="store_true", help="replace an existing codebook")

    p = command("inspect", "Show row count, field types and null counts; no model calls.")
    _input(p)

    p = command("validate", "Validate codebook, inputs, labels and output names offline.")
    _input(p)
    p.add_argument("--codebook", required=True, type=Path)
    p.add_argument("--max-chars", type=int, help="explicit per-row truncation limit")

    p = command("run", "Apply a codebook, checkpoint cells, and resume when repeated.")
    _input(p)
    _output(p)
    _backend(p)
    p.add_argument("--codebook", required=True, type=Path)
    saved = p.add_mutually_exclusive_group()
    saved.add_argument("--project", type=Path, help="checkpoint path (default: OUTPUT.jcol.sqlite)")
    saved.add_argument("--no-project", action="store_true", help="explicitly disable checkpointing")
    p.add_argument("--budget", type=float, default=2.0, help="per-invocation spending threshold in dollars (default: 2)")
    p.add_argument("--concurrency", type=int, default=32, help="maximum calls in flight (default: 32)")
    p.add_argument("--max-chars", type=int, help="truncate serialized rows; default sends full rows")
    p.add_argument("--no-cache", action="store_true", help="disable shared answer cache; retain project checkpoints")
    p.add_argument("--dry-run", action="store_true", help="validate and describe work without auth, calls or writes")
    p.add_argument("--report", type=Path, help="JSON report path; otherwise stdout, or stderr when streaming a table")

    p = command("status", "Read saved column completion, including during an active run.")
    p.add_argument("project", type=Path)

    p = command("export", "Recover saved results with the original source table; no API key needed.")
    p.add_argument("project", type=Path)
    p.add_argument("--source", required=True, dest="file", help="original table, or - for stdin")
    p.add_argument("--input-format", choices=FORMATS)
    _output(p, required=True)
    p.add_argument("--report", type=Path, help="optional JSON export report")

    p = command("evaluate", "Compare existing coded columns with reviewed labels offline.")
    _input(p)
    p.add_argument("--codebook", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--report", type=Path)

    p = command("doctor", "Check local configuration; optionally probe endpoint reachability.")
    _backend(p)
    p.add_argument("--check", action="store_true", help="send an unauthenticated HEAD request (no inference or billing)")

    command("browse", "Open the optional browser interface; jcol browse --help for options.")
    return root


def _read(a):
    if a.file == "-":
        if not a.input_format:
            raise ValueError("stdin requires --input-format")
        stream = getattr(sys.stdin, "buffer", None)
        if stream is None:
            stream = io.BytesIO(sys.stdin.read().encode("utf-8"))
        return read_table(stream, format=a.input_format)
    return read_table(a.file, format=a.input_format)


def _output_format(a):
    return table_format(a.output, a.output_format or ("jsonl" if a.output == "-" else None))


def _write(df, a):
    format = _output_format(a)
    if a.output == "-":
        target = getattr(sys.stdout, "buffer", sys.stdout) if format == "parquet" else sys.stdout
        if format == "parquet" and not hasattr(sys.stdout, "buffer"):
            raise ValueError("Parquet stdout requires a binary stream")
        write_table(df, target, format=format)
        target.flush()
    else:
        write_table(df, a.output, format=format)


def _paths(inputs=(), outputs=(), project=None):
    sources = [Path(p) for p in inputs if p is not None and str(p) != "-"]
    destinations = [Path(p) for p in outputs if p is not None and str(p) != "-"]
    if project is not None:
        # An output must never replace a live SQLite WAL or lock.
        sources += [Path(str(project) + suffix) for suffix in ("", ".lock", "-wal", "-shm")]
    paths = sources + destinations
    for i, path in enumerate(paths):
        for other in paths[:i]:
            if path.resolve() == other.resolve() or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError("input, codebook, output, project and report paths must be distinct")
    for path in destinations + ([Path(project)] if project else []):
        if not path.parent.is_dir():
            raise ValueError(f"output directory does not exist: {path.parent}")
        if path.is_dir():
            raise ValueError(f"expected a file, found directory: {path}")


def _redact(message):
    settings = Settings.from_env()
    if endpoint := settings.url:
        message = message.replace(endpoint, _endpoint(endpoint))
    for provider in PROVIDERS.values():
        try:
            key, _ = provider.credential(settings)
        except OSError:
            key = ""
        if key:
            for form in sorted({key, repr(key)[1:-1], json.dumps(key)[1:-1]}, key=len, reverse=True):
                message = message.replace(form, "[redacted]")
    return message


def _encoded(value):
    def clean(item):
        if isinstance(item, str):
            return _redact(item)
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, list):
            return [clean(value) for value in item]
        return item

    return json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _emit(a, data, *, table_stream=False):
    payload = {"schema_version": 1, "command": a.command, "ok": True, "data": data} if a.json else data
    encoded = _encoded(payload)
    report = getattr(a, "report", None)
    if report:
        with atomic_path(report) as temporary:
            temporary.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="", file=sys.stderr if table_stream else sys.stdout)


def _summary(a, text):
    if not a.quiet:
        print(f"jcol: {text}", file=sys.stderr)


def _validation(df, book, *, max_chars=None):
    df, book, inputs, texts, truncated = prepare(df, book, max_chars=max_chars)
    sent = [text[:max_chars] for text in texts] if max_chars is not None else texts
    return {"valid": True, "rows": df.height, "inputs": list(inputs),
            "outputs": [name for v in book.variables for name in (v.spec.name, v.spec.name + "__confidence")],
            "cells": df.height * len(book.variables), "unique_inputs": len(set(sent)),
            "truncated_rows": truncated, "max_chars": max_chars}


def _endpoint(url):
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "[invalid endpoint]"
    return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))


def _doctor(a):
    settings = Settings.from_env()
    configured = []
    for provider in PROVIDERS.values():
        key, source = provider.credential(settings)
        configured.append({"api": provider.name, "available": bool(key), "source": source if key else "missing",
                           "environment_variable": provider.key_env})
    requested = a.api or settings.api
    if requested and requested not in PROVIDERS:
        raise ValueError(f"unknown API {requested!r}")
    selected = requested or next((item["api"] for item in configured if item["available"]), "typesafe")
    provider = PROVIDERS[selected]
    ready = next(item["available"] for item in configured if item["api"] == selected)
    endpoint = provider.endpoint(settings)
    result = {"version": __version__, "ready": ready, "api": selected,
              "model": a.model or settings.model or provider.model,
              "endpoint": _endpoint(endpoint), "credentials": configured,
              "offline_commands": ["init", "inspect", "validate", "status", "export", "evaluate", "run --dry-run"],
              "reachability": {"checked": False},
              "next_step": None if ready else f"Set {provider.key_env} or put a key in {provider.key_file(settings)}"}
    if a.check:
        try:
            response = httpx.head(endpoint, timeout=10, follow_redirects=False)
            result["reachability"] = {"checked": True, "reachable": True, "status": response.status_code,
                                      "authentication_verified": False}
        except httpx.HTTPError:
            result["reachability"] = {"checked": True, "reachable": False, "authentication_verified": False}
    return result


def _status(project, context, columns, counts):
    expected = len(context["codebook"]["columns"]) if "codebook" in context else len(columns)
    fields = [{"name": c.spec.name, "filled": counts.get(c.id, 0),
               "missing": context["rows"] - counts.get(c.id, 0)} for c in columns]
    return {"project": str(project), "rows": context["rows"], "inputs": context["inputs"],
            "model": context.get("model"), "columns": fields,
            "complete": len(columns) == expected and all(c["missing"] == 0 for c in fields)}


def _run(a):
    if not a.output and not a.dry_run:
        raise ValueError("run requires --output (use - for stdout)")
    if a.output == "-" and not a.project and not a.no_project and not a.dry_run:
        raise ValueError("stdout requires --project PATH to checkpoint, or --no-project")
    if a.output:
        _output_format(a)
    a.project = a.project or (Path(str(a.output) + ".jcol.sqlite") if a.output and a.output != "-" and not a.no_project else None)
    _paths([a.file, a.codebook], [a.output, a.report], a.project)
    validate_settings(a.budget, a.concurrency, a.max_chars)
    df, book = _read(a), Codebook.load(a.codebook)
    if a.dry_run:
        data = _validation(df, book, max_chars=a.max_chars)
        config = _doctor(argparse.Namespace(api=a.api, model=a.model, check=False))
        data.update(dry_run=True, project=str(a.project) if a.project else None,
                    api=config["api"], model=config["model"], ready=config["ready"], budget=a.budget,
                    concurrency=a.concurrency, resume=False)
        if a.project and a.project.exists():
            context, columns, counts = read_project(a.project)
            expected = identity(df, book.inputs, model=config["model"],
                                endpoint=PROVIDERS[config["api"]].endpoint(Settings.from_env()),
                                max_chars=a.max_chars, codebook=book.to_dict())
            if context != expected:
                raise ValueError("project does not match this table or inference settings; use a new project path")
            data.update(resume=True, saved=_status(a.project, context, columns, counts))
        a.report = None  # Dry runs have no filesystem effects, including report writes.
        _emit(a, data)
        return 0

    def progress(state):
        _summary(a, f"{state['filled']}/{state['total']} cells; ${state['stats']['dollars']:.4f}")

    result = annotate(df, book, project=a.project, api=a.api, model=a.model, budget=a.budget,
                      concurrency=a.concurrency, max_chars=a.max_chars, cache=not a.no_cache,
                      progress=progress if sys.stderr.isatty() and not a.quiet else None)
    _write(result.table, a)
    _summary(a, f"{'complete' if result.complete else 'partial'}; wrote {a.output}; project {a.project or 'disabled'}")
    _emit(a, {"run": result.metadata, "evaluation": result.evaluation}, table_stream=a.output == "-")
    return 0 if result.complete else 2


def _dispatch(a):
    if a.command == "run":
        return _run(a)
    if a.command == "init":
        variable = {"name": a.name, "kind": a.kind, "definition": a.definition}
        if a.option is not None:
            variable["options"] = a.option
        if a.gold:
            variable["gold"] = a.gold
        book = Codebook.load({"version": 1, "inputs": a.input, "columns": [variable]}).to_dict()
        if a.output == "-":
            print(json.dumps(book, indent=2, ensure_ascii=False))
        else:
            _paths(outputs=[a.output])
            if Path(a.output).exists() and not a.force:
                raise ValueError("codebook already exists; use --force to replace it")
            with atomic_path(a.output) as temporary:
                temporary.write_text(json.dumps(book, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _emit(a, {"output": a.output, "codebook": book})
    elif a.command == "doctor":
        _emit(a, _doctor(a))
    elif a.command == "inspect":
        df = _read(a)
        _emit(a, {"rows": df.height, "columns": [{"name": n, "type": str(t), "nulls": df[n].null_count()}
                                                   for n, t in df.schema.items()]})
    elif a.command == "validate":
        _emit(a, _validation(_read(a), a.codebook, max_chars=a.max_chars))
    elif a.command == "status":
        context, columns, counts = read_project(a.project)
        _emit(a, _status(a.project, context, columns, counts))
    elif a.command == "export":
        _output_format(a)
        _paths([a.file], [a.output, a.report], a.project)
        df = _read(a)
        context, columns, counts = read_project(a.project, values=True)
        observed = identity(df, tuple(context["inputs"]))
        if any(context.get(key) != value for key, value in observed.items()):
            raise ValueError("source table does not match the saved project; use the original data, schema and row order")
        _write(append_columns(df, columns), a)
        status = _status(a.project, context, columns, counts)
        _emit(a, {**status, "output": a.output}, table_stream=a.output == "-")
        _summary(a, f"exported {'complete' if status['complete'] else 'partial'} results to {a.output}")
    elif a.command == "evaluate":
        _paths([a.file, a.codebook], [a.report])
        _emit(a, evaluate(_read(a), a.codebook, threshold=a.threshold))
    return 0


def cli(argv=None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    json_mode = "--json" in args
    a = None
    try:
        first = next((i for i, arg in enumerate(args) if arg not in ("--json", "--quiet")), None)
        command = args[first] if first is not None else None
        # Retain stream routing even when argument parsing fails.
        stream = any(arg in ("--output=-", "-o-") or (arg in ("--output", "-o") and args[i + 1:i + 2] == ["-"])
                     for i, arg in enumerate(args))
        a = argparse.Namespace(command=command, output="-" if stream or command == "init" else None)
        if first is not None and args[first] == "browse":
            if json_mode:
                raise ValueError("browse does not emit JSON; use headless commands with --json")
            from .app import cli as browser_cli
            return browser_cli(args[first + 1:])
        if args and not args[0].startswith("-") and Path(args[0]).suffix.lower() in (".parquet", ".csv", ".tsv", ".jsonl", ".ndjson"):
            if json_mode:
                raise ValueError("browser invocation does not emit JSON; use jcol run")
            from .app import cli as browser_cli
            return browser_cli(args)
        a = root.parse_args(args)
        a.json = getattr(a, "json", False)
        a.quiet = getattr(a, "quiet", False)
        if a.command is None:
            root.print_help()
            return
        code = _dispatch(a)
        if code:
            raise SystemExit(code)
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        raise SystemExit(141) from None
    except KeyboardInterrupt:
        _error("interrupted", "Interrupted; saved cells remain in the project. Repeat the run to resume or use export.", json_mode, a)
        raise SystemExit(130) from None
    except (ValueError, OSError, JevError, JevFatal, sqlite3.Error, pl.exceptions.PolarsError) as exc:
        _error("usage_error" if isinstance(exc, UsageError) else "command_error", str(exc), json_mode, a)
        raise SystemExit(1) from None


def _error(kind, message, json_mode, a):
    table_stream = a is not None and a.command in ("run", "export", "init") and getattr(a, "output", None) == "-"
    if json_mode:
        payload = {"schema_version": 1, "command": a.command if a else None,
                   "ok": False, "error": {"code": kind, "message": message}}
        print(_encoded(payload), end="", file=sys.stderr if table_stream else sys.stdout)
    else:
        print(f"jcol: {_redact(message)}", file=sys.stderr)

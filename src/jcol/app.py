"""jcol: open a table in the browser and add columns to it by describing them.

    jcol data/complaints-5k.parquet --text narrative

The server holds the API key, the data and the scheduler. The page is one static file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import webbrowser
from pathlib import Path

import polars as pl
import uvicorn
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from . import __version__
from .core import Cache, Jev, JevFatal, resolve_backend
from .engine import Engine
from .pool import LocalPool, ProcessPool

STATIC = Path(__file__).parent / "static"
PREVIEW_CHARS = 320


def load_table(path: Path, text: str | None, limit: int | None) -> tuple[pl.DataFrame, str]:
    if not path.exists():
        raise SystemExit(f"jcol: no such file: {path}")
    readers = {".parquet": pl.read_parquet, ".csv": pl.read_csv, ".tsv": lambda p: pl.read_csv(p, separator="\t")}
    if path.suffix.lower() not in readers:
        raise SystemExit(f"jcol: {path.name}: expected a .parquet, .csv or .tsv file")
    df = readers[path.suffix.lower()](path)
    if limit:
        df = df.head(limit)
    strings = [c for c, t in df.schema.items() if t == pl.String]
    if text is None:
        if not strings:
            raise SystemExit(f"jcol: {path.name} has no text column to read")
        # the column to read is the one with the most text in it
        text = max(strings, key=lambda c: df[c].str.len_chars().mean() or 0)
    if text not in df.columns:
        raise SystemExit(f"jcol: {path.name} has no column {text!r}; its columns are {df.columns}")
    return df.with_columns(pl.col(text).cast(pl.String).fill_null("")), text


def build(df: pl.DataFrame, text: str, *, source: str, budget: float, api: str | None, model: str | None,
          cache: bool, workers: int, per_worker: int) -> Starlette:
    backend, key = resolve_backend(api)
    model = model or backend.model
    pool = (ProcessPool(backend.name, key, model, workers, per_worker) if workers > 0
            else LocalPool(Jev(key, backend, model=model, timeout=12), per_worker))
    engine = Engine(df[text].to_list(), pool, model=model, cache=Cache() if cache else None, budget=budget)
    fields = [c for c in df.columns if c != text]
    table = {
        "version": __version__, "source": source, "n": df.height, "text": text, "fields": fields, "order": engine.order,
        "previews": [t[:PREVIEW_CHARS] for t in engine.texts],
        "values": {c: df[c].cast(pl.String).fill_null("").to_list() for c in fields},
    }

    async def index(request):
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})  # the page changes with the package

    async def init(request):
        return JSONResponse(table | {"stats": engine.stats()})

    async def row(request):
        i = int(request.path_params["i"])
        if not 0 <= i < df.height:
            return JSONResponse({"error": "no such row"}, status_code=404)
        return JSONResponse({k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                             for k, v in df.row(i, named=True).items()})

    async def ws(socket: WebSocket):
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        engine.listeners.add(queue)
        await socket.send_text(json.dumps(engine.snapshot()))

        async def pump():
            while True:
                await socket.send_text(json.dumps(await queue.get()))

        sender = asyncio.ensure_future(pump())
        try:
            while True:
                m = json.loads(await socket.receive_text())
                kind = m.get("type")
                if kind == "preview":
                    engine.preview(m.get("header", ""), [int(r) for r in m.get("rows", [])][:80])
                elif kind == "commit":
                    engine.visible = [int(r) for r in m.get("rows", [])][:80]
                    engine.commit(m.get("header", ""))
                elif kind == "view":
                    engine.view([int(r) for r in m.get("rows", [])][:80])
                elif kind == "remove":
                    engine.remove(str(m.get("id")))
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            engine.listeners.discard(queue)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await engine.start()
        print(f"jcol: ready ({workers or 1} worker{'s' if workers != 1 else ''})", file=sys.stderr)
        yield
        await engine.close()

    return Starlette(routes=[Route("/", index), Route("/api/init", init), Route("/api/row/{i}", row),
                             WebSocketRoute("/ws", ws)], lifespan=lifespan)


def cli() -> None:
    ap = argparse.ArgumentParser(prog="jcol", description="Open a table in the browser and add columns by describing them.")
    ap.add_argument("file", help="a .parquet, .csv or .tsv table")
    ap.add_argument("--text", metavar="COLUMN", help="the column Jev reads (default: the one with the most text)")
    ap.add_argument("--limit", type=int, metavar="N", help="use only the first N rows")
    ap.add_argument("--budget", type=float, default=2.0, metavar="DOLLARS", help="stop spending at this much (default 2.00)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--api", choices=("typesafe", "openrouter"), help="which API to call (default: whichever has a key)")
    ap.add_argument("--model", metavar="ID")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the answer cache")
    ap.add_argument("--workers", type=int, default=16, metavar="N",
                    help="worker processes; one process tops out near 200 calls a second (default 16; 0 runs in-process)")
    ap.add_argument("--per-worker", type=int, default=64, metavar="N", help="background calls in flight per worker (default 64)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser window")
    ap.add_argument("--version", action="version", version=f"jcol {__version__}")
    a = ap.parse_args()
    df, text = load_table(Path(a.file), a.text, a.limit)
    try:
        app = build(df, text, source=Path(a.file).name, budget=a.budget, api=a.api, model=a.model,
                    cache=not a.no_cache, workers=a.workers, per_worker=a.per_worker)
    except JevFatal as e:
        raise SystemExit(f"jcol: {e}")
    url = f"http://127.0.0.1:{a.port}"
    print(f"jcol: {df.height:,} rows of {Path(a.file).name}; reading column {text!r}; budget ${a.budget:.2f}\njcol: {url}",
          file=sys.stderr)
    if not a.no_open:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    cli()

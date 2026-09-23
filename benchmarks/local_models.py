"""The product-group proxy check on this machine's two local decision servers, beside the recorded Jev run.

Run from the repository root, one invocation at a time (the servers share one GPU, and wall time is a result).
Each run needs its own empty XDG_CACHE_HOME and a fresh output directory; both are refused if already used.

    OUT=benchmarks/results/local-models-2026-09-22
    CACHE=$PWD/bench/out/local-2026-09-22
    env -u JEV_URL UV_CACHE_DIR=$HOME/.cache/uv JEV_API=diffusiongemma JEV_MODEL=openjev-0.1 XDG_CACHE_HOME=$CACHE/cache-diffusiongemma-product100 \
      uv run python benchmarks/local_models.py run --api diffusiongemma --model openjev-0.1 --rows 100 \
      --output-dir $OUT/diffusiongemma-product100 > $OUT/diffusiongemma-product100.log 2>&1
    env -u JEV_URL UV_CACHE_DIR=$HOME/.cache/uv JEV_API=laya JEV_MODEL=laya-421m XDG_CACHE_HOME=$CACHE/cache-laya-product100 \
      uv run python benchmarks/local_models.py run --api laya --model laya-421m --rows 100 \
      --output-dir $OUT/laya-product100 > $OUT/laya-product100.log 2>&1
    # the same with --rows 1000 and product1000 in the names; then, offline:
    uv run python benchmarks/local_models.py summarize

`run` repeats product_agreement.py (same 100 rows, codebook, engine, no answer cache, concurrency 4,
no truncation) with three changes a local server needs:

- The provider is named, must be DiffusionGemma or Laya, and must resolve to a loopback URL; nothing hosted
  can be reached. JEV_URL is refused because it would override the provider's URL.
- The per-call deadline is 300 s (DiffusionGemma) or 120 s (Laya), not the 12 s jcol.annotate builds its
  client with: a local server answers requests one at a time, so a call can wait behind three others.
- The spending threshold is $0.000001, not $0.10. jcol stops scheduling once cost >= budget, so a budget of
  exactly 0 would send nothing even though local calls are metered at $0; any priced call would stop the run.

With --rows 1000 the sample is Polars sample(n=1000, seed=70, shuffle=True), whose first 100 rows are the
100-row sample. Laya refuses (HTTP 422) any narrative that does not fit its 512-token window with the
question; the run reads the server's audit log (--laya-audit) and counts those refusals by state hash.

`summarize` writes docs/benchmarks/local-models-2026-09-22.json from the recorded Jev report and each local
report.json. product_group is a derived administrative label: agreement with it is a proxy, not accuracy.
"""

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
import polars as pl

import jcol
from jcol import batch
from jcol.core import PROVIDERS
from jevkit_runtime import Client, JevError, JevFatal, resolve

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/complaints-5k.parquet"
CODEBOOK = ROOT / "examples/complaints-codebook.json"
JEV = ROOT / "benchmarks/results/product-agreement-2026-09-21"
RESULTS = ROOT / "benchmarks/results/local-models-2026-09-22"
FROZEN = ROOT / "docs/benchmarks/local-models-2026-09-22.json"
LAYA_AUDIT = Path(os.environ["LAYA_AUDIT"]) if os.environ.get("LAYA_AUDIT") else None  # the Laya server's --audit file
LOCAL = {
    "diffusiongemma": {"deadline": 300.0, "server": "OpenJev commit e04794a; weights mlx-community/diffusiongemma-26B-A4B-it-4bit revision a7a8140"},
    "laya": {"deadline": 120.0, "server": "jevkit-core scripts/laya_server.py; laya-mlx commit fc1df62; checkpoint aac6fef/laya-mlx revision 0476785; 512-token window, no truncation"},
}
BUDGET = 1e-6
CONCURRENCY = 4
MACHINE = "Apple M3 Ultra, 96 GiB unified memory"
LABEL = "product_group derives from the original administrative product label; proxy agreement only, not accuracy"


def state_hash(text: str) -> str:
    """The hash the Laya adapter logs for a request's state."""
    return hashlib.sha256(json.dumps(text, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sample(rows: int) -> pl.DataFrame:
    return pl.read_parquet(SOURCE).sample(n=rows, seed=70, shuffle=True)


def audit_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def local_backend(api: str, model: str):
    """Resolve the named provider and refuse anything that could leave this machine."""
    if os.environ.get("JEV_URL"):
        raise SystemExit("JEV_URL is set and would override the local server URL; unset it")
    if os.environ.get("JEV_PRICE_PER_MTOK"):
        raise SystemExit("JEV_PRICE_PER_MTOK is set; local runs must be metered at $0")
    backend = resolve(PROVIDERS, api, model=model)
    host = urlsplit(backend.url).hostname
    if backend.name not in LOCAL or host not in ("127.0.0.1", "localhost", "::1") or backend.price_per_mtok != 0:
        raise SystemExit(f"refusing {backend.name} at {backend.url}: this benchmark only runs local servers")
    return backend


def fresh_cache() -> Path:
    """The run's XDG_CACHE_HOME, refused if JevKit has used it. `uv run` alone would put its own cache
    here, so pin UV_CACHE_DIR in the command; anything but a `uv` directory is refused too."""
    raw = os.environ.get("XDG_CACHE_HOME")
    if not raw:
        raise SystemExit("set XDG_CACHE_HOME to a fresh directory for this run")
    cache = Path(raw)
    if cache.exists() and [p.name for p in cache.iterdir() if p.name != "uv"]:
        raise SystemExit(f"{cache} is not empty; use a new cache directory for every run")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def recording_client(deadline: float, failures: list):
    """jcol's client with the local deadline, keeping each failure's message (annotate only counts them)."""

    class LocalClient(Client):
        def __init__(self, backend, **kwargs):
            super().__init__(backend, **(kwargs | {"timeout": deadline}))

        async def ask(self, state, questions, **kwargs):
            try:
                return await super().ask(state, questions, **kwargs)
            except (JevError, JevFatal) as exc:
                failures.append({"state_sha256": state_hash(state), "error": str(exc)[:300]})
                raise

    return LocalClient


def scrubbed(report: dict | None) -> dict | None:
    """A server's /health reply without machine paths: the Laya adapter reports its checkpoint's full path."""
    if report and isinstance(report.get("checkpoint"), str) and "/" in report["checkpoint"]:
        return report | {"checkpoint": "…/" + "/".join(Path(report["checkpoint"]).parts[-3:])}
    return report


def health(url: str) -> dict | None:
    parts = urlsplit(url)
    try:
        return httpx.get(f"{parts.scheme}://{parts.netloc}/health", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return None


def progress_printer(started: float):
    last = [0.0]

    def show(state: dict) -> None:
        now = time.perf_counter()
        if now - last[0] >= 15:
            last[0] = now
            s = state["stats"]
            print(f"  {now - started:7.1f}s  {state['filled']}/{state['total']} cells; {s['calls']} calls, "
                  f"{s['errors']} errors, {s['inflight']} in flight", file=sys.stderr, flush=True)

    return show


def jev_comparison(table: pl.DataFrame) -> dict | None:
    """Per-row agreement with Jev's recorded choices, on the rows both runs have. None outside the 100."""
    jev = pl.read_csv(JEV / "predictions.csv").select("complaint_id", pl.col("coded_product").alias("jev"))
    if not set(table["complaint_id"].to_list()) <= set(jev["complaint_id"].to_list()):
        return None
    both = table.select("complaint_id", "product_group", "coded_product").join(jev, on="complaint_id", how="left")
    answered = both.filter(pl.col("coded_product").is_not_null())
    return {"rows": both.height, "local_answered": answered.height,
            "agree_with_jev": int((answered["coded_product"] == answered["jev"]).sum()),
            "on_rows_local_answered": {"local_agrees_with_product_group": int((answered["coded_product"] == answered["product_group"]).sum()),
                                       "jev_agrees_with_product_group": int((answered["jev"] == answered["product_group"]).sum())},
            "on_rows_local_missed": {"rows": both.height - answered.height,
                                     "jev_agrees_with_product_group": int(both.filter(pl.col("coded_product").is_null())
                                                                          .select((pl.col("jev") == pl.col("product_group")).sum()).item())},
            "source": "benchmarks/results/product-agreement-2026-09-21/predictions.csv (Jev 1.13, recorded 2026-09-21; not re-run)"}


def run(args) -> None:
    backend = local_backend(args.api, args.model)
    cache = fresh_cache()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; use a fresh directory to preserve prior evidence")
    output.mkdir(parents=True, exist_ok=True)
    deadline = args.deadline or LOCAL[backend.name]["deadline"]
    df = sample(args.rows)
    df.write_parquet(output / "sample.parquet")
    jev_manifest = json.loads((JEV / "manifest.json").read_text())
    sample_sha = hashlib.sha256((output / "sample.parquet").read_bytes()).hexdigest()
    ids = df["complaint_id"].to_list()
    if args.rows == 100 and (sample_sha != jev_manifest["sample_sha256"] or ids != jev_manifest["complaint_ids"]):
        raise SystemExit("the 100-row sample does not match the recorded Jev manifest; check the Polars version")
    texts = df["narrative"].cast(pl.String).fill_null("").to_list()
    manifest = {"source": "data/complaints-5k.parquet", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "sample_sha256": sample_sha, "codebook_sha256": hashlib.sha256(CODEBOOK.read_bytes()).hexdigest(),
                "sampling": f"{args.rows} rows without replacement, Polars sample(n={args.rows}, seed=70, shuffle=True)",
                "matches_jev_sample": sample_sha == jev_manifest["sample_sha256"],
                "contains_jev_rows": len(set(jev_manifest["complaint_ids"]) & set(ids)),
                "jev_rows_are_prefix": ids[:100] == jev_manifest["complaint_ids"],
                "complaint_ids": ids, "narrative_chars": sum(map(len, texts)), "polars": pl.__version__,
                "python": platform.python_version(), "jcol": jcol.__version__, "label_status": LABEL,
                "provider": backend.name, "endpoint": backend.url, "model_requested": backend.model,
                "server": LOCAL[backend.name]["server"], "server_health": scrubbed(health(backend.url)), "machine": MACHINE,
                "settings": {"cache": False, "concurrency": args.concurrency, "deadline_seconds": deadline,
                             "budget": BUDGET, "max_chars": None,
                             "xdg_cache_home": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else cache.name}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{backend.name} {backend.model} at {backend.url}: {args.rows} rows, concurrency {args.concurrency}, "
          f"deadline {deadline:g}s, cache {cache}", file=sys.stderr, flush=True)
    audit_before = len(audit_lines(args.laya_audit)) if backend.name == "laya" else None
    failures: list[dict] = []
    batch.Client = recording_client(deadline, failures)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result = jcol.annotate(df, CODEBOOK, project=output / "run.jcol.sqlite", api=backend.name, model=backend.model,
                           cache=False, budget=BUDGET, concurrency=args.concurrency, progress=progress_printer(started))
    seconds = time.perf_counter() - started
    result.write(output / "predictions.parquet")
    table = result.table
    table.select("complaint_id", "product_group", "coded_product", "coded_product__confidence").write_csv(output / "predictions.csv")
    with sqlite3.connect(output / "run.jcol.sqlite") as db:
        answers = [{"row_index": row, "column_id": col, "answer": json.loads(answer)}
                   for col, row, answer in db.execute("SELECT col, row, answer FROM cells ORDER BY row, col")]
    (output / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
    answered = table.filter(pl.col("coded_product").is_not_null())
    majority = df["product_group"].value_counts().sort("count", descending=True).row(0)
    summary = {"rows": df.height, "answered": answered.height, "unanswered": df.height - answered.height,
               "agree_with_product_group": int((answered["coded_product"] == answered["product_group"]).sum()),
               "majority_label": {"label": majority[0], "rows": majority[1]},
               "macro_f1_answered_rows_all_10_labels": result.evaluation["columns"]["coded_product"]["macro_f1"],
               "mean_confidence": answered["coded_product__confidence"].mean(),
               "seconds": seconds, "rows_per_second": df.height / seconds,
               "calls": result.metadata["stats"]["calls"], "shared_or_cached": result.metadata["stats"]["cached"],
               "errors": result.metadata["stats"]["errors"], "dollars": result.metadata["stats"]["dollars"],
               "input_tokens": result.metadata["stats"]["tokens"], "returned_model": result.metadata["stats"]["model"],
               "failed_requests": len(failures),
               "failure_messages": sorted({f["error"] for f in failures})[:10]}
    if backend.name == "laya":
        new = audit_lines(args.laya_audit)[audit_before:]
        ours = {state_hash(t) for t in texts}
        mine = [a for a in new if a.get("state_sha256") in ours]
        rejected = {a["state_sha256"] for a in mine if a.get("status") == "context_rejected"}
        summary["laya_audit"] = {"path": args.laya_audit.name, "lines_before": audit_before, "lines_after": audit_before + len(new),
                                 "requests": len(mine), "foreign_requests": len(new) - len(mine),
                                 "statuses": {s: sum(a.get("status") == s for a in mine) for s in sorted({a.get("status") for a in mine})},
                                 "rejected_requests": sum(a.get("status") == "context_rejected" for a in mine),
                                 "rejected_rows": sum(state_hash(t) in rejected for t in texts),
                                 "note": "jcol re-asks failed cells in up to two more sweeps, so one refused row can log up to three refusals"}
    summary["cache_after_run"] = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if p.parts[len(cache.parts)] != "uv")
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "started_at": started_at.isoformat(),
              "started_at_new_york": started_at.astimezone(ZoneInfo("America/New_York")).isoformat(),
              "seconds": seconds, "summary": summary, "jev": jev_comparison(table) if args.rows <= 100 else None,
              "run": result.metadata, "evaluation": result.evaluation, "failures": failures}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"complete": result.complete, "summary": summary, "jev": report["jev"]}, indent=2, default=str), flush=True)


def comparisons(local: dict) -> dict:
    """Offline cross-run numbers from the recorded predictions: the two local models on the same rows, and the
    Jev 100 rows split by whether Laya answered them. No model calls."""
    def preds(name, alias):
        return pl.read_csv(RESULTS / name / "predictions.csv", schema_overrides={"coded_product": pl.String}) \
                 .select("complaint_id", "product_group", pl.col("coded_product").alias(alias))

    def agree(df, col):
        return int((df[col] == df["product_group"]).sum())

    out = {}
    jev = pl.read_csv(JEV / "predictions.csv").select("complaint_id", pl.col("coded_product").alias("jev"))
    for rows in (100, 1000):
        dg, laya = f"diffusiongemma-product{rows}", f"laya-product{rows}"
        if dg not in local or laya not in local:
            continue
        both = preds(dg, "dg").join(preds(laya, "laya").drop("product_group"), on="complaint_id")
        if rows == 100:
            both = both.join(jev, on="complaint_id")
        answered, missed = both.filter(pl.col("laya").is_not_null()), both.filter(pl.col("laya").is_null())
        entry = {"rows": both.height,
                 "rows_laya_answered": {"rows": answered.height, "laya_agrees_with_product_group": agree(answered, "laya"),
                                        "diffusiongemma_agrees_with_product_group": agree(answered, "dg"),
                                        "diffusiongemma_same_label_as_laya": int((answered["dg"] == answered["laya"]).sum()),
                                        "majority_label_baseline": answered["product_group"].value_counts().sort("count", descending=True).row(0)[1]},
                 "rows_laya_refused": {"rows": missed.height, "diffusiongemma_agrees_with_product_group": agree(missed, "dg")}}
        if rows == 100:
            entry["rows_laya_answered"]["jev_agrees_with_product_group"] = agree(answered, "jev")
            entry["rows_laya_refused"]["jev_agrees_with_product_group"] = agree(missed, "jev")
            entry["diffusiongemma_vs_jev"] = {"both_agree_with_product_group": int(((both["dg"] == both["product_group"]) & (both["jev"] == both["product_group"])).sum()),
                                              "only_diffusiongemma_agrees": int(((both["dg"] == both["product_group"]) & (both["jev"] != both["product_group"])).sum()),
                                              "only_jev_agrees": int(((both["dg"] != both["product_group"]) & (both["jev"] == both["product_group"])).sum())}
        out[f"diffusiongemma_vs_laya_product{rows}"] = entry
    for name, entry in local.items():
        if entry["jev"] is None:
            first = set(pl.read_csv(JEV / "predictions.csv")["complaint_id"].to_list())
            rest = preds(name, "x").filter(~pl.col("complaint_id").is_in(list(first)))
            answered = rest.drop_nulls("x")
            out[f"{name}_rows_outside_jev_100"] = {"rows": rest.height, "answered": answered.height, "agree_with_product_group": agree(answered, "x"),
                                                    "majority_label_baseline": rest["product_group"].value_counts().sort("count", descending=True).row(0)[1]}
    return out


def summarize(args) -> None:
    jev = json.loads((JEV / "report.json").read_text())
    jev_eval = jev["evaluation"]["columns"]["coded_product"]
    frozen = {"date": "2026-09-22", "tool": "jcol", "machine": MACHINE, "label_status": LABEL,
              "task": "Code 'coded_product' (10-label category question) from the complaint narrative; compare with product_group",
              "codebook": "examples/complaints-codebook.json",
              "jev": {"source": "benchmarks/results/product-agreement-2026-09-21 (restored from git 0e1e7c6^; not re-run)",
                      "rows": 100, "answered": jev_eval["evaluated"],
                      "agree_with_product_group": round(jev_eval["accuracy"] * jev_eval["evaluated"]),
                      "macro_f1_all_10_labels": jev_eval["macro_f1"], "seconds": jev["seconds"],
                      "calls": jev["run"]["stats"]["calls"], "dollars": jev["run"]["stats"]["dollars"],
                      "requested_model": jev["run"]["identity"]["model"], "returned_model": jev["run"]["stats"]["model"],
                      "settings": "OpenRouter, no answer cache, concurrency 4, $0.10 threshold, 12 s deadline"},
              "local": {}}
    for directory in sorted(p for p in RESULTS.iterdir() if (p / "report.json").exists()):
        report = json.loads((directory / "report.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        frozen["local"][directory.name] = {"provider": manifest["provider"], "model": manifest["model_requested"],
                                           "server": manifest["server"], "settings": manifest["settings"],
                                           "sampling": manifest["sampling"], "sample_sha256": manifest["sample_sha256"],
                                           "contains_jev_rows": manifest["contains_jev_rows"],
                                           "started_at_new_york": report["started_at_new_york"], "summary": report["summary"],
                                           "jev": report["jev"], "result_dir": str(directory.relative_to(ROOT))}
    # the 100-row runs against the first 100 rows of the larger run on the same model: does a second read agree?
    for name, entry in frozen["local"].items():
        if entry["jev"] is not None:
            continue
        small = next((d for d, e in frozen["local"].items() if e["provider"] == entry["provider"] and e["jev"] is not None), None)
        if small:
            a = pl.read_csv(RESULTS / small / "predictions.csv").select("complaint_id", pl.col("coded_product").alias("a"))
            b = pl.read_csv(RESULTS / name / "predictions.csv").select("complaint_id", pl.col("coded_product").alias("b"))
            both = a.join(b, on="complaint_id").drop_nulls()
            entry["repeat_on_shared_rows"] = {"against": small, "rows_both_answered": both.height,
                                              "same_label": int((both["a"] == both["b"]).sum())}
    for entry in frozen["local"].values():
        s = entry["summary"]
        entry["wall_clock_seconds"] = s["seconds"]
        entry["calls"] = s["calls"]
        entry["failed_decisions"] = s["unanswered"]
        entry["laya_rejections"] = ({k: s["laya_audit"][k] for k in ("rejected_rows", "rejected_requests", "foreign_requests")}
                                    if "laya_audit" in s else None)
        entry["raw_outputs"] = [f"{entry['result_dir']}/{f}" for f in ("manifest.json", "report.json", "predictions.csv", "answers.json",
                                                                       "predictions.parquet", "sample.parquet")] + [f"{entry['result_dir']}.log"]
    frozen["servers"] = {name: {"endpoint": "http://127.0.0.1:%d/v1/systemone" % (8080 if name == "diffusiongemma" else 8081),
                                "model": "openjev-0.1" if name == "diffusiongemma" else "laya-421m", "identity": spec["server"],
                                "deadline_seconds": spec["deadline"], "concurrency": CONCURRENCY}
                         for name, spec in LOCAL.items()}
    frozen["jev"]["provenance"] = {
        "date": "2026-09-21 (recorded_at 2026-09-21T04:52:28Z)",
        "files": {"agree_with_product_group, macro_f1_all_10_labels, answered": "benchmarks/results/product-agreement-2026-09-21/report.json (evaluation.columns.coded_product)",
                  "seconds, calls, dollars, requested_model, returned_model": "benchmarks/results/product-agreement-2026-09-21/report.json (seconds, run.stats, run.identity)",
                  "per-row choices used for local-vs-Jev agreement": "benchmarks/results/product-agreement-2026-09-21/predictions.csv",
                  "sample hash and ordered complaint IDs": "benchmarks/results/product-agreement-2026-09-21/manifest.json"},
        "restored_from": "git show 0e1e7c6^:<path> in the jcol repo (commit 0e1e7c6 removed them on 2026-09-21)",
        "majority_label_baseline": "55/100 credit reporting (benchmarks/README.md restored from 0e1e7c6^)",
        "comparability": "Proxy agreement with an administrative label, not accuracy. Jev ran hosted via OpenRouter with a 12 s deadline; "
                          "its 6.63 s wall time is not comparable hardware-wise with local wall-clock on the M3 Ultra. No Jev calls were made on 2026-09-22.",
        "not_available": "No Jev number exists for the 1000-row sample (or its rows 101-1000); those runs have no Jev column."}
    frozen["comparisons"] = comparisons(frozen["local"])
    frozen["run_conditions"] = (
        "Runs were serialised: DiffusionGemma product100, DiffusionGemma product1000, Laya product100, Laya product1000, "
        "one at a time, each with a fresh XDG_CACHE_HOME under bench/out/local-2026-09-22/ and no answer cache. "
        "While one server was being timed the other was loaded but idle (the OpenJev access log shows only this run's "
        "4 client connections during each DiffusionGemma run and no requests after it; the Laya audit log shows no "
        "foreign requests during either Laya run). Concurrency 4 and deadlines 300 s / 120 s were used throughout; no "
        "deadline errors occurred. No hosted model was called. See benchmarks/results/local-models-2026-09-22/RESULTS-NOTES.md.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2, default=str) + "\n")
    print(f"wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="one timed run on one local server")
    p.add_argument("--api", choices=tuple(LOCAL), required=True)
    p.add_argument("--model", required=True, help="openjev-0.1 for diffusiongemma, laya-421m for laya")
    p.add_argument("--rows", type=int, default=100)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    p.add_argument("--deadline", type=float, help="per-call seconds (default: 300 diffusiongemma, 120 laya)")
    p.add_argument("--laya-audit", type=Path, default=LAYA_AUDIT)
    p.set_defaults(func=run)
    p = commands.add_parser("summarize", help="freeze every recorded run into one JSON file; no model calls")
    p.add_argument("--output", type=Path, default=FROZEN)
    p.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

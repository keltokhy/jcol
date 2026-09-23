"""Reproducible proxy-label check on a fixed sample of the shipped public complaints.

This invokes the paid API unless --prepare-only is passed. The codebook is fixed before
sampling; product_group is a derived administrative label, not independently reviewed truth.
"""

import argparse
import hashlib
import json
import platform
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

import jcol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="typesafe/jev-1.13")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "data/complaints-5k.parquet"
    codebook = root / "examples/complaints-codebook.json"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        parser.error("output directory already contains a report; use a fresh directory to preserve prior evidence")
    sample = pl.read_parquet(source).sample(n=100, seed=70, shuffle=True)
    sample.write_parquet(output / "sample.parquet")
    manifest = {"source": "data/complaints-5k.parquet", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sample_sha256": hashlib.sha256((output / "sample.parquet").read_bytes()).hexdigest(),
                "codebook_sha256": hashlib.sha256(codebook.read_bytes()).hexdigest(),
                "sampling": "100 rows without replacement, Polars sample(n=100, seed=70, shuffle=True)",
                "complaint_ids": sample["complaint_id"].to_list(), "polars": pl.__version__,
                "python": platform.python_version(), "jcol": jcol.__version__,
                "label_status": "product_group derives from the original administrative product label; proxy agreement only",
                "model_requested": args.model}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        return
    started = time.perf_counter()
    result = jcol.annotate(sample, codebook, project=output / "run.jcol.sqlite", api="openrouter", model=args.model,
                           cache=False, budget=args.budget, concurrency=4)
    result.write(output / "predictions.parquet")
    result.table.select("complaint_id", "product_group", "coded_product", "coded_product__confidence").write_csv(output / "predictions.csv")
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
              "run": result.metadata, "evaluation": result.evaluation}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with sqlite3.connect(output / "run.jcol.sqlite") as db:
        answers = [{"row_index": row, "column_id": col, "answer": json.loads(answer)}
                   for col, row, answer in db.execute("SELECT col, row, answer FROM cells ORDER BY row, col")]
    (output / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
    print(json.dumps({"complete": result.complete, "seconds": report["seconds"], "stats": result.metadata["stats"],
                      "metrics": result.evaluation["columns"]["coded_product"]}, indent=2))
    if not result.complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

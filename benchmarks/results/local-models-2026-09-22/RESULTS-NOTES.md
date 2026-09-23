# Notes on the local-model runs, 2026-09-22

Frozen numbers: `docs/benchmarks/local-models-2026-09-22.json`. Raw outputs are in the four run directories here, with a `.log` beside each one.
Agreement with `product_group` is a proxy agreement check against an administrative label, not accuracy (see `benchmarks/README.md`).
The Jev 1.13 figures come from `benchmarks/results/product-agreement-2026-09-21/` (recorded 2026-09-21, restored from `0e1e7c6^`). Jev was not re-run.

## What ran (America/New_York)

| run | start | wall s | rows answered | calls | Laya refusals (rows / requests) |
|---|---|---:|---:|---:|---|
| diffusiongemma-product100 | 20:54:23 | 134.6 | 100/100 | 100 | n/a |
| diffusiongemma-product1000 | 20:56:45 | 693.5 | 1000/1000 | 1000 | n/a |
| laya-product100 | 21:08:29 | 71.4 | 73/100 | 73 | 27 / 81 |
| laya-product1000 | 21:09:47 | 58.8 | 798/1000 | 798 | 202 / 606 |

- The runs went one at a time in the order shown. Each had a fresh cache dir, concurrency 4 and deadlines of 300 s (DiffusionGemma) and 120 s (Laya). There were no deadline errors and no HTTP 503s, so concurrency stayed at 4.
- Every run returned the local model (`openjev-0.1` / `laya-421m`) at $0.0. No hosted endpoint was called.
- While one server was being timed, the other was loaded but idle. The OpenJev access log shows exactly 100 and then 1000 POSTs, all on this runner's 4 connections, and nothing after 21:08. `foreign_requests` in the Laya audit log was 0 for both Laya runs.
- No 500-row fallback was needed. The 45-minute check was 134.6 s × 8.0 (ratio of narrative characters) ≈ 1,077 s, well under 2,700 s. The product100 run itself served as the probe, because its rows are the first 100 rows of the 1000-row sample. The 1000-row run took 11.6 min, which is under the 15 to 20 min estimate.
- The 1000-row sample (Polars `sample(n=1000, seed=70, shuffle=True)`) contains all 100 Jev rows as its prefix (`contains_jev_rows=100`, `jev_rows_are_prefix=true`). The 100-row sample hash `756f3977…9965` and the ordered complaint IDs match the Jev manifest.

## Odd things seen

1. **DiffusionGemma product100 wall time includes a slow start.** The progress line showed 0/100 cells at 16 s. In the 1000-row run, the same first 100 rows finished in about 90 s, against 134.6 s here (0.74 against 1.44 rows/s). The OpenJev log shows no other client, so the likely cause is warm-up after idle, but I have not verified it. Following the rules, the run was not repeated to get a better number.
2. **Laya product100 was much slower per request than Laya product1000.** Server-side seconds in the audit log sum to 71.0 s over the 73 answered requests. The median was 0.76 s for roughly the first 60 requests, then dropped to about 0.03 s. In the 1000-row run the median was 0.03 s (p90 0.07 s). DiffusionGemma was idle and no other client used Laya, so the cause is unknown (warm-up or contention outside these two servers). The 1000-row run is the better figure for Laya throughput (about 17 rows/s including refusals).
3. **An earlier attempt of this stage was stopped part-way (about 19:40).** I moved its outputs to the trash and redid every run. Its DiffusionGemma product100 run gave the same 100 labels as today's (78/100, macro-F1 0.5091, 86/100 agreement with Jev) in 111.3 s. Its 1000-row run was stopped at 334/1000 after 331 s and was slowing down (about 0.6 rows/s at the end). Copies of that attempt are in `/tmp/jcol-earlier-attempt/` and in the trash; none of its numbers are used.
4. **Determinism.** On the shared first 100 rows, DiffusionGemma gave 100/100 identical labels in its 100-row and 1000-row runs, and it also matched the stopped attempt. Laya gave 73/73 identical labels and refused the same 27 rows in both runs.

## Laya coverage

- Laya's 512-token window leaves 316 state tokens for this 10-option question. Narratives that do not fit are refused with HTTP 422 (`context_rejected`). jcol retries each failed cell in two more sweeps, so every refused row appears 3 times in the audit log. jcol reports these rows as failed decisions (`complete=false`, errors 81 and 606), not as answers.
- The refusals affect a biased subset: the longest narratives. On the 27 refused Jev rows, Jev agreed with product_group on 22 and DiffusionGemma on 25. Laya's agreement and macro-F1 therefore describe short narratives only. Its macro-F1 is computed over answered rows.
- On the 73 rows Laya answered, it agreed with product_group on 50, against 53 for Jev, 53 for DiffusionGemma and 43 for the majority label. On the 1000-row run it agreed on 562 of 798 answered rows, against 660 for DiffusionGemma on the same rows and 498 for the majority label.
- Laya overuses `other`: it chose it 54 times in 1000 rows, while the reference has 1 such row. It underuses `vehicle loan` (3 against 20 in the reference).

## Runner behaviour to know (not package changes)

- Budget is 1e-6, not 0. jcol stops scheduling when cost >= budget, so `--budget 0` sends no requests even to a $0 server. This is a jcol bug; a fix should be proposed.
- `jcol.annotate` hard-codes a 12 s client deadline. The runner replaces `jcol.batch.Client` to apply the 300 s and 120 s deadlines. Jev's recorded run used 12 s.
- In this stage I extended `local_models.py summarize` so the frozen JSON also carries server identifiers and deadlines, Jev provenance per number, per-run failed decisions, Laya refusals and raw-output paths, and offline cross-model comparisons (the `comparisons` key). The `run` code path was not changed after the runs.

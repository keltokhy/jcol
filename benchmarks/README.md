# Product-group proxy agreement

The first check uses a fixed 100-row random sample of the repository's public complaint data.
The definition was fixed before the call and was not revised after seeing results.

| Measure | Recorded value |
|---|---:|
| Rows / answered / labeled | 100 / 100 / 100 |
| Agreement with product_group | 75/100 |
| Majority-label baseline | 55/100 |
| Macro-F1, all 10 declared labels | 0.4961 |
| Input tokens reported by API | 89,152 |
| Cost reported by API, rounded | $0.00374 |
| Wall time, one local invocation | 6.63 s |
| Requested model | typesafe/jev-1.13 |
| Returned model | typesafe/jev-1.13-20260917 |

The macro average assigns zero F1 to absent categories. In this sample `other` has no reference
rows; `personal loan` and `vehicle loan` have one each. No uncertainty or generalization claim
is inferred from this small sample.

**Interpretation.** `product_group` derives from the original administrative product labels via
`data/SAMPLE.json`; it is not independently reviewed narrative coding. A narrative may discuss
several products. The frozen codebook also has boundary differences from the reference mapping:
it puts prepaid cards under money transfer, while the reference puts them under credit card;
the reference puts the generic historical `consumer loan` label under vehicle loan. Thus
agreement includes differences between coding schemes, not just model mistakes. The 25
disagreements should be inspected, and this result must not be described as validated accuracy
for other definitions or datasets.

## Evidence

The [recorded result directory](results/product-agreement-2026-09-21/) contains:

- `manifest.json`: source, sample and codebook hashes, row IDs, versions and sampling method.
- `report.json`: actual run settings, returned model, metrics, confusion matrix and disagreements.
- `predictions.csv`: row ID, reference label, prediction and confidence.
- `answers.json`: raw per-question answer payloads from the API, before cell rounding.

The report retains the original temporary project path as provenance. The source narratives can
be recovered from the committed source data using the manifest's ordered complaint IDs.
Raw answers are individual model answers, not full HTTP response envelopes or request logs.
The recorded run used the initial format-1 Arrow-buffer project fingerprint. Before release,
project fingerprints changed to format 2, hashing logical values and schema so that unchanged
data survives a Parquet round trip. The original report is retained; its internal fingerprint
is historical and its temporary project is not a format-2 resumable project. Source-file and
codebook hashes, ordered row IDs, raw answers and offline metric replay remain verifiable.

## Reproduce

```bash
uv run python benchmarks/product_agreement.py --output-dir /tmp/jcol-product-check
```

This performs paid API calls with OpenRouter, no answer cache, a $0.10 spending threshold,
concurrency 4, and no truncation. Use a fresh directory for each recorded run; an existing report
is never overwritten. A project left by an interrupted run can resume, so cost and time in its
final report then cover only that invocation. `--prepare-only` generates the sample and manifest
without contacting the API. The manifest records the Polars version used for sampling.

To recompute the committed metrics without credentials or model calls:

```bash
jcol evaluate benchmarks/results/product-agreement-2026-09-21/predictions.csv \
  --codebook examples/complaints-codebook.json --report /tmp/jcol-validation.json
```

The binary, scale and multi-field paths have offline integration tests. This experiment tests
one narrative-only category definition. It provides no measured accuracy for those other paths.

## Local models, 2026-09-22

On September 22, 2026, the same check ran on the two local decision servers, DiffusionGemma through
OpenJev (`openjev-0.1`) and Laya (`laya-421m`), on an Apple M3 Ultra with 96 GiB of unified memory.
Each coded `coded_product` with the unchanged codebook, first on the 100 rows of the Jev run above and
then on 1,000 rows drawn the same way (Polars `sample(n=1000, seed=70, shuffle=True)`), which contain
those 100. [`local_models.py`](local_models.py) ran the four runs one at a time, with no answer cache,
an empty cache directory for each, 4 requests in flight and no truncation. The Jev column is the
recorded run above, reused and not re-run; no Jev number exists for the 1,000 rows.
[local-models-2026-09-22.json](../docs/benchmarks/local-models-2026-09-22.json) is the frozen output;
the manifests, reports, predictions, raw answers and logs of each run are in
[results/local-models-2026-09-22/](results/local-models-2026-09-22/). Agreement with `product_group`
is proxy agreement, not accuracy, for the reasons under **Interpretation** above.

### 100 complaints

| | Jev 1.13 (`typesafe/jev-1.13`, OpenRouter) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Rows answered | 100 of 100 | 100 of 100 | 73 of 100 |
| Agreement with `product_group` | 75/100 | 78/100 | 50/100 |
| on the 73 rows Laya answered | 53 | 53 | 50 |
| on the 27 rows Laya refused | 22 | 25 | none answered |
| Same label as Jev | | 86/100 | 50 of 73 |
| Macro-F1, all 10 declared labels | 0.4961 | 0.5091 | 0.4586, answered rows only |
| Calls | 100 | 100 | 73 |
| Failed decisions | 0 | 0 | 27 (81 refused requests) |
| Wall-clock | 6.63 s, hosted | 134.6 s | 71.4 s |
| API cost | $0.00374 | $0 | $0 |

Always choosing the most common reference label, credit reporting, agrees on 55 of the 100 rows and
on 43 of the 73 that Laya answered. DiffusionGemma and Jev both agree with `product_group` on 71
rows, DiffusionGemma alone on 7 and Jev alone on 4.

### 1,000 complaints (local models only)

| | Jev 1.13 (`typesafe/jev-1.13`, OpenRouter) | DiffusionGemma (OpenJev, `openjev-0.1`, MLX) | Laya (`laya-421m`, MLX) |
|---|---:|---:|---:|
| Rows answered | not run | 1,000 of 1,000 | 798 of 1,000 |
| Agreement with `product_group` | | 830/1,000 | 562/1,000 |
| on the 798 rows Laya answered | | 660 | 562 |
| on the 202 rows Laya refused | | 170 | none answered |
| on the 900 rows outside the Jev sample | | 752 of 900 | 512 of 725 answered |
| Macro-F1, all 10 declared labels | | 0.6569 | 0.4899, answered rows only |
| Same label as the 100-row run, on shared rows | | 100 of 100 | 73 of 73 |
| Calls | | 1,000 | 798 |
| Failed decisions | | 0 | 202 (606 refused requests) |
| Wall-clock | | 693.5 s (1.44 rows/s) | 58.8 s (17.0 rows/s) |
| API cost | | $0 | $0 |

The most common label, credit reporting, agrees on 596 of the 1,000 rows, 498 of the 798 that Laya
answered and 541 of the 900 outside the Jev sample.

### What the tables say

- **DiffusionGemma is level with Jev on this check.** It agrees with `product_group` on 78 rows
  against Jev's 75, with macro-F1 0.5091 against 0.4961, and picks Jev's label on 86 of 100. The 11
  rows where only one of them matches the reference split 7 to 4, and 100 rows cannot tell the two
  apart. On 1,000 rows it agrees on 830, against 596 for the most common label. Macro-F1 is not
  comparable between the two samples, because the 100-row sample has almost no reference rows for
  some labels (see above).
- **Laya is fast and bounded by its window.** It reads at most 512 tokens, question included, and
  refuses a longer request with HTTP 422 instead of truncating it, so 27 of 100 and 202 of 1,000 rows
  came back as failed decisions, not answers. The refused rows are the long narratives; on them
  DiffusionGemma agrees with the reference on 25 of 27 and 170 of 202, so Laya's figures describe
  shorter narratives only. On the rows it read, Laya agrees on 50 of 73 and 562 of 798; on the same
  rows DiffusionGemma agrees on 53 and 660, and the most common label on 43 and 498. It coded the
  1,000 rows in 58.8 s, against 693.5 s for DiffusionGemma.
- **Both local models are repeatable here.** On the rows the two samples share, DiffusionGemma gave
  the same label 100 times out of 100 and Laya 73 out of 73.
- **Cost and speed.** Jev coded the 100 rows in 6.63 s for $0.00374 through OpenRouter; that is hosted
  time and not comparable with local hardware. Both local models cost $0. DiffusionGemma's 100-row
  time includes a slow start, visible in
  [its log](results/local-models-2026-09-22/diffusiongemma-product100.log), and the same 100 rows
  open the 1,000-row run, so 1.44 rows/s is the better figure for its throughput. Laya's 100-row run
  was also slow at first; 17.0 rows/s from its 1,000-row run is the better figure, and it includes
  the refused rows.

### Provenance and caveats

- The Jev column is the run recorded on 2026-09-21 at 00:52 ET in
  [results/product-agreement-2026-09-21/](results/product-agreement-2026-09-21/), restored from git
  `0e1e7c6^` and not re-run. Rows answered, agreement, macro-F1, wall-clock, calls and cost come from
  `report.json`; the per-row labels behind "same label as Jev" and the row splits from
  `predictions.csv`; the sample hash and ordered complaint IDs from `manifest.json`. It requested
  `typesafe/jev-1.13` and got `typesafe/jev-1.13-20260917` through OpenRouter, with no answer cache,
  concurrency 4, a $0.10 threshold and a 12 s deadline. No Jev calls were made on 2026-09-22.
- Servers: DiffusionGemma is OpenJev at commit `e04794a`, serving `openjev-0.1` from
  `mlx-community/diffusiongemma-26B-A4B-it-4bit` revision `a7a8140` at `127.0.0.1:8080`, which runs
  model work one call at a time on the GPU. Laya is laya-mlx at commit `fc1df62` with
  `aac6fef/laya-mlx` revision `0476785`, served as `laya-421m` at `127.0.0.1:8081` through
  jevkit-core's `scripts/laya_server.py`, which refuses with HTTP 422 any request whose state would
  be cropped to fit its 512-token window, question included. Both on an Apple M3 Ultra with 96 GiB
  of unified memory and loaded throughout.
- Settings: 4 requests in flight on both servers, with per-call deadlines of 300 s for DiffusionGemma
  and 120 s for Laya. `jcol.annotate` builds its client with a 12 s deadline, so the runner substitutes
  one with these. There were no deadline errors, so concurrency was never lowered. The spending
  threshold was $0.000001 instead of $0, because jcol stops scheduling once cost reaches the budget
  and a budget of 0 would send nothing. No answer cache; each run had a fresh `XDG_CACHE_HOME` under
  `bench/out/local-2026-09-22/`.
- Order, in America/New_York time: DiffusionGemma on 100 rows from 20:54, on 1,000 rows from 20:56,
  Laya on 100 rows from 21:08 and on 1,000 rows from 21:09, one at a time. While one server was timed
  the other was loaded but idle: the OpenJev access log shows only the runner's connections, and the
  Laya audit log shows no requests from other clients during the Laya runs. No hosted model was called.
- Laya's refusals come from the adapter's audit log (`context_rejected`), counted by lines before and
  after each run: 81 requests for 27 rows and 606 for 202 rows, because jcol asks a failed cell again
  in up to two more sweeps. jcol reports these rows as failed decisions and the run as incomplete;
  they are not answers. Laya's macro-F1 covers only the rows it answered.
- Samples: the 100-row sample hash (`756f3977…9965`) and its ordered complaint IDs match the Jev
  manifest. The 1,000-row sample (`680a9e31…3d64`) contains all 100 Jev rows; the other 900 have no
  Jev answers.
- Nothing was tuned. The codebook and data are those of the Jev run; only the provider, the deadline
  and the spending threshold changed. Each figure comes from one run. An earlier attempt, stopped
  part-way, was discarded and none of its numbers are used;
  [RESULTS-NOTES.md](results/local-models-2026-09-22/RESULTS-NOTES.md) describes it and the slow
  starts.
- One codebook, one column and one dataset, compared with an administrative label.

### Reproduce the local runs

```bash
env -u JEV_URL UV_CACHE_DIR=$HOME/.cache/uv JEV_API=diffusiongemma JEV_MODEL=openjev-0.1 \
  XDG_CACHE_HOME=/tmp/jcol-cache-diffusiongemma-100 \
  uv run python benchmarks/local_models.py run --api diffusiongemma --model openjev-0.1 \
  --rows 100 --output-dir /tmp/jcol-local/diffusiongemma-product100
```

Repeat with `JEV_API=laya JEV_MODEL=laya-421m` and `--api laya --model laya-421m`, and with
`--rows 1000`. Each run needs a new output directory and an empty cache directory; the script
refuses used ones, and refuses any provider that is not a local server. Laya runs read the adapter's
audit log (`--laya-audit`) to count refusals. `summarize` rebuilds the frozen JSON from the recorded
runs without model calls:

```bash
uv run python benchmarks/local_models.py summarize --output /tmp/local-models-2026-09-22.json
```

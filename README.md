# jcol

Apply a natural-language codebook to a table from your terminal. Validate the inputs,
run with checkpoints, resume interrupted work, and pipe or export the results.
Each definition becomes a column of judgments from
[Jev](https://docs.typesafe.ai), TypeSafe's decision model.

```bash
jcol run data/complaints-5k.parquet \
  --codebook examples/complaints-codebook.json \
  --output coded.parquet --report report.json
```

The output retains every source column and adds a value and confidence for each variable.
The command checkpoints successful cells in `coded.parquet.jcol.sqlite`; rerunning the same
command resumes missing cells. The browser also saves its committed columns across restarts.

jcol also has a Python API and an optional browser interface. Software tests use a local
fake API and do not establish model accuracy. See [Validation](#validation).

## Install

Python 3.10 or later:

```bash
uv tool install jev-col        # the command it installs is jcol
jcol --help
jcol doctor
```

This installs `jcol` on PATH in its own environment. For `import jcol`, add `jev-col` to your
project (`uv add jev-col` or `pip install jev-col`). If your shell cannot find the command, run
`uv tool update-shell` and restart the shell. For local development, clone the repository
and run `make install-local` (or `uv tool install --force .`). To build a package:

```bash
uv build
uv tool install --force dist/jev_col-*.whl
```

Set `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`, or put a key in
`~/.config/jev/typesafe.key` or `~/.config/jev/openrouter.key`. With both keys, TypeSafe wins.
Choose explicitly with `--api` or `JEV_API`. The key stays in the Python process.
Only `run` and `browse` need a key. `doctor` checks local configuration; `doctor --check`
also sends an unauthenticated HEAD request to test reachability, without doing inference.
Reachability does not verify authentication or model availability.

### Local servers (experimental)

`--api diffusiongemma`, `--api laya` and `--api gliner` send the same questions to a System One server on your own
machine, an [OpenJev](https://github.com/razorback16/openjev), [laya-mlx](https://github.com/mizorewww/laya-mlx)
or [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) process that you run separately. They are never
chosen automatically, need no key, and count as $0 against the budget unless `JEV_PRICE_PER_MTOK`
is set; `doctor` lists them. The runtime's [DiffusionGemma](https://github.com/keltokhy/jevkit-core/blob/main/docs/diffusiongemma.md),
[Laya](https://github.com/keltokhy/jevkit-core/blob/main/docs/laya.md) and [GLiNER](https://github.com/keltokhy/jevkit-core/blob/main/docs/gliner.md) guides explain the setup; keep
`--concurrency` low while a local model warms up.
See [How well does it work](#how-well-does-it-work) for a comparison with Jev.

## Terminal workflow

```bash
jcol inspect reviews.csv
jcol init --input review --name refund_request \
  --definition 'The customer explicitly asks for a refund.' -o codebook.json
jcol validate reviews.csv --codebook codebook.json
jcol run reviews.csv --codebook codebook.json --dry-run
jcol run reviews.csv --codebook codebook.json -o coded.parquet --budget 0.25
jcol status coded.parquet.jcol.sqlite
jcol export coded.parquet.jcol.sqlite --source reviews.csv -o recovered.csv
```

Repeat the same `run` command to resume. Successful cells are retained across interrupts;
`export` recovers them without credentials or API calls. Export requires the original table,
including its schema and row order. `status` and `export` can read an active project.

| Command | Purpose |
|---|---|
| `init` | Generate a valid, editable codebook from a definition |
| `inspect` | Discover field names, types, null counts and row count |
| `validate` | Check inputs, codebook, gold labels and output-name conflicts offline |
| `run --dry-run` | Describe work and check an existing checkpoint without writes or calls |
| `run` | Annotate, checkpoint, resume and write results |
| `status` | Read completion per saved column |
| `export` | Recover saved results, including partial results |
| `evaluate` | Compare predictions with supplied reviewed labels |
| `doctor` | Inspect configuration without revealing keys |
| `browse` | Open the optional local browser interface |

For a category, add `--kind category --option billing --option other` to `init`.
For a scale, use `--kind scale` and repeat `--option` from lowest to highest.
Edit the generated JSON to add more variables or detailed option definitions.
Existing codebooks are protected unless `init --force` is given.

### Pipes and automation

```bash
cat reviews.csv | jcol run - --input-format csv --codebook codebook.json \
  --project study.sqlite -o - --output-format jsonl --report report.json \
  > coded.jsonl
jcol --json status study.sqlite
```

Input/output formats are CSV, TSV, Parquet and JSONL (`.ndjson` is also recognized).
Stdin requires `--input-format`; stdout defaults to JSONL. A streamed run requires
`--project PATH` or an explicit `--no-project`. Tables are held in memory; pipe support
does not make processing out-of-core. Named table outputs and reports are replaced
atomically after successful serialization. These two files are written separately.

Metadata commands emit JSON objects. `--json`, before or after the subcommand, wraps
reports in `{"schema_version":1,"command":"status","ok":true,"data":{...}}`.
Errors use `{"schema_version":1,"command":"status","ok":false,"error":{"code":"command_error","message":"..."}}`.
Argument errors use `usage_error`; interruptions use `interrupted`. For a partial run,
the report is valid (`ok: true`), `data.run.complete` is false, and exit status is 2.
Without `--json`, `run` and `evaluate` retain their original unwrapped report formats.

With `-o -`, stdout contains only the table (or the raw codebook for `init`). Run/export
reports go to `--report` or stderr; errors also go to stderr. Use `--quiet` to suppress
human summaries and terminal progress when parsing stderr as JSON. Help and version are
always text. `browse` does not support JSON output.

Exit codes: **0** success (including exporting partial results or discovering missing
credentials with `doctor`), **1** invalid input or execution failure, **2** partial annotation,
**130** interrupted, **141** closed output pipe. In scripts, check `doctor`'s `ready` field,
`status`'s `complete` field, and the run exit code. No-argument `jcol` prints help.
Use `python -m jcol` when invoking an installed package through a specific Python environment.

## Codebooks

A codebook is a versioned JSON file with explicit input fields and named output variables:

```json
{
  "version": 1,
  "inputs": ["narrative", "issue"],
  "columns": [
    {
      "name": "unauthorized_activity",
      "kind": "binary",
      "definition": "The consumer alleges an account or transaction they did not authorize. Include identity theft. Exclude disputes about authorized fees or loan terms.",
      "gold": "reviewer_label"
    }
  ]
}
```

`definition` is the actual instruction sent to Jev. It can contain punctuation, inclusion
rules and exclusions without the browser header grammar. Only `inputs` are sent. With
multiple inputs each row is a JSON object with field names and values; a single input is
sent as plain text. A missing single-field input becomes empty text. Source nulls remain
null in the result. Multiple-field inputs retain JSON nulls.

| Kind | Required options | Output |
|---|---|---|
| `binary` | None | Probability from 0 to 1 |
| `category` | 2–255 distinct labels, as a list or an object mapping labels to definitions | Chosen label |
| `scale` | 2–10 distinct levels, listed from lowest to highest | Numeric position from 0 to number of levels minus one |

Every variable also adds `NAME__confidence`. For a binary variable this is
`max(p, 1-p)`; for other kinds it is the API's confidence when supplied, otherwise zero.
These are model scores, not guarantees of correctness or calibrated confidence intervals.
Cell values retain the browser's rounding: binary probabilities and confidence to three
decimals, scale values to two. Saved projects also retain the unrounded per-question answer.

`gold` optionally names a reference-label column for evaluation. Declared gold fields
cannot also be model inputs. Unknown codebook fields, invalid labels, duplicate names and
output names that would overwrite source columns are rejected before a batch run makes calls.
Check the remaining inputs yourself for indirect label leakage.

Examples:

- [Product codebook](https://github.com/keltokhy/jcol/blob/main/examples/complaints-codebook.json): narrative-only product classification with proxy labels.
- [Multiple-field codebook](https://github.com/keltokhy/jcol/blob/main/examples/multifield-codebook.json): narrative and issue used together, with binary and category definitions.

## Batch and Python

```bash
jcol run reviews.csv --codebook codebook.json --output coded.csv
jcol run reviews.tsv --codebook codebook.json --output coded.parquet \
  --project study.jcol.sqlite --model typesafe/jev-1.13 --api openrouter \
  --budget 0.25 --concurrency 4 --report report.json
jcol evaluate coded.parquet --codebook codebook.json --report validation.json
```

Tables can be CSV, TSV, Parquet or JSONL. Batch mode sends full serialized rows by default;
`--max-chars N` explicitly truncates them and reports how many were affected. Export always
preserves the full source values. Parquet preserves source types; CSV/TSV use their usual
text representations.

`--project` defaults to `OUTPUT.jcol.sqlite`. Successful cells are durable as they arrive,
including if the process crashes. A project checks the full table contents, schema and order,
selected inputs, codebook, model ID, endpoint and truncation setting before resuming.
Changed inputs require a new project path. One process may own a project at a time.

Partial output and the JSON report are still written after budget exhaustion or API failures.
A report goes to stdout unless `--report` names a file or the table is streamed to stdout.
All requested columns share each row's call; repeated identical rows share answers.

```python
import jcol

result = jcol.annotate(
    "reviews.parquet", "codebook.json",
    project="study.jcol.sqlite",
    budget=0.25,
    concurrency=4,
)
result.write("coded.parquet")
print(result.complete)
print(result.evaluation)
```

`annotate` accepts a Polars DataFrame or a table path; the codebook can be a path, dictionary
or `jcol.Codebook`. The result contains a Polars `table`, `metadata`, and `evaluation`.
The Python API persists when `project` is provided; without it, results live in the returned
object. In a notebook or another running event loop, use `await jcol.annotate_async(...)`.
Call `jcol.evaluate(dataframe, codebook)` to evaluate existing predictions without model calls.

## Validation

Evaluation reports labeled rows, evaluated rows, missing predictions and coverage alongside
metrics. Missing labels are excluded. Missing predictions reduce coverage and are excluded
from the metric denominator, so always read the two together. Invalid nonmissing labels or
predictions are errors.

- **Binary:** accuracy at a configurable threshold (default 0.5), Brier score, per-class
  precision/recall/F1, macro-F1 and a confusion matrix. Gold accepts true/false, yes/no or 1/0.
- **Category:** exact-label accuracy, per-class metrics, macro-F1 and a confusion matrix.
  Matching is case-sensitive. Macro-F1 averages every declared category, including absent
  classes, whose F1 is zero.
- **Scale:** mean absolute error in scale positions; gold may be a numeric position or an
  exact level name.
- **Disagreements:** zero-based row indices with the reference and predicted values.

Freeze your definitions, reserve representative reviewed rows, and examine disagreements
before using a derived variable.

## How well does it work

One recorded check codes 100 public complaints with the
[product codebook](https://github.com/keltokhy/jcol/blob/main/examples/complaints-codebook.json), which picks the financial product at the
center of each narrative from ten labels, and compares the answers with `product_group`, a grouping
of each complaint's administrative product label. This is proxy agreement, not accuracy: a narrative
can discuss several products, and the codebook draws some boundaries differently from the reference.
[benchmarks/README.md](https://github.com/keltokhy/jcol/blob/main/benchmarks/README.md) has the full record and its caveats.

| 100 complaints | Jev 1.13 (OpenRouter) | DiffusionGemma (`openjev-0.1`, local) | Laya (`laya-421m`, local) |
|---|---:|---:|---:|
| Rows answered | 100 | 100 | 73 |
| Agreement with `product_group` (majority-label baseline: 55) | 75 | 78 | 50 |
| Wall time | 6.63 s | 134.6 s | 71.4 s |

DiffusionGemma answered every row and agreed with `product_group` as often as Jev; 100 rows are too
few to separate the two. On 1,000 rows it agreed on 830, against 596 for the majority label, at about
1.4 rows a second on an Apple M3 Ultra. Laya took under a minute on the same 1,000 rows, but it reads
at most 512 tokens, question included, and refused the narratives that did not fit: 27 of 100 and 202
of 1,000. Jev remains the default; DiffusionGemma is a reasonable substitute when the text must stay
on your machine, and Laya is not one for narratives like these.

## Browser

```bash
jcol browse data/complaints-5k.parquet --text narrative
jcol browse tickets.csv --text subject body --max-chars 0 --no-open
```

The legacy `jcol FILE --text ...` invocation also works. The browser opens on
`http://127.0.0.1:8765`. Type a header, inspect the preview, and press Enter:

```text
alleges fraud?
product: mortgage, credit card, other
tone: calm < upset < furious
```

Headers without a colon are binary descriptions. A colon followed by comma-separated options
creates a category column; levels separated by `<` create a scale. The typed description is
inserted into a short question template. Use a JSON codebook and batch mode for detailed rules.

- Click a column name to sort, or a cell to filter. Click a filter chip to remove it.
- `show fields` reveals source fields. Click a row to read all selected inputs and fields.
- Category columns offer `compare with…` for a quick comparison with a source field.
- `export csv` downloads the filtered, sorted rows with **full source values**, numeric scale
  positions, and committed values/confidences. A generated name that conflicts with a source
  field is prefixed with `jcol_COLUMNID__` until it is unique.
- Columns persist in `FILE.jcol.sqlite` by default. Use `--project PATH` to choose a location
  or `--no-project` for a temporary session. Removing a column also removes it from the project.

| Browser option | Default / meaning |
|---|---|
| `--text COLUMN [COLUMN ...]` | Selected inputs; default is the longest string column on average |
| `--limit N` | First N rows |
| `--max-chars N` | 4,000 characters per serialized row; warns on truncation; 0 sends full rows |
| `--budget DOLLARS` | 2.00, or `$JEV_BUDGET`; no call goes out that would take spending past it; `none` for no limit |
| `--workers N` / `--per-worker N` | 16 sending processes, 64 background calls each; workers 0 sends in-process |
| `--api` / `--model` | Backend and model ID; `JEV_MODEL` is also accepted |
| `--no-cache` | Disable the separate answer cache, while retaining project persistence |
| `--port` / `--no-open` | Port 8765; optionally suppress opening the browser |

The browser prioritizes visible rows and may resend a slow visible-row call. Background rows
follow a fixed shuffle (seed 70). Its running estimates describe model judgments over the table;
the displayed margins cover sampling uncertainty only. They do not measure classification error.

## Costs, persistence and limits

- Every call, background or on screen, sets its estimated price aside before it goes out, and
  the first goes alone to learn the real price, so calls in flight together cannot pass the
  budget; only a price that rises while calls are in the air can. Budget and cost counters reset
  each invocation. A saved project preserves cells, not a cumulative spending limit.
- Answers also cache in `~/.cache/jev/answers.v3.sqlite`, shared with the other JevKit tools. Keys
  include provider, endpoint, requested model, serialized input and question. `JEV_URL` overrides
  the endpoint. No API keys are stored in projects or exports.
- The model is pinned to a Jev release (`jev-1.13.0`, `typesafe/jev-1.13` on OpenRouter), so a
  project's answers do not mix versions when an alias moves. `--model jev-latest` asks for the
  alias; the run's record in `metadata["run"]` names every model that answered.
- The full table and results are held in memory. The browser receives previews and source fields
  for every row and sorts locally. This is intended for thousands of rows, not out-of-core datasets.
- Transient failures are retried; the scheduler makes up to two additional sweeps for missing cells.
  Invalid answers remain missing. Fatal authentication or credit errors stop new background work.
- Every browser tab shares the same columns and preview. The server binds only to localhost,
  rejects nonlocal Host headers and cross-origin HTTP/WebSocket requests, and has no user
  authentication. Other local programs can access it. Data sent to the model goes to the selected API provider.
- Jev follows the definition provided; text in a row can influence its answer. Review coded data,
  and do not treat a judgment column as a security boundary.

## Data and development

`data/complaints-5k.parquet` contains 5,000 public CFPB complaints and their fields.
[data/SAMPLE.json](https://github.com/keltokhy/jcol/blob/main/data/SAMPLE.json) records the Hugging Face snapshot, original sampling method,
seed, dates and exact mapping used to derive `product_group`. Original source files and the
original 5,000-row sampling script are not bundled.

```bash
uv sync
uv run pytest -q   # offline fake API; no credentials or external connections
uv build          # source distribution and wheel
```

CI runs tests and installs the built wheel outside the checkout on Python 3.10 and 3.13,
checking the CLI, public imports and packaged browser assets.

`codebook.py`, `batch.py`, `evaluation.py` and `project.py` implement the reusable workflow.
`engine.py` is the shared scheduler; `core.py` and `pool.py` implement API calls and caching.
`app.py` and `static/index.html` implement the browser. MIT license.

## Shared JevKit development

This tool uses [`jevkit-runtime`](https://github.com/keltokhy/jevkit-core), imported
as `jevkit_runtime`. Clone that repository beside this one as `../jevkit-core`, then
run `uv sync`. Core Python edits apply on the next invocation of this tool;
restart long-lived Python processes after editing.

The distribution name is `jevkit-runtime` because `jevkit-core` on PyPI belongs
to a different project. The runtime is [available on PyPI](https://pypi.org/project/jevkit-runtime/).
Use the sibling checkout for shared development, or `uv sync --no-sources` for a
standalone source checkout. Existing published versions of this tool are
unaffected by this source migration.

From the core checkout, `python scripts/dev.py setup`, `check`, and `wheel-check`
set up and validate all five consumers in separate environments.
The codebook, the column grammar, the two-lane scheduler and projects remain in this repository;
answer identity, the answer store, transport, metering, the budget and the worker processes are
the runtime's. Runtime 0.4 keeps answers in `answers.v3.sqlite`, so the first run after upgrading
re-asks once.

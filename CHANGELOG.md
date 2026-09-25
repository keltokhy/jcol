# Changelog

## 0.7.0

On `jevkit-runtime` 0.4 ([jevkit-core#14](https://github.com/keltokhy/jevkit-core/issues/14)).

- The engine asks through the runtime's client, which now does what jcol's own pool and engine did:
  the cache, sharing a call already in the air, re-sending slow on-screen calls, and sending from
  worker processes (`--workers`). jcol keeps the two lanes, the shuffled background order, projects
  and the browser. `pool.py` is gone.
- The budget holds: every call sets its estimated price aside first, the first goes alone to learn
  the price, and on-screen previews are inside the budget too. `--budget none` is no limit and
  `JEV_BUDGET` sets the default; `annotate(budget=...)` also takes a runtime `Budget`.
- Columns ask runtime questions (`Noul`, `Choice`, `Score`), which validate options and bounds.
- `annotate()` works inside a notebook's running loop instead of raising.
- `AnnotationResult.metadata["run"]` is the runtime's record of the run.
- A System One gateway of your own is offered; the model is pinned to `jev-1.13.0`.
- The cache moves to `~/.cache/jev/answers.v3.sqlite`; the first run after upgrading re-asks.

## Unreleased

- Fix answer reuse on `--api diffusiongemma`. DiffusionGemma answers each question in the light of the
  others in its call, but jcol cached each column's answer on its own, so an answer given beside one set
  of columns could be served to another. Its answers are now keyed on the whole call, as
  `jevkit-runtime` keys them, and served from the cache only to the same call. DiffusionGemma answers
  cached by earlier versions are no longer used; other providers are unaffected.
  ([#3](https://github.com/keltokhy/jcol/issues/3))
- Answers jcol writes to the shared cache now record the provider, the requested model and the model
  that answered, as the runtime's own entries do.

## 0.6.0

- Add `--api gliner`, a local [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide)
  server (`JEV_GLINER_URL`, port 8082). Requires `jevkit-runtime>=0.3.2`.

## 0.5.1

- First release on PyPI. The distribution is named `jev-col`, matching `jev-grep`, `jev-sort` and
  `jev-select`. The import (`jcol`) and the command (`jcol`) are unchanged.
- README links point to GitHub, so they also work on PyPI.
- Restore the recorded product check in `benchmarks/`, with runs on the local DiffusionGemma and Laya
  servers beside Jev. It measures proxy agreement with an administrative label, not accuracy.

## 0.5.0

- Add `--api diffusiongemma` and `--api laya` for System One servers running on your own machine,
  through `jevkit-runtime` 0.3: chosen only by name, no key needed, and metered at zero API fees
  unless `JEV_PRICE_PER_MTOK` is set.

## 0.4.0

- Move provider selection, answer identity, the cache, validation and metering to the shared
  `jevkit-runtime` 0.2. The tool now only names the providers it offers.

## 0.3.0

- Command-line annotation workflow: apply a codebook to a table, resume interrupted runs, export
  labels and evaluate them against a gold column.
- Local browser access is hardened, and API responses with invalid metering are rejected.

## 0.1.0

- First version: a browser table whose columns Jev fills from plain-English headers.

# Changelog

## Unreleased

- The PyPI distribution is named `jev-col`, matching `jev-grep`, `jev-sort` and `jev-select`. The
  import (`jcol`) and the command (`jcol`) are unchanged.

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

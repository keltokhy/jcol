# jcol

A table column, but the formula is a description.

```console
$ jcol data/complaints-5k.parquet --text narrative
jcol: 5,000 rows of complaints-5k.parquet; reading column 'narrative'; budget $2.00
jcol: http://127.0.0.1:8765
jcol: ready (16 workers)
```

A browser window opens on the table. The last header cell is a text box, and a header typed into
it has one of three shapes:

```
alleges fraud?                                   yes/no      -> a probability
product: mortgage, credit card, other            categories  -> a label and its probability
tone: calm < upset < furious                     a scale     -> a position on it
```

While you type, the rows on screen are judged and shown as a preview. Press Enter and the column is
kept: the rows on screen first, then every other row in the background, with a running estimate for
the whole table in the column's header.

Each cell is one question to [Jev](https://docs.typesafe.ai), TypeSafe's decision model. Jev does
not generate text. It returns a probability in about 200 ms for about a thousandth of a cent; the
measurements behind those two figures are in [jgrep](https://github.com/keltokhy/jgrep)'s README.
jcol is a demonstration of what that speed and price allow, a column of judgments that fills while
you watch. It is not a product. It comes from the same family as jgrep,
[jsort](https://github.com/keltokhy/jsort), [jlink](https://github.com/keltokhy/jlink) and
[jselect](https://github.com/keltokhy/jselect), but it has no releases, is not on PyPI, and has no
accuracy or speed benchmarks of its own. This README describes what the code does and reports no
results.

## Install

jcol is not on PyPI. Clone it, which also gets you the sample data:

```bash
git clone https://github.com/keltokhy/jcol && cd jcol
uv sync
uv run jcol data/complaints-5k.parquet --text narrative
```

Or install only the command, and point it at a table of your own:

```bash
uv tool install git+https://github.com/keltokhy/jcol
```

It needs Python 3.10 or later and a key for one of two APIs. With keys for both, it uses TypeSafe's.

| API | Key | Get one |
|---|---|---|
| TypeSafe | `TYPESAFE_API_KEY` | [console.typesafe.ai](https://console.typesafe.ai/settings/keys) |
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

Set the environment variable, or put the key in `~/.config/jev/typesafe.key` or
`~/.config/jev/openrouter.key`, which is where the sibling tools look too. Force a choice with
`--api` or `JEV_API`. jcol does not have the gateway backend that jgrep and jsort have.

## Use

```bash
jcol data/complaints-5k.parquet --text narrative    # the sample in this repository
jcol reviews.csv                                    # reads the column with the most text in it
jcol tickets.tsv --text body --limit 2000           # the first 2,000 rows
jcol notes.parquet --budget 0.25 --port 9000 --no-open
```

The table is a `.parquet`, `.csv` or `.tsv` file. jcol reads one text column from it and shows the
rest as fields.

| Option | Meaning |
|---|---|
| `--text COLUMN` | The column Jev reads. Default: the string column whose values are longest on average. |
| `--limit N` | Use only the first N rows. |
| `--budget DOLLARS` | Stop filling in the background once this much is spent. Default 2.00. See [Cost](#cost-and-the-cache). |
| `--port PORT` | Default 8765. The server listens on 127.0.0.1 only. |
| `--api`, `--model ID` | Which API, and which model ID. Default: whichever API has a key, and its alias for the latest Jev. |
| `--no-cache` | Do not read or write the answer cache. |
| `--workers N` | Worker processes that make the calls. Default 16; 0 makes them in the server process. |
| `--per-worker N` | Background calls in flight per worker. Default 64. |
| `--no-open` | Do not open a browser window. |

### Headers

| Shape | Example | Cell |
|---|---|---|
| yes/no | `alleges fraud?` | The probability, with a bar. |
| categories | `product: mortgage, credit card, other` | The chosen label and its probability. |
| scale | `tone: calm < upset < furious` | The nearest level's name and the position on the scale. |

A header with no colon is a description, and needs at least three characters; a final `?` is
dropped. `name: a, b, c` needs 2 to 255 distinct options and `name: low < mid < high` needs 2 to 10
distinct levels. Nothing is asked while a header fits none of these.

The header goes into one question per row. A description becomes `The text fits this description:
"..."`, a list becomes `Choose the <name> that best describes the text.` with the options, and a
scale becomes `Rate the text on this scale: <name>.` with the levels. Anything that completes those
sentences sensibly will work.

### In the page

- **Typing.** The page waits for a pause of 160 ms, or 450 ms once the header has a colon and a list
  is still being written, and then asks for a preview of the rows on screen, 80 at most. The hint
  under the box says how long the screen took to fill, as timed by the browser. Enter keeps the
  column and Escape clears the box.
- **A kept column's header** shows the running estimate, how many rows are done, and, when the
  column is complete, how long it took.
- **Sort** by clicking a column's name: highest first, then lowest first, then off. Rows with no
  value yet sort last.
- **Filter** by clicking a cell. A yes/no cell keeps the rows at 0.5 or above, a category cell the
  rows with that label, a scale cell the rows at that level or above. Each filter is a chip above
  the table; click it to remove it.
- **Compare.** A category column has a "compare with…" menu of the table's fields. Pick one and the
  header adds the share of rows in the [running sample](#the-running-estimate) whose label equals
  that field's value, ignoring case.
- **show fields** adds the table's own columns to the view. Clicking a row's text opens the full
  text with all of its fields and your columns' values. ✕ removes a column.
- **export csv** downloads the rows in view, after filters and sorting: the row number, the first
  320 characters of the text, the table's fields, and each added column's value and probability.
- **The ticker** at the top right counts calls, answers that needed no call (from the cache, or
  shared with an identical call in the air), calls that were sent a second time, dollars spent
  against the budget, and the model that answered.

### The running estimate

The background does not fill the table from top to bottom. It follows one fixed random order (a
shuffle with seed 70), so the rows finished so far, taken in that order, are a simple random sample
of the table. The header's figure is computed from that sample: for a yes/no column, the share of
rows at 0.5 or above; for categories, the two most common labels; for a scale, the mean position
and the level nearest to it. Yes/no and scale columns carry a 95% margin of error with a
finite-population correction, which shrinks to nothing as the column completes, and then the figure
is exact.

The margin covers sampling only. It says how far the partial figure may be from what Jev would say
about the whole table, not how far Jev is from the truth.

## How it works

**One call per row, not per cell.** A call carries one row's text and one question for every column
that still needs that row. Columns added together share their calls.

**Two lanes.** Rows on screen go in the hot lane. They are sent at once, outside the concurrency
limit, and a call still unanswered after 0.35 seconds is sent a second time; whichever answer lands
first is used. Every other row goes in the background lane, in the random order above, with at most
`--workers` × `--per-worker` calls in flight. The background sends nothing new while a hot call is
pending.

**Worker processes.** By default 16 processes make the calls, each with its own event loop and its
own HTTP/2 connections, fed by one queue for hot jobs and one for background jobs. The comments at
the top of `src/jcol/pool.py` and in `src/jcol/core.py` record the measurements that led to this
design, and the scripts in `spike/` are what took them. They are not repeated here.

**Repeats are free.** Answers are cached under the model ID, the text sent and the question. A text
that appears twice, or a header typed a second time, is answered from the cache or joins the
identical call that is already in the air.

**Failures.** A call is retried inside a 12-second budget after a transport error or an HTTP 408,
429, 500, 502, 503, 504 or 529. If it still fails its cell stays empty, and when the background
pass ends it goes over the missing cells again, up to two more times. A 401, 402 or 403 is shown in
a bar across the page. Rows on screen stop being judged and no new pass starts, though the pass
under way still runs through its remaining rows.

**The server and the page.** The server holds the key, the table and the scheduler. The page is one
static HTML file with no dependencies and no build step, and talks to the server over a WebSocket.
A page that is reloaded, or opened in a second tab, is sent the kept columns as they stand.

## The data

`data/complaints-5k.parquet` is 5,000 complaints from the CFPB Consumer Complaint Database: the
consumer's `narrative` and ten fields (`complaint_id`, `date_received`, `product`, `sub_product`,
`issue`, `company`, `state`, `company_response`, `timely` and `product_group`).
`data/SAMPLE.json` records where it came from:

- **Source.** The `has-text` configuration of
  [BEE-spoke-data/consumer-finance-complaints](https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints),
  a CC0 snapshot of the CFPB database, 1,689,573 complaints. A snapshot is used because, as of
  2026-09-18, the CFPB's own bulk file and API no longer include the narratives.
- **Sample.** A simple random sample of 5,000 without replacement, `DataFrame.sample(seed=70)` in
  polars, drawn on 2026-09-18, with no filtering by length, date or topic. The complaints in it were
  received between 2015-03-24 and 2024-02-03.
- **`product_group`** is the one derived field. It folds the CFPB's product names, 19 of them in
  the sample, into nine groups and `other`, by the first keyword in `SAMPLE.json` that the product
  name contains. It is there so that a `product: ...` column has something to be compared with.

The narratives are the consumers' own words as the CFPB published them, with personal details
replaced by runs of X; 3,901 of the 5,000 contain `XXXX`. The median narrative is about 650
characters and the mean about 1,000. 134 are longer than the 4,000 characters jcol sends, and 4,815
are distinct.

The snapshot's three full files, over 900 MB, are not in the repository, and neither is the script
that drew the sample. `SAMPLE.json` is the record of the method.

## Cost and the cache

jgrep's README gives the rule of thumb for what Jev bills: about 270 tokens of overhead per call
plus the text and the question, at $0.042 per million tokens. By that rule, with a token for every
four characters, a first yes/no column over the sample is roughly 2.5 million tokens, or about ten
cents. That is arithmetic, not a measurement. The ticker shows what a session actually spends.
OpenRouter reports the cost of each call; TypeSafe's API reports tokens, which jcol prices at
`JEV_PRICE_PER_MTOK`, 0.042 by default.

Two things cost more than they might seem to. A preview asks a new question of every row on screen
at each pause in typing, so a header typed in three bursts is three screens of calls. And a call
that is sent a second time may be billed twice.

`--budget` is a seat belt, not a cap. It is checked by the background lane before each call it
sends, against the cost of the calls that have come back, so the calls in flight at that moment
(up to 1,024 with the default workers) land on top of it. When it trips, filling stops, the page
says so, and rows on screen stop being judged too. Previews and scrolling alone never trip it:
until a column is kept, and again once every kept column is complete, nothing checks the budget.
It counts one run of the server, and starts from zero at the next.

Answers are kept in `~/.cache/jev/answers.sqlite`, so adding a column you have added before makes
no calls. That is the file the sibling tools use, but the entries are not shared: jgrep and jsort
now key their answers by endpoint as well, and jcol keys them by model ID, text and question only.
If you point jcol at another endpoint with `JEV_URL`, use `--no-cache` or another `XDG_CACHE_HOME`.

## Limits

- It is a demonstration. There are offline tests, but nothing here measures how often Jev's answers
  about this data are right. Read some rows before you believe a column. "compare with…" is a quick
  check where the table already has a label.
- What jgrep's README says under [Things to know](https://github.com/keltokhy/jgrep#how-well-does-it-work)
  applies here. Jev answers the description you wrote, not the one you meant. It is close to
  deterministic, not exactly so, and the cache is what makes a rerun exact. Text in a row can try to
  steer its own answer, so do not use a column as a security boundary.
- The default model ID is an alias for the latest Jev, and the cache is keyed on the ID you asked
  for. When the alias moves, answers cached under it are still served. For results that must
  reproduce, pin a model with `--model` (for example `typesafe/jev-1.13` on OpenRouter).
- Only the first 4,000 characters of a text are sent, without a warning, and there is no option to
  change that. A missing text is sent as an empty one.
- The whole table is held in memory, and the page is sent a 320-character preview of every row and
  every field up front, then sorts and filters in the browser. It is meant for thousands of rows,
  not millions. `--limit` takes the first N.
- Columns live in the server's memory and are gone when it stops. The answers stay in the cache, so
  typing the header again refills the column without new calls. To keep a column, export it; the
  export carries only the first 320 characters of each text, so join it back on a field such as
  `complaint_id`.
- A cell whose call failed on every pass stays blank, and the page does not show the error count.
  The running estimate stops advancing at the first blank row in the random order.
- One person at a time. Every tab shares the same columns and the same single preview. The server
  has no authentication, which is why it listens on 127.0.0.1 only.

## Development

```bash
uv sync && uv run pytest        # offline: a fake API and a fake pool; no key, and no test can reach the network
```

The scripts in `spike/` are the experiments the design came from. They call the live API and spend
money, and they were written to be run once. `latency.py` times a screenful of calls with and
without re-sending, latency by text length, and throughput by calls in flight; it reads
`spike/texts.json`, which is not in the repository, and uses `spike/core.py`, the client as it was
before HTTP/2 and re-sending. `shard.py` measures throughput by the number of client processes on
the sample. `drive.py` drives a running server over its WebSocket the way the page does, and times a
preview, a kept column, and a preview typed during the background fill.

`src/jcol/spec.py` is the header grammar. `engine.py` is the scheduler: the two lanes, the random
order, shared calls and what is sent to the page. `pool.py` is where calls run, in worker processes
or in-process. `core.py` began as the client shared by jgrep and jlink (backends, retries inside a
time budget, the cache, in-flight deduplication and the cost meter) and adds HTTP/2 and re-sent
calls. `app.py` is the server and the command line. `static/index.html` is the page.

MIT license.

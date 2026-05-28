# Download Progress Bar — Design

Date: 2026-05-28
Status: Approved (brainstorm)

## Purpose

Show progress while `adrs` downloads data so users running scripts/notebooks can see what is happening during long fetches. Two-level visibility: which topic is being fetched (outer), and how far the current topic's API pagination has progressed (inner).

## Scope

In:
- Progress UI for downloads triggered via `Datamap.init` (the primary entry point for multi-topic data loading).
- Inner per-page progress in `Datasource.query_paginated` for both range mode and limit mode.

Out:
- Progress for `Datasource.query` direct calls outside a `Datamap.init` context (no UI; behavior unchanged).
- yfinance handler (downloads in one synchronous `yf.Ticker.history` call; no meaningful pagination to track).
- Persistence/logging of progress events.

## Architecture

One shared `rich.progress.Progress` instance per `Datamap.init` call. Shared so the two concurrent layers (outer topic tasks, inner pagination tasks across topics fetched in a `TaskGroup`) render together in a single live region. Sharing is achieved via a `contextvars.ContextVar` so the inner layer (`Datasource.query_paginated`) can discover the Progress instance without threading it through `DataLoader → Cache → Datasource`.

### Components

1. `adrs/data/progress.py` (new)
   - `_progress_var: ContextVar[Progress | None]` — holds the active Progress instance.
   - `progress_context(total_topics: int)` — async context manager. Creates a `Progress` (disabled when `not sys.stdout.isatty()`), starts it, sets the contextvar, yields a handle exposing the outer `TaskID`. On exit, stops Progress and resets the contextvar. Cleanup runs in `finally` so an exception in the body still stops the live renderer.
   - `inner_task(description: str, total: int | None)` — async context manager. Reads contextvar; if `None`, yields a no-op handle (`advance` is a no-op). Else adds a task on the shared Progress, yields a handle with `advance(n)`, removes the task on exit.

2. `adrs/data/datamap.py` (edit)
   - In `Datamap.init`, wrap the `TaskGroup` block in `async with progress_context(len(self.topics)) as outer:`.
   - After each `_init` task completes, call `outer.advance(1)`. Implementation: wrap the call in a small inline coroutine that awaits `_init(...)` and then advances; submit *that* to the `TaskGroup` instead of `_init` directly.

3. `adrs/data/datasource.py` (edit)
   - In `query_paginated`, after resolving args and computing `interval_ms` / `total` / `limit`, compute `inner_total`:
     - Range mode, non-block: `inner_total = (end_ms - int(start_time.timestamp() * 1000)) // interval_ms`.
     - Limit mode: `inner_total = limit`.
     - Block topics (`topic.is_block()`): `inner_total = None` (indeterminate; rich shows a spinner).
   - Open `async with inner_task(str(topic), inner_total) as bar:` around the pagination loop. Call `bar.advance(num)` after each page (where `num = len(resp)`).
   - When `inner_total` is `None`, `advance` calls still drive the spinner cadence (rich treats advance on `total=None` as a pulse).

4. `pyproject.toml` (edit)
   - Promote `rich` to a direct dependency (currently transitive). Pin a minimum compatible with Python 3.x in use.

### Progress display

Columns (rich.progress defaults that fit a two-tier view):

```
SpinnerColumn(),
TextColumn("[progress.description]{task.description}"),
BarColumn(),
TaskProgressColumn(),  # "n/total" or "%"
TimeRemainingColumn(),
```

Outer task description: `"topics"`. Inner task description: the `str(topic)`.

## Data flow

```
Datamap.init(infos, start, end)
  └─ async with progress_context(len(topics)) as outer:
       async with TaskGroup() as tg:
         for topic in topics:
           tg.create_task(_init_and_tick(topic))   # wrapper that calls _init then outer.advance(1)
       # exit Progress (stops live render, clears bars or leaves them per rich default)

  _init(topic) → dataloader.load → cache.fetch → cache.download → datasource.query_paginated
                                                                   └─ async with inner_task(str(topic), inner_total) as bar:
                                                                        while not done:
                                                                          resp = await self.query(...)
                                                                          bar.advance(len(resp))
```

## Error handling

- `progress_context` cleans up via `finally`. Any exception inside the TaskGroup still stops the live region and resets the contextvar.
- `inner_task` is a no-op when the contextvar is unset (e.g. `query_paginated` called from a script or test outside `Datamap.init`). No new error paths.
- Block topics with unknown total: inner bar runs as indeterminate (`total=None`). User still sees activity; no division by zero or false "100%".
- Empty downloads (`current_start >= end_time` from the start): `inner_task` opens and closes immediately; bar shows then disappears. Outer bar still advances when the parent `_init` returns.

## Testing

- All existing tests must pass unchanged. Tests that exercise `Datamap.init` still create a `Progress`, but `disable=not sys.stdout.isatty()` suppresses output under pytest. Tests that call `query_paginated` directly do not set the contextvar, so the inner layer is a no-op.
- New unit test in `tests/data/test_progress.py`:
  1. Stub `Datasource` whose `query` returns N rows per call across M pages.
  2. Drive `query_paginated` inside `progress_context(1)` with an explicit inner task expectation.
  3. Assert the inner task's `completed` equals `N * M` and `total` equals the computed `inner_total`.
  4. Repeat with `topic.is_block()` true and assert `total is None`.
- Manual verification: run an existing example script that triggers a real download against the live API; confirm bars render in a TTY and are absent when stdout is piped to a file.

## Open questions

None at design time.

## Out-of-scope follow-ups

- Hook `cybotrade_handler` / direct `cache.fetch` callers (outside `Datamap.init`) into the same progress UI if they prove to be common entry points.
- Add a `--no-progress` env var or kwarg if users report the auto-disable on non-TTY is insufficient.

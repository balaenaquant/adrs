# Download Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a two-level progress UI (outer = topics, inner = paginated datapoints) while `adrs.data.Datamap.init` downloads market data, using `rich.progress`.

**Architecture:** A shared `rich.progress.Progress` instance is created in `Datamap.init` and exposed to the inner pagination loop in `Datasource.query_paginated` via a `contextvars.ContextVar`. The renderer auto-disables on non-TTY. Block topics show indeterminate inner bars.

**Tech Stack:** Python 3.12+, `rich` (promoted from transitive to direct dep), `polars`, `httpx`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-05-28-download-progress-bar-design.md`

**Code conventions:**
- The project uses `asyncio` + `pytest-asyncio` (see `tests/test_datamap.py`). All new tests are async with `@pytest.mark.asyncio`.
- Tests run via `uv run coverage run -m pytest`.
- The downloader entry point is `Datamap.init` (see `adrs/data/datamap.py:166`).
- The paginator is `Datasource.query_paginated` (see `adrs/data/datasource.py:47`).

---

## File Structure

- **New:** `adrs/data/progress.py` — contextvar, `progress_context`, `inner_task`, handle classes. ~80 lines, one responsibility.
- **New:** `tests/data/test_progress.py` — unit tests for the progress module and its wiring into the paginator (with a stub datasource, no network).
- **Modify:** `adrs/data/datasource.py` — open an inner task around the pagination loop in `query_paginated`.
- **Modify:** `adrs/data/datamap.py` — wrap `Datamap.init` body in `progress_context` and advance the outer bar as each `_init` finishes.
- **Modify:** `pyproject.toml` — promote `rich` to a direct dependency.

---

## Task 1: Add `rich` as direct dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add rich to dependencies**

Edit the `dependencies = [ ... ]` list in `pyproject.toml` to include `"rich>=13.0"`. Insert in alphabetical order between `"pydantic>=2.11.7"` and `"websockets>=13.0"`.

Resulting block:

```toml
dependencies = [
    "bq-aion-rs>=0.1.0",
    "colorlog>=6.10.1",
    "cybotrade-datasource>=0.1.9",
    "httpx>=0.28.1",
    "nats-py>=2.14.0",
    "numpy>=2.3.1",
    "pandas>=2.3.0",
    "pandera>=0.26.0",
    "polars>=1.35.0",
    "pyarrow>=22.0.0",
    "pydantic>=2.11.7",
    "rich>=13.0",
    "websockets>=13.0",
    "yfinance>=1.2.0",
]
```

- [ ] **Step 2: Sync and verify import**

Run:

```bash
uv sync
uv run python -c "from rich.progress import Progress; print(Progress)"
```

Expected: prints `<class 'rich.progress.Progress'>` with no error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: promote rich to direct dependency"
```

---

## Task 2: Progress module — handle classes

**Files:**
- Create: `adrs/data/progress.py`
- Create: `tests/data/test_progress.py`

We build the module in two stages. Stage A: a no-op handle and a `ContextVar`. Stage B: real Progress and `inner_task` (Task 3). This split keeps the first round of tests free of any rich state.

- [ ] **Step 1: Write the failing test for the no-op inner handle**

Create `tests/data/test_progress.py`:

```python
import pytest

from adrs.data.progress import inner_task, _progress_var


@pytest.mark.asyncio
async def test_inner_task_no_op_when_contextvar_unset():
    """Outside progress_context, inner_task must be a no-op that still yields a usable handle."""
    assert _progress_var.get() is None

    async with inner_task("topic-x", total=10) as bar:
        bar.advance(3)
        bar.advance(7)
        assert bar.completed == 0  # no-op handle does not track
        assert bar.total is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/data/test_progress.py::test_inner_task_no_op_when_contextvar_unset -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'adrs.data.progress'`.

- [ ] **Step 3: Create the module with the no-op path**

Create `adrs/data/progress.py`:

```python
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


_progress_var: ContextVar["Progress | None"] = ContextVar(
    "adrs_progress", default=None
)


@dataclass
class _NoOpBar:
    completed: int = 0
    total: int | None = None

    def advance(self, n: int) -> None:
        return None


@dataclass
class _InnerBar:
    progress: Progress
    task_id: TaskID
    completed: int = 0
    total: int | None = None

    def advance(self, n: int) -> None:
        self.progress.advance(self.task_id, n)
        self.completed += n


@asynccontextmanager
async def inner_task(
    description: str, total: int | None
) -> AsyncIterator[_NoOpBar | _InnerBar]:
    progress = _progress_var.get()
    if progress is None:
        yield _NoOpBar(completed=0, total=None)
        return

    task_id = progress.add_task(description, total=total)
    bar = _InnerBar(progress=progress, task_id=task_id, completed=0, total=total)
    try:
        yield bar
    finally:
        progress.remove_task(task_id)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/data/test_progress.py::test_inner_task_no_op_when_contextvar_unset -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adrs/data/progress.py tests/data/test_progress.py
git commit -m "feat(data): add progress module skeleton with no-op fallback"
```

---

## Task 3: Progress module — `progress_context` and active `inner_task`

**Files:**
- Modify: `adrs/data/progress.py`
- Modify: `tests/data/test_progress.py`

- [ ] **Step 1: Write the failing test for `progress_context`**

Append to `tests/data/test_progress.py`:

```python
from adrs.data.progress import progress_context


@pytest.mark.asyncio
async def test_progress_context_sets_and_resets_contextvar():
    """progress_context sets the contextvar inside the body and resets on exit."""
    assert _progress_var.get() is None
    async with progress_context(total_topics=2) as outer:
        assert _progress_var.get() is not None
        outer.advance(1)
        outer.advance(1)
        assert outer.completed == 2
        assert outer.total == 2
    assert _progress_var.get() is None


@pytest.mark.asyncio
async def test_inner_task_tracks_when_active():
    """Inside progress_context, inner_task records advances on its handle."""
    async with progress_context(total_topics=1):
        async with inner_task("topic-y", total=5) as bar:
            bar.advance(2)
            bar.advance(3)
            assert bar.completed == 5
            assert bar.total == 5


@pytest.mark.asyncio
async def test_inner_task_indeterminate():
    """total=None is allowed (block topics); advance still updates the handle."""
    async with progress_context(total_topics=1):
        async with inner_task("topic-z", total=None) as bar:
            bar.advance(123)
            assert bar.completed == 123
            assert bar.total is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_progress.py -v
```

Expected: the three new tests FAIL with `ImportError` (no `progress_context`) or `AttributeError`. The first test still passes.

- [ ] **Step 3: Implement `progress_context` and the outer handle**

Add the following to `adrs/data/progress.py` below the existing `_InnerBar` dataclass (and update the `inner_task` return type annotation if needed):

```python
@dataclass
class _OuterBar:
    progress: Progress
    task_id: TaskID
    completed: int = 0
    total: int = 0

    def advance(self, n: int = 1) -> None:
        self.progress.advance(self.task_id, n)
        self.completed += n


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        disable=not sys.stdout.isatty(),
    )


@asynccontextmanager
async def progress_context(total_topics: int) -> AsyncIterator[_OuterBar]:
    progress = _make_progress()
    progress.start()
    token = _progress_var.set(progress)
    outer_id = progress.add_task("topics", total=total_topics)
    outer = _OuterBar(
        progress=progress, task_id=outer_id, completed=0, total=total_topics
    )
    try:
        yield outer
    finally:
        _progress_var.reset(token)
        progress.stop()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/data/test_progress.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add adrs/data/progress.py tests/data/test_progress.py
git commit -m "feat(data): add progress_context with outer task and inner_task tracking"
```

---

## Task 4: Wire inner bar into `Datasource.query_paginated`

**Files:**
- Modify: `adrs/data/datasource.py:47-199`
- Modify: `tests/data/test_progress.py`

This task adds an `inner_task` around the pagination loop and verifies the per-page advance using a stub `Datasource` subclass, so no network is required.

- [ ] **Step 1: Write the failing test using a stub datasource (range mode)**

Append to `tests/data/test_progress.py`:

```python
from datetime import datetime, timezone

from adrs.data.datasource import Datasource


class _StubDatasource(Datasource):
    """Returns deterministic pages: each call returns `page_size` rows starting from the requested start_time."""

    def __init__(self, page_size: int, interval_ms: int):
        super().__init__(api_key="", base_url="", max_limit=page_size)
        self.page_size = page_size
        self.interval_ms = interval_ms
        self.calls = 0

    async def query(self, topic, start_time=None, end_time=None, limit=None, flatten=False):
        self.calls += 1
        # produce `limit` rows stepping by interval_ms starting at start_time
        return [
            {
                "start_time": datetime.fromtimestamp(
                    start_time.timestamp() + (i * self.interval_ms / 1000),
                    tz=timezone.utc,
                ),
                "value": float(i),
            }
            for i in range(limit)
        ]


@pytest.mark.asyncio
async def test_query_paginated_advances_inner_bar(monkeypatch):
    """Range mode: inner bar total equals total intervals; completed equals returned rows."""
    from adrs.data import progress as progress_mod

    # Patch _make_progress so the live renderer doesn't paint during tests
    monkeypatch.setattr(
        progress_mod,
        "_make_progress",
        lambda: progress_mod.Progress(disable=True),
    )

    interval_ms = 60_000  # 1 minute
    page_size = 10
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 0, 30, 0, tzinfo=timezone.utc)  # 30 intervals

    ds = _StubDatasource(page_size=page_size, interval_ms=interval_ms)
    captured: dict = {}

    # Patch inner_task so we can observe the bar created by query_paginated
    real_inner_task = progress_mod.inner_task

    @progress_mod.asynccontextmanager
    async def spying_inner_task(description, total):
        async with real_inner_task(description, total) as bar:
            captured["bar"] = bar
            yield bar

    monkeypatch.setattr(progress_mod, "inner_task", spying_inner_task)
    # query_paginated imports inner_task at module load; patch the bound name too
    import adrs.data.datasource as ds_mod
    monkeypatch.setattr(ds_mod, "inner_task", spying_inner_task)

    async with progress_mod.progress_context(total_topics=1):
        await ds.query_paginated(
            topic="binance-linear|candle?symbol=BTCUSDT&interval=1m",
            start_time=start,
            end_time=end,
        )

    bar = captured["bar"]
    assert bar.total == 30  # 30 one-minute intervals
    assert bar.completed == 30


@pytest.mark.asyncio
async def test_query_paginated_block_topic_indeterminate(monkeypatch):
    """Block topics have total=None (indeterminate) and advance still works."""
    from adrs.data import progress as progress_mod
    import adrs.data.datasource as ds_mod

    monkeypatch.setattr(
        progress_mod, "_make_progress", lambda: progress_mod.Progress(disable=True)
    )

    captured: dict = {}
    real_inner_task = progress_mod.inner_task

    @progress_mod.asynccontextmanager
    async def spying_inner_task(description, total):
        async with real_inner_task(description, total) as bar:
            captured["bar"] = bar
            yield bar

    monkeypatch.setattr(progress_mod, "inner_task", spying_inner_task)
    monkeypatch.setattr(ds_mod, "inner_task", spying_inner_task)

    ds = _StubDatasource(page_size=5, interval_ms=1000)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

    async with progress_mod.progress_context(total_topics=1):
        await ds.query_paginated(
            topic="cryptoquant|eth/exchange-flows/netflow?exchange=coinbase_advanced&window=block",
            start_time=start,
            end_time=end,
        )

    bar = captured["bar"]
    assert bar.total is None
    assert bar.completed > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_progress.py::test_query_paginated_advances_inner_bar tests/data/test_progress.py::test_query_paginated_block_topic_indeterminate -v
```

Expected: FAIL — `query_paginated` does not import or call `inner_task` yet.

- [ ] **Step 3: Wire `inner_task` into `query_paginated`**

Edit `adrs/data/datasource.py`. Add to imports near the top:

```python
from adrs.data.progress import inner_task
```

Then modify `query_paginated`. Inside `query_paginated`, immediately after the existing block that resolves `interval_ms` and instantiates `datas = SortedDataList()`, compute `inner_total`:

```python
        if start_time and end_time:
            end_ms_compute = (
                (int(end_time.timestamp() * 1000) // interval_ms) * interval_ms
            )
            inner_total: int | None = (
                None
                if topic.is_block()
                else max(
                    0,
                    (end_ms_compute - int(start_time.timestamp() * 1000)) // interval_ms,
                )
            )
        else:
            inner_total = None if topic.is_block() else limit
```

Then wrap the existing pagination logic (both the range-mode `while current_start < end_time:` block AND the limit-mode `while remaining > 0:` block) inside a single `async with inner_task(str(topic), inner_total) as _bar:` block. Inside each loop, immediately after `num = len(resp)`, add:

```python
                _bar.advance(num)
```

The resulting `query_paginated` structure (showing only the changed shape, with existing code abbreviated as `...`):

```python
    async def query_paginated(self, topic, start_time=None, end_time=None, limit=None, flatten=False):
        # ... existing arg resolution unchanged ...
        interval_ms = int(interval.total_seconds() * 1000)
        datas = SortedDataList()

        if start_time and end_time:
            end_ms_compute = (
                (int(end_time.timestamp() * 1000) // interval_ms) * interval_ms
            )
            inner_total: int | None = (
                None
                if topic.is_block()
                else max(
                    0,
                    (end_ms_compute - int(start_time.timestamp() * 1000)) // interval_ms,
                )
            )
        else:
            inner_total = None if topic.is_block() else limit

        async with inner_task(str(topic), inner_total) as _bar:
            if start_time and end_time:
                # ... existing range-mode loop, with `_bar.advance(num)` after `num = len(resp)` ...
            else:
                # ... existing limit-mode loop, with `_bar.advance(num)` after `num = len(resp)` ...

        if not datas.data:
            return pl.DataFrame()
        return datas.to_df()
```

Do not change any other behavior. Existing assertions on `resp[0]` keys remain inside the loop.

- [ ] **Step 4: Run the new tests to verify they pass**

Run:

```bash
uv run pytest tests/data/test_progress.py -v
```

Expected: all six tests in this file PASS.

- [ ] **Step 5: Run the full suite to verify no regression**

Run:

```bash
uv run pytest tests/data/test_progress.py tests/test_datamap.py -v
```

Expected: all PASS. (Other tests under `tests/test_datasource.py` hit the live API and may not be runnable locally — skip them if credentials are absent.)

- [ ] **Step 6: Commit**

```bash
git add adrs/data/datasource.py tests/data/test_progress.py
git commit -m "feat(data): show per-page inner progress bar in query_paginated"
```

---

## Task 5: Wire outer bar into `Datamap.init`

**Files:**
- Modify: `adrs/data/datamap.py:166-180`
- Modify: `tests/data/test_progress.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_progress.py`:

```python
@pytest.mark.asyncio
async def test_datamap_init_advances_outer_bar(monkeypatch):
    """Datamap.init advances the outer bar once per topic completed.

    We monkey-patch Datamap._init to a no-op so this test is purely about wiring
    of the outer bar — it does not exercise the network/cache stack.
    """
    from adrs.data import progress as progress_mod
    from adrs.data.datamap import Datamap
    from adrs.data.types import DataInfo, DataColumn

    captured: dict = {}
    real_progress_context = progress_mod.progress_context

    @progress_mod.asynccontextmanager
    async def spying_progress_context(total_topics):
        async with real_progress_context(total_topics) as outer:
            captured["outer"] = outer
            yield outer

    monkeypatch.setattr(
        progress_mod, "_make_progress", lambda: progress_mod.Progress(disable=True)
    )
    monkeypatch.setattr(progress_mod, "progress_context", spying_progress_context)
    import adrs.data.datamap as dm_mod
    monkeypatch.setattr(dm_mod, "progress_context", spying_progress_context)

    async def _noop_init(self, dataloader, topic, start_time, end_time, should_lookback=True):
        return None

    monkeypatch.setattr(Datamap, "_init", _noop_init)

    infos = [
        DataInfo(
            topic="cryptoquant|btc/market-data/price-ohlcv?exchange=binance&market=spot&window=hour",
            columns=[DataColumn(src="value", dst="v")],
            lookback_size=1,
        ),
        DataInfo(
            topic="cryptoquant|eth/market-data/price-ohlcv?exchange=binance&market=spot&window=hour",
            columns=[DataColumn(src="value", dst="v")],
            lookback_size=1,
        ),
    ]

    dm = Datamap()
    await dm.init(
        dataloader=None,  # _init is patched to no-op so this is never used
        infos=infos,
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    outer = captured["outer"]
    assert outer.total == 2
    assert outer.completed == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/data/test_progress.py::test_datamap_init_advances_outer_bar -v
```

Expected: FAIL — `Datamap.init` does not import or use `progress_context` yet.

- [ ] **Step 3: Wire `progress_context` into `Datamap.init`**

Edit `adrs/data/datamap.py`. Add to imports:

```python
from adrs.data.progress import progress_context
```

Replace the body of `Datamap.init` (currently lines 166-180) with:

```python
    async def init(
        self,
        dataloader: DataLoader,
        infos: list[DataInfo],
        start_time: datetime,
        end_time: datetime,
        should_lookback: bool = True,
    ):
        self.data_infos = dedup_data_infos_by_max_lookback_size(infos + self.data_infos)
        self.topics = {Topic.from_str(data_info.topic) for data_info in self.data_infos}

        async with progress_context(total_topics=len(self.topics)) as outer:
            async def _init_and_tick(topic: Topic) -> None:
                await self._init(
                    dataloader, topic, start_time, end_time, should_lookback
                )
                outer.advance(1)

            async with asyncio.TaskGroup() as tg:
                for topic in self.topics:
                    tg.create_task(_init_and_tick(topic))
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
uv run pytest tests/data/test_progress.py::test_datamap_init_advances_outer_bar -v
```

Expected: PASS.

- [ ] **Step 5: Run the broader datamap suite to verify no regression**

Run:

```bash
uv run pytest tests/test_datamap.py tests/data/test_progress.py -v
```

Expected: all PASS (network-dependent tests in `test_datamap.py` may be skipped if credentials are missing — that is not a regression introduced here).

- [ ] **Step 6: Commit**

```bash
git add adrs/data/datamap.py tests/data/test_progress.py
git commit -m "feat(data): show outer per-topic progress bar in Datamap.init"
```

---

## Task 6: Manual TTY verification

**Files:**
- None modified

- [ ] **Step 1: Run an example end-to-end against the real API**

```bash
uv run python -c "
import asyncio
from datetime import datetime, timezone
from adrs.data.datamap import Datamap
from adrs.data.dataloader import DataLoader
from adrs.data.types import DataInfo, DataColumn

async def main():
    dl = DataLoader(data_dir='data', credentials=__import__('json').load(open('credentials.json')))
    infos = [
        DataInfo(
            topic='cryptoquant|btc/market-data/price-ohlcv?exchange=binance&market=spot&window=hour',
            columns=[DataColumn(src='close', dst='price')],
            lookback_size=1,
        ),
    ]
    dm = Datamap()
    await dm.init(dl, infos, datetime(2025,1,1,tzinfo=timezone.utc), datetime(2025,1,8,tzinfo=timezone.utc))

asyncio.run(main())
"
```

Expected when run in a real terminal: bars render with a "topics" row and a transient inner row per topic. When stdout is piped (e.g. `... | cat`), no bars should appear.

- [ ] **Step 2: Pipe the same command through cat and confirm no progress UI**

```bash
uv run python -c "..." | cat
```

Expected: no `\r`-driven progress lines, only normal log output. (Reuse the snippet from Step 1.)

- [ ] **Step 3: Run typecheck and full test suite**

```bash
just check
just test
```

Expected: `just check` succeeds; `just test` passes for all tests that do not require live API credentials.

---

## Done

When all tasks above are checked, the spec is fully implemented:

- Outer bar in `Datamap.init` → covered by Tasks 3 + 5.
- Inner bar in `Datasource.query_paginated` (range + limit + block) → covered by Tasks 2 + 3 + 4.
- Contextvar bridge without arg threading → covered by Tasks 2 + 3.
- TTY auto-detect → covered by Task 3 (`disable=not sys.stdout.isatty()`).
- `rich` promoted to direct dep → Task 1.
- No new behavior when contextvar is unset → covered by Task 2 (no-op path test).
- Manual TTY check → Task 6.

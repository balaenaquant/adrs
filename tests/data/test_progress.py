import pytest
from datetime import datetime, timezone

from adrs.data.datasource import Datasource
from adrs.data.progress import inner_task, _progress_var, progress_context


@pytest.mark.asyncio
async def test_inner_task_no_op_when_contextvar_unset():
    """Outside progress_context, inner_task must be a no-op that still yields a usable handle."""
    assert _progress_var.get() is None

    async with inner_task("topic-x", total=10) as bar:
        bar.advance(3)
        bar.advance(7)
        assert bar.completed == 0  # no-op handle does not track
        assert bar.total is None


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


class _StubDatasource(Datasource):
    """Returns deterministic pages: each call returns `page_size` rows starting from the requested start_time."""

    def __init__(self, page_size: int, interval_ms: int):
        super().__init__(api_key="", base_url="", max_limit=page_size)
        self.page_size = page_size
        self.interval_ms = interval_ms
        self.calls = 0

    async def query(
        self, topic, start_time=None, end_time=None, limit=None, flatten=False
    ):
        self.calls += 1
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

    real_inner_task = progress_mod.inner_task

    @progress_mod.asynccontextmanager
    async def spying_inner_task(description, total):
        async with real_inner_task(description, total) as bar:
            captured["bar"] = bar
            yield bar

    monkeypatch.setattr(progress_mod, "inner_task", spying_inner_task)
    import adrs.data.datasource as ds_mod

    monkeypatch.setattr(ds_mod, "inner_task", spying_inner_task)

    async with progress_mod.progress_context(total_topics=1):
        await ds.query_paginated(
            topic="binance-linear|candle?symbol=BTCUSDT&interval=1m",
            start_time=start,
            end_time=end,
        )

    bar = captured["bar"]
    assert bar.total == 30
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

    ds = _StubDatasource(page_size=5, interval_ms=600_000)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 0, 20, 0, tzinfo=timezone.utc)  # 2 block intervals

    async with progress_mod.progress_context(total_topics=1):
        await ds.query_paginated(
            topic="cryptoquant|eth/exchange-flows/netflow?exchange=coinbase_advanced&window=block",
            start_time=start,
            end_time=end,
        )

    bar = captured["bar"]
    assert bar.total is None
    assert bar.completed > 0


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
    import adrs.data.datamap as dm_mod

    monkeypatch.setattr(dm_mod, "progress_context", spying_progress_context)

    async def _noop_init(
        self, dataloader, topic, start_time, end_time, should_lookback=True
    ):
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
        dataloader=None,
        infos=infos,
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    outer = captured["outer"]
    assert outer.total == 2
    assert outer.completed == 2

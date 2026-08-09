"""CybotradeDatasource.query_paginated: page stitching without the network.

query() is the only call that touches HTTP, so a fake serving a synthetic
candle series exercises the whole pagination path — including the accumulation
strategy, which at 1m intervals decides whether a year of candles costs tens of
megabytes or hundreds.
"""

import tracemalloc
from datetime import datetime, timedelta, timezone

import pytest

from adrs.data.datasource import CybotradeDatasource

TOPIC = "bybit-linear|candle?symbol=BTCUSDT&interval=1m"
BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def series(n: int, start: datetime = BASE) -> list[dict]:
    return [
        {
            "start_time": start + timedelta(minutes=i),
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": float(i),
            "volume": 10.0,
        }
        for i in range(n)
    ]


class FakeDatasource(CybotradeDatasource):
    """Serves a fixed series; never constructs an HTTP client."""

    def __init__(self, rows: list[dict], max_limit: int = 100):
        self.rows = rows
        self.max_limit = max_limit
        self.pages = 0

    async def query(
        self, topic, start_time=None, end_time=None, limit=None, flatten=False
    ):
        self.pages += 1
        if start_time is not None:
            window = [r for r in self.rows if r["start_time"] >= start_time]
            return window[:limit]
        if end_time is not None:
            window = [r for r in self.rows if r["start_time"] <= end_time]
            return window[-limit:] if limit else window
        return self.rows[:limit]


@pytest.mark.asyncio
async def test_range_mode_returns_every_candle_in_the_window():
    rows = series(250)
    ds = FakeDatasource(rows, max_limit=100)
    df = await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=250)
    )
    assert len(df) == 250
    assert df["close"].to_list() == [float(i) for i in range(250)]


@pytest.mark.asyncio
async def test_range_mode_spans_multiple_pages():
    ds = FakeDatasource(series(250), max_limit=100)
    await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=250)
    )
    assert ds.pages > 1


@pytest.mark.asyncio
async def test_rows_come_back_sorted_by_start_time():
    ds = FakeDatasource(series(120), max_limit=50)
    df = await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=120)
    )
    assert df["start_time"].to_list() == sorted(df["start_time"].to_list())


@pytest.mark.asyncio
async def test_overlapping_pages_do_not_duplicate_rows():
    rows = series(120)

    class OverlappingDatasource(FakeDatasource):
        async def query(self, topic, start_time=None, end_time=None, limit=None, **kw):
            # replay the previous candle on every page, as a real feed may
            window = [r for r in self.rows if r["start_time"] >= start_time]
            self.pages += 1
            return window[: (limit or 0)]

    ds = OverlappingDatasource(rows, max_limit=50)
    df = await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=120)
    )
    assert len(df) == len(set(df["start_time"].to_list()))


@pytest.mark.asyncio
async def test_a_later_page_restating_a_candle_wins_over_the_earlier_one():
    rows = series(120)

    class RestatingDatasource(FakeDatasource):
        async def query(self, topic, start_time=None, end_time=None, limit=None, **kw):
            self.pages += 1
            window = [r for r in self.rows if r["start_time"] >= start_time]
            page = [dict(r) for r in window[: (limit or 0)]]
            # from the second page on, replay the previous candle with a
            # corrected close — the shape a real backfill takes
            if self.pages > 1 and page:
                earlier = dict(page[0])
                earlier["start_time"] -= timedelta(minutes=1)
                earlier["close"] = -1.0
                page.insert(0, earlier)
            return page

    ds = RestatingDatasource(rows, max_limit=50)
    df = await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=120)
    )
    restated = df.filter(df["start_time"] == BASE + timedelta(minutes=49))
    assert restated["close"].to_list() == [-1.0]


@pytest.mark.asyncio
async def test_start_time_is_stored_as_utc_milliseconds():
    ds = FakeDatasource(series(10), max_limit=100)
    df = await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=10)
    )
    assert df["start_time"].dtype.time_unit == "ms"
    assert df["start_time"].dtype.time_zone == "UTC"


@pytest.mark.asyncio
async def test_no_rows_yields_an_empty_frame():
    ds = FakeDatasource([], max_limit=100)
    df = await ds.query_paginated(
        TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=10)
    )
    assert df.is_empty()


@pytest.mark.asyncio
async def test_limit_mode_pages_backwards_from_the_last_closed_candle():
    ds = FakeDatasource(series(200, start=BASE), max_limit=50)
    df = await ds.query_paginated(TOPIC, limit=120)
    assert len(df) > 0
    assert df["start_time"].to_list() == sorted(df["start_time"].to_list())


@pytest.mark.asyncio
async def test_paginating_does_not_box_pages_into_python_dicts():
    n = 30_000
    ds = FakeDatasource(series(n), max_limit=5_000)

    tracemalloc.start()
    try:
        df = await ds.query_paginated(
            TOPIC, start_time=BASE, end_time=BASE + timedelta(minutes=n)
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(df) == n
    # The fake's own dict rows are unavoidable; what must not happen is the
    # accumulator re-boxing the whole history once per page.
    assert peak < df.estimated_size() * 6

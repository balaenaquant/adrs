import json
import pytest
from datetime import datetime, timezone

from adrs.data.datasource import CybotradeDatasource


def api_key() -> str:
    return json.load(open("credentials.json"))["cybotrade_api_key"]


@pytest.mark.asyncio
async def test_query_paginated_limit_block():
    """(end_time, limit) — block data, expects 2 queries"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|eth/exchange-flows/netflow?exchange=coinbase_advanced&window=block",
        limit=200_000,
    )
    assert len(data) > 0
    assert all("start_time" in d for d in data)


@pytest.mark.asyncio
async def test_query_paginated_limit_single_query():
    """(end_time, limit) — 1 query"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|eth/exchange-flows/netflow?exchange=coinbase_advanced&window=hour",
        limit=100_000,
    )
    assert len(data) > 0


@pytest.mark.asyncio
async def test_query_paginated_limit_two_queries():
    """(end_time, limit) — 2 queries"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|btc/market-data/liquidations?exchange=deribit&window=min",
        limit=200_000,
    )
    assert len(data) > 0


@pytest.mark.asyncio
async def test_query_paginated_start_end_single_query():
    """(start_time, end_time) — 1 query"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|btc/market-data/liquidations?exchange=deribit&window=min",
        start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    assert len(data) > 0


@pytest.mark.asyncio
async def test_query_paginated_start_end_two_queries():
    """(start_time, end_time) — 2 queries"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|btc/market-data/liquidations?exchange=deribit&window=min",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    assert len(data) > 0


@pytest.mark.asyncio
async def test_query_paginated_exact_count():
    """(start_time, end_time) — assert exact datapoint count"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|eth/exchange-flows/netflow?exchange=binance&window=hour",
        start_time=datetime(2021, 1, 19, tzinfo=timezone.utc),
        end_time=datetime(2025, 3, 27, 6, 58, tzinfo=timezone.utc),
    )
    assert len(data) >= 36678  # count at time of writing; may grow with backfill


@pytest.mark.asyncio
async def test_query_paginated_start_end_block():
    """(start_time, end_time) — block data"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="cryptoquant|eth/exchange-flows/netflow?exchange=coinbase_advanced&window=block",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    assert len(data) > 0


@pytest.mark.asyncio
async def test_query_paginated_more_than_100k():
    """(start_time, end_time) — more than 100k results, multi-page"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="binance-linear|candle?symbol=BTCUSDT&interval=15m",
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert len(data) > 100_000


@pytest.mark.asyncio
async def test_query_paginated_start_before_first_data():
    """(start_time, end_time) — start_time earlier than first available datapoint"""
    ds = CybotradeDatasource(api_key())
    data = await ds.query_paginated(
        topic="bybit-linear|candle?symbol=BTCUSDT&interval=15m",
        start_time=datetime(2017, 6, 1, tzinfo=timezone.utc),
        end_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert len(data) > 0


@pytest.mark.asyncio
async def test_query_single():
    """single query without pagination"""
    ds = CybotradeDatasource(api_key())
    resp = await ds._query(
        topic="cryptoquant|eth/exchange-flows/netflow?exchange=coinbase_advanced&window=block",
        limit=100_000,
    )
    assert "data" in resp
    assert "page" in resp
    assert len(resp["data"]) > 0

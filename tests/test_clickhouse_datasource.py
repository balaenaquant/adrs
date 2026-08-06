from datetime import datetime, timezone

import pandas as pd
import polars as pl
import pytest

from adrs.data.datasource import ClickhouseDatasource
from adrs.types import Topic


class _FakeAsyncClient:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def query_df(self, query: str) -> pd.DataFrame:
        return self._df


@pytest.mark.asyncio
async def test_query_one_range_normalizes_start_time_to_utc_ms():
    """ClickHouse's plain `DateTime` column carries no explicit tz, so
    clickhouse-connect -> pandas -> pl.from_pandas produces a tz-naive
    start_time. Regression for the bug where this leaked through
    unnormalized: cache.py's pl.concat of old (Cybotrade-sourced, tz-aware)
    and new (Clickhouse-sourced, tz-naive) cached shards for the same
    series raised `SchemaError: failed to determine supertype of
    datetime[ms] and datetime[ms, UTC]`.
    """
    naive_df = pd.DataFrame(
        {
            "start_time": pd.to_datetime(
                ["2026-01-01 00:00:00", "2026-01-01 01:00:00"]
            ),
            "data": [{"close": 1.0}, {"close": 2.0}],
        }
    )
    assert naive_df["start_time"].dt.tz is None

    ds = ClickhouseDatasource(host="unused")
    ds.ch = _FakeAsyncClient(naive_df)

    topic = Topic(
        provider="bybit-linear",
        endpoint="candle",
        params={"symbol": "BTCUSDT", "interval": "1h"},
    )

    df = await ds._query_one_range(
        topic,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert df.schema["start_time"] == pl.Datetime("ms", "UTC")
    assert df["close"].to_list() == [1.0, 2.0]

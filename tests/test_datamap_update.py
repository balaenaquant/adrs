"""Datamap's in-memory candle bookkeeping: ingest, restatement, lookback trim.

Network-free — these drive Datamap against hand-built rows so the store's
behaviour is pinned independently of the integration tests in test_datamap.py.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import polars as pl

from adrs.data.datamap import Datamap
from adrs.data.types import DataColumn, DataInfo
from adrs.types import Topic

TOPIC_STR = "bybit-linear|candle?symbol=BTCUSDT&interval=1m"
TOPIC = Topic.from_str(TOPIC_STR)
BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def info(lookback_size: int) -> DataInfo:
    return DataInfo(
        topic=TOPIC_STR,
        columns=[DataColumn(src="close", dst="price")],
        lookback_size=lookback_size,
    )


def datamap(lookback_size: int = 10) -> Datamap:
    return Datamap(data_infos=[info(lookback_size)])


def candle(minute: int, close: float) -> dict:
    return {"start_time": BASE + timedelta(minutes=minute), "close": close}


def closes(dm: Datamap) -> list[float]:
    return dm.map[TOPIC].to_df()["close"].to_list()


def test_update_stores_the_first_candle():
    dm = datamap()
    dm.update(TOPIC, candle(0, 1.0))
    assert closes(dm) == [1.0]


def test_update_appends_later_candles_in_order():
    dm = datamap()
    for i in range(3):
        dm.update(TOPIC, candle(i, float(i)))
    assert closes(dm) == [0.0, 1.0, 2.0]


def test_update_restates_the_last_candle_when_it_arrives_twice():
    dm = datamap()
    dm.update(TOPIC, candle(0, 1.0))
    dm.update(TOPIC, candle(1, 2.0))
    dm.update(TOPIC, candle(1, 99.0))  # same start_time — a restatement
    assert closes(dm) == [1.0, 99.0]


def test_update_trims_history_to_the_lookback_size():
    dm = datamap(lookback_size=3)
    for i in range(10):
        dm.update(TOPIC, candle(i, float(i)))
    assert closes(dm) == [7.0, 8.0, 9.0]


def test_is_ready_once_enough_candles_have_arrived():
    dm = datamap(lookback_size=3)
    dm.topics = {TOPIC}
    for i in range(2):
        dm.update(TOPIC, candle(i, float(i)))
    assert not dm.is_ready()
    dm.update(TOPIC, candle(2, 2.0))
    assert dm.is_ready()


def test_get_renames_source_columns_to_their_destinations():
    dm = datamap()
    dm.update(TOPIC, candle(0, 1.0))
    df = dm.get(info(lookback_size=10))
    assert df.columns == ["start_time", "price"]
    assert df["price"].to_list() == [1.0]


class _StubDataLoader:
    """Returns a fixed frame, standing in for the REST/cache fetch."""

    def __init__(self, df: pl.DataFrame):
        self.df = df

    async def load(self, topic, start_time, end_time, override_existing=False):
        return self.df


def test_resync_folds_fetched_rows_in_and_trims_to_the_lookback():
    dm = datamap(lookback_size=3)
    for i in range(3):
        dm.update(TOPIC, candle(i, float(i)))

    fetched = pl.DataFrame(
        {
            "start_time": [BASE + timedelta(minutes=i) for i in (2, 3, 4)],
            "close": [22.0, 3.0, 4.0],  # minute 2 restated by the authoritative fetch
        }
    )
    asyncio.run(dm.resync(TOPIC, _StubDataLoader(fetched)))

    assert closes(dm) == [22.0, 3.0, 4.0]


def test_ipc_round_trip_preserves_rows():
    dm = datamap()
    for i in range(3):
        dm.update(TOPIC, candle(i, float(i)))
    restored = Datamap.read_ipc(dm.write_ipc())
    assert restored.map[TOPIC].to_df().equals(dm.map[TOPIC].to_df())

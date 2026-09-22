"""DataProcessor.process drops duplicated start_time rows from the joined frame."""

from datetime import datetime, timedelta, timezone

import polars as pl

from adrs.data.datamap import Datamap
from adrs.data.processor import DataProcessor
from adrs.data.types import DataColumn, DataInfo
from adrs.types import SortedDataList, Topic

TOPIC_STR = "bybit-linear|candle?symbol=BTCUSDT&interval=1h"
TOPIC = Topic.from_str(TOPIC_STR)
BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _frame(hours, dup_hour=None):
    rows = [{"start_time": BASE + timedelta(hours=h), "close": float(h)} for h in range(hours)]
    if dup_hour is not None:
        rows.append({"start_time": BASE + timedelta(hours=dup_hour), "close": 999.0})
    return pl.DataFrame(rows).sort("start_time")


def _processor():
    info = DataInfo(topic=TOPIC_STR, columns=[DataColumn(src="close", dst="price")], lookback_size=8)
    dm = Datamap(data_infos=[info])
    proc = DataProcessor()
    proc.data_infos = [info]
    return dm, proc


def test_process_is_identity_on_a_clean_frame():
    dm, proc = _processor()
    dm.map[TOPIC] = SortedDataList.from_df(_frame(8))
    out = proc.process(datamap=dm)
    assert out.height == 8 and out["start_time"].n_unique() == 8


def test_process_drops_duplicate_timestamps_keeping_the_newest_row():
    dm, proc = _processor()
    dm.map[TOPIC] = SortedDataList.from_df(_frame(8, dup_hour=5))
    out = proc.process(datamap=dm)
    assert out.height == 8
    assert out["start_time"].n_unique() == 8
    assert out.filter(pl.col("start_time") == BASE + timedelta(hours=5))["price"].item() == 999.0

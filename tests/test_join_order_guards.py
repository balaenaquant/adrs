"""Order-dependent code after joins: DataProcessor topic merge and Drawdown dates.

Both join without a guaranteed row order in Polars (`maintain_order` defaults to
"none") and then rely on chronological rows: the processor reads `[-1]` as the
latest bar and alphas run rolling/shift on its output; Drawdown forward-fills the
drawdown start time. They now pass maintain_order="left"; these tests pin the
chronological result.
"""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from adrs.data.datamap import Datamap
from adrs.data.processor import DataProcessor
from adrs.data.types import DataColumn, DataInfo
from adrs.performance.metric import Drawdown
from adrs.types import SortedDataList, Topic

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
TOPICS = [
    "bybit-linear|candle?symbol=BTCUSDT&interval=1m",
    "bybit-linear|candle?symbol=ETHUSDT&interval=1m",
    "bybit-linear|candle?symbol=SOLUSDT&interval=1m",
]


def test_processor_merges_topics_in_chronological_order():
    n = 5_000
    infos, frames = [], []
    for k, topic in enumerate(TOPICS):
        # Each topic starts later than the previous one and misses some minutes,
        # so the inner join drops rows from every side.
        minutes = [m for m in range(k * 7, n) if (m + k) % 97 != 0]
        frames.append(
            pl.DataFrame(
                {
                    "start_time": [BASE + timedelta(minutes=m) for m in minutes],
                    "close": [float(m) + k / 10 for m in minutes],
                }
            )
        )
        infos.append(
            DataInfo(
                topic=topic,
                columns=[DataColumn(src="close", dst=f"c{k}")],
                lookback_size=n,
            )
        )

    dm = Datamap(data_infos=infos)
    for topic, frame in zip(TOPICS, frames):
        dm.map[Topic.from_str(topic)] = SortedDataList.from_df(frame)
    proc = DataProcessor()
    proc.data_infos = infos

    out = proc.process(datamap=dm)

    common = set(frames[0]["start_time"])
    for frame in frames[1:]:
        common &= set(frame["start_time"])
    assert out["start_time"].to_list() == sorted(common)
    minute = (out["start_time"] - BASE).dt.total_minutes().cast(pl.Float64)
    for k in range(len(TOPICS)):
        assert (out[f"c{k}"] - minute).to_list() == pytest.approx([k / 10] * out.height)


def test_drawdown_dates_on_a_hand_checked_equity_path():
    # peak 0.30 at t2, trough 0.05 at t4 (drawdown -0.25), first new high at t6
    equity = [0.0, 0.1, 0.3, 0.2, 0.05, 0.15, 0.35]
    pnl = [0.0] + [b - a for a, b in zip(equity, equity[1:])]
    times = [BASE + timedelta(hours=h) for h in range(len(equity))]
    df = pl.DataFrame({"start_time": times, "pnl": pnl, "equity": equity})

    result = Drawdown(interval=timedelta(hours=1)).compute(df)

    assert result["max_drawdown"] == pytest.approx(-0.25)
    assert result["max_drawdown_start_date"] == times[3]  # first bar below the peak
    assert result["max_drawdown_end_date"] == times[4]
    assert result["max_drawdown_recover_date"] == times[6]

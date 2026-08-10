"""Evaluator.eval exec_time column: the actual timestamp of the shifted price
each row is evaluated against."""

from datetime import datetime, timedelta, timezone

import polars as pl

from adrs.data.datamap import DataInfo, Datamap
from adrs.data.types import DataColumn
from adrs.performance import Evaluator
from adrs.types import SortedDataList, Topic

TOPIC = "binance-spot|candle?symbol=BTCUSDT&interval=1m"
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _setup(minutes: list[int]) -> tuple[Evaluator, Datamap, pl.LazyFrame]:
    """1-minute price rows at the given minute offsets, hourly long signal."""
    times = [START + timedelta(minutes=m) for m in minutes]
    prices = pl.DataFrame(
        {"start_time": times, "close": [100.0 + m for m in minutes]}
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone(None).dt.replace_time_zone("UTC")
    )
    datamap = Datamap()
    datamap.map[Topic.from_str(TOPIC)] = SortedDataList.from_df(prices)

    evaluator = Evaluator(
        assets={
            "BTC": DataInfo(
                topic=TOPIC,
                columns=[DataColumn(src="close", dst="price")],
                lookback_size=0,
            )
        }
    )
    hours = sorted({m // 60 for m in minutes})
    signal_lf = pl.DataFrame(
        {
            "start_time": [START + timedelta(hours=h) for h in hours],
            "signal": [1] * len(hours),
        }
    ).lazy()
    return evaluator, datamap, signal_lf


def _eval(evaluator, datamap, signal_lf, price_shift: int) -> pl.DataFrame:
    return evaluator.eval(
        signal_lf=signal_lf.with_columns(
            pl.col("start_time").dt.cast_time_unit("ms")
        ),
        base_asset="BTC",
        datamap=datamap,
        start_time=START,
        end_time=START + timedelta(hours=6),
        fees=0.0,
        interval=timedelta(hours=1),
        price_shift=price_shift,
    ).collect()


def test_exec_time_equals_start_of_last_row_when_unshifted():
    # full 1m grid over 3 hours
    evaluator, datamap, signal_lf = _setup(list(range(180)))
    df = _eval(evaluator, datamap, signal_lf, price_shift=0)
    # each hourly bar's price comes from its last minute row (:59)
    assert (
        df.select(
            (pl.col("exec_time") - pl.col("start_time") == timedelta(minutes=59))
            .all()
        ).item()
    )


def test_exec_time_reflects_row_shift_on_full_grid():
    evaluator, datamap, signal_lf = _setup(list(range(180)))
    df = _eval(evaluator, datamap, signal_lf, price_shift=70)
    # last row of bar t is t+59min; shifted 70 rows -> t+129min on a full grid
    assert (
        df.select(
            (pl.col("exec_time") - pl.col("start_time") == timedelta(minutes=129))
            .all()
        ).item()
    )
    # price value agrees with the exec_time row's price (price = 100 + minute)
    minutes = df.select(
        ((pl.col("exec_time") - pl.lit(START)).dt.total_minutes()).alias("m")
    )["m"]
    assert (df["price"] == pl.Series([100.0 + m for m in minutes])).all()


def test_exec_time_jumps_across_gaps():
    # remove minutes 60..119 (a one-hour hole after the first hour)
    minutes = [m for m in range(240) if not (60 <= m < 120)]
    evaluator, datamap, signal_lf = _setup(minutes)
    df = _eval(evaluator, datamap, signal_lf, price_shift=30)
    first = df.row(0, named=True)
    # bar 00:00's last row is 00:59; +30 ROWS lands at 02:29 (rows jump the
    # hole), not 01:29 wall-clock
    assert first["exec_time"] == START + timedelta(hours=2, minutes=29)


def test_shift_zero_keeps_pnl_columns_unchanged():
    evaluator, datamap, signal_lf = _setup(list(range(180)))
    df = _eval(evaluator, datamap, signal_lf, price_shift=0)
    for col in ("start_time", "price", "signal", "prev_signal", "trade", "pnl", "equity"):
        assert col in df.columns

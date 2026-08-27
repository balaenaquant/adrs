"""Evaluator checks against a real 1m candle history.

Opt-in: needs a parquet of 1m candles with `start_time` and `close`. Point
`ADRS_TEST_CANDLES_PARQUET` at one, or drop it at `xiang/test/eth_spot_binance_1m.parquet`.
Skipped entirely when absent, so CI without the data stays green.

These mirror the manual notebook check: rebuild the price column from the raw
candles by a route that never touches `group_by_dynamic`, then assert the
evaluator agrees.
"""

import math
import os
import polars as pl
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adrs.performance import Evaluator
from adrs.performance.evaluator import grid_phase, phase_drift
from adrs.performance.metric import Ratio
from adrs.data.datamap import Datamap
from adrs.data.types import DataInfo, DataColumn
from adrs.types import Topic

TOPIC = "binance-spot|candle?symbol=ETHUSDT&interval=1m"
INFO = DataInfo(
    topic=TOPIC,
    columns=[DataColumn(src="close", dst="price")],
    lookback_size=0,
)

START = datetime(2025, 5, 10, tzinfo=timezone.utc)
END = datetime(2026, 3, 9, tzinfo=timezone.utc)
FEES = 0.05

# (interval, phase) — the 7min/1min pair is the grid the manual check ran on.
GRIDS = [
    (timedelta(minutes=7), timedelta(minutes=1)),
    (timedelta(minutes=15), timedelta(0)),
    (timedelta(minutes=15), timedelta(minutes=4)),
]


def _candles_path() -> Path | None:
    if (env := os.environ.get("ADRS_TEST_CANDLES_PARQUET")) and Path(env).is_file():
        return Path(env)
    fallback = Path(__file__).parent.parent / "xiang/test/eth_spot_binance_1m.parquet"
    return fallback if fallback.is_file() else None


CANDLES = _candles_path()

pytestmark = pytest.mark.skipif(
    CANDLES is None,
    reason="no 1m candle parquet; set ADRS_TEST_CANDLES_PARQUET to enable",
)


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    assert CANDLES is not None
    return (
        pl.scan_parquet(CANDLES)
        .select("start_time", pl.col("close").alias("price"))
        .collect()
    )


@pytest.fixture(scope="module")
def datamap(prices: pl.DataFrame) -> Datamap:
    class _Real(Datamap):
        def __init__(self):
            super().__init__([INFO])
            self.topics = {Topic.from_str(TOPIC)}

        def keys(self):
            return [Topic.from_str(TOPIC)]

        def get(self, info):
            return prices

    return _Real()


def _signal(interval: timedelta, phase: timedelta, dtype: pl.DataType) -> pl.DataFrame:
    """Signal on a grid anchored so its epoch phase is exactly `phase`."""
    step, off = interval.total_seconds(), phase.total_seconds()
    first = datetime.fromtimestamp(
        int((START.timestamp() - off) // step) * step + off, tz=timezone.utc
    )
    while first < START:
        first += interval

    ts, t = [], first
    while t < END:
        ts.append(t)
        t += interval

    return pl.DataFrame(
        {
            "start_time": ts,
            "signal": [float((j // 11) % 3 - 1) for j in range(len(ts))],
        }
    ).with_columns(pl.col("start_time").cast(dtype))


def _run(
    datamap: Datamap, prices: pl.DataFrame, interval: timedelta, phase: timedelta
) -> tuple[Evaluator, pl.DataFrame, pl.DataFrame]:
    signal = _signal(interval, phase, prices.schema["start_time"])
    evaluator = Evaluator(assets={"ETH": INFO})
    out = evaluator.eval(
        signal_lf=signal.lazy(),
        base_asset="ETH",
        datamap=datamap,
        start_time=START,
        end_time=END,
        fees=FEES,
        interval=interval,
    ).collect()
    return evaluator, signal, out


@pytest.mark.parametrize("interval,phase", GRIDS)
def test_phase_and_drift_on_real_candles(datamap, prices, interval, phase):
    evaluator, signal, out = _run(datamap, prices, interval, phase)

    assert grid_phase(signal["start_time"].min(), interval) == phase
    assert evaluator.last_grid_phase == phase
    assert phase_drift(out["start_time"], interval) == []
    assert out["start_time"].diff().drop_nulls().unique().to_list() == [interval]
    assert out.filter(pl.col("signal") != 0).height > 0


@pytest.mark.parametrize("interval,phase", GRIDS)
def test_price_matches_arithmetic_bucketing(datamap, prices, interval, phase):
    """price[t] = last close in [t, t+interval), bucketed by plain integer maths."""
    _, _, out = _run(datamap, prices, interval, phase)

    iv_us = interval // timedelta(microseconds=1)
    ph_us = phase // timedelta(microseconds=1)
    expected = (
        prices.lazy()
        .with_columns(
            (
                ((pl.col("start_time").dt.epoch("us") - ph_us) // iv_us) * iv_us + ph_us
            ).alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.col("price").last())
        .with_columns(
            pl.from_epoch("bucket", time_unit="us")
            .dt.replace_time_zone("UTC")
            .cast(prices.schema["start_time"])
            .alias("start_time")
        )
        .select("start_time", pl.col("price").alias("expected"))
        .collect()
    )

    joined = out.select("start_time", "price").join(
        expected, on="start_time", how="left"
    )
    assert joined["expected"].null_count() == 0, "evaluator produced an off-grid label"
    assert (
        joined.select((pl.col("price") - pl.col("expected")).abs().max()).item() == 0.0
    )


@pytest.mark.parametrize("interval,phase", GRIDS)
def test_price_matches_shift_oracle(datamap, prices, interval, phase):
    """The manual notebook check: close.shift(1) on raw candles, then shift(-1) on the grid."""
    _, _, out = _run(datamap, prices, interval, phase)

    shifted = prices.select("start_time", pl.col("price").shift(1).alias("close_prev"))
    compared = (
        out.select("start_time", "price")
        .join(shifted, on="start_time", how="left")
        .with_columns(pl.col("close_prev").shift(-1).alias("expected"))
        .drop_nulls(["price", "expected"])
    )

    # Only the final row is uncomparable, since shift(-1) has nothing to pull from.
    assert compared.height >= out.height - 1
    assert (
        compared.select((pl.col("price") - pl.col("expected")).abs().max()).item()
        == 0.0
    )


@pytest.mark.parametrize("interval,phase", GRIDS)
def test_pnl_and_equity_match_plain_python(datamap, prices, interval, phase):
    _, signal, out = _run(datamap, prices, interval, phase)

    sig_at = dict(zip(signal["start_time"].to_list(), signal["signal"].to_list()))
    px = out["price"].to_list()

    held, signals = 0.0, []
    for t in out["start_time"].to_list():
        held = sig_at.get(t, held)
        signals.append(held)

    prev = [0.0] + signals[:-1]
    returns = [0.0] + [(px[i] - px[i - 1]) / px[i - 1] for i in range(1, len(px))]
    trade = [signals[i] - prev[i] for i in range(len(signals))]
    pnl = [
        prev[i] * returns[i] - abs(trade[i]) * FEES / 100 for i in range(len(signals))
    ]

    equity, run = [], 0.0
    for x in pnl:
        run += x
        equity.append(run)

    for col, want in [
        ("signal", signals),
        ("returns", returns),
        ("trade", trade),
        ("pnl", pnl),
        ("equity", equity),
    ]:
        assert out[col].to_list() == pytest.approx(want), f"{col} mismatch"


@pytest.mark.parametrize("interval,phase", GRIDS)
def test_sharpe_matches_manual_formula(datamap, prices, interval, phase):
    """The notebook's `365*24*60/interval_min` annualisation, now derived not hardcoded."""
    _, _, out = _run(datamap, prices, interval, phase)
    pnl = out["pnl"].to_numpy()

    dpy = 365 * 24 * 60 * 60 / interval.total_seconds()
    got = Ratio(interval=interval).compute(out)

    assert got["datapoints_per_year"] == pytest.approx(dpy)
    assert got["sharpe_ratio"] == pytest.approx(
        pnl.mean() / pnl.std(ddof=1) * math.sqrt(dpy)
    )


def test_mismatched_time_unit_raises(datamap, prices):
    """Parquet gives ms; a signal built from Python datetimes is us. Must not pass silently."""
    interval = timedelta(minutes=7)
    signal = _signal(interval, timedelta(minutes=1), pl.Datetime("us", "UTC"))
    assert prices.schema["start_time"] != signal.schema["start_time"]

    with pytest.raises(pl.exceptions.SchemaError, match="join keys"):
        Evaluator(assets={"ETH": INFO}).eval(
            signal_lf=signal.lazy(),
            base_asset="ETH",
            datamap=datamap,
            start_time=START,
            end_time=END,
            fees=FEES,
            interval=interval,
        ).collect()

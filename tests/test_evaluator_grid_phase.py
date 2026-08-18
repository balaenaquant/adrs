import logging
import math
import polars as pl
import pytest
from datetime import datetime, timedelta, timezone

from adrs.performance import Evaluator
from adrs.performance.evaluator import grid_phase, phase_drift
from adrs.data.datamap import Datamap
from adrs.data.types import DataInfo, DataColumn
from adrs.types import Topic

TOPIC = "binance|candle?symbol=BTCUSDT&interval=1m"
INFO = DataInfo(
    topic=TOPIC,
    columns=[DataColumn(src="close", dst="price")],
    lookback_size=1,
)

START = datetime(2024, 1, 3, tzinfo=timezone.utc)  # a Wednesday
DAYS = 30
END = START + timedelta(days=DAYS)
FEES = 0.02


def _prices() -> pl.DataFrame:
    """1m candles on the epoch-aligned grid, as every exchange feed delivers them."""
    n = DAYS * 24 * 60
    return pl.DataFrame(
        {
            "start_time": [START + timedelta(minutes=i) for i in range(n)],
            "price": [
                100.0 + 10 * math.sin(i / 500) + (i % 37) * 0.01 for i in range(n)
            ],
        }
    )


PRICES = _prices()


class _FakeDatamap(Datamap):
    def __init__(self):
        super().__init__([INFO])
        self.topics = {Topic.from_str(TOPIC)}

    def keys(self):
        return [Topic.from_str(TOPIC)]

    def get(self, info):
        return PRICES


def _signal(step: timedelta, phase: timedelta = timedelta(0)) -> pl.DataFrame:
    """Alternating signal on a grid `step` apart, shifted off the epoch grid by `phase`."""
    ts, k = [], 0
    while START + phase + k * step < END:
        ts.append(START + phase + k * step)
        k += 1
    return pl.DataFrame(
        {
            "start_time": ts,
            "signal": [1.0 if (j // 5) % 2 == 0 else -1.0 for j in range(len(ts))],
        }
    )


def _eval(signal_lf: pl.LazyFrame, interval: str | timedelta) -> pl.DataFrame:
    return (
        Evaluator(assets={"BTC": INFO})
        .eval(
            signal_lf=signal_lf,
            base_asset="BTC",
            datamap=_FakeDatamap(),
            start_time=START,
            end_time=END,
            fees=FEES,
            interval=interval,
        )
        .collect()
    )


def _pre_phase_pipeline(
    signal_lf: pl.LazyFrame, interval: str | timedelta
) -> pl.DataFrame:
    """Verbatim replica of the evaluator before phase derivation: epoch-anchored grid.

    Kept as the backward-compatibility oracle — anything already on the epoch grid
    must still produce a frame identical to this.
    """
    return (
        PRICES.lazy()
        .group_by_dynamic(index_column="start_time", every=interval)
        .agg(pl.col("price").last())
        .drop_nulls()
        .join(signal_lf, how="left", on="start_time")
        .filter(pl.col("start_time").is_between(START, END, closed="left"))
        .with_columns(
            pl.col("signal").forward_fill().fill_null(strategy="zero"),
            pl.col("signal")
            .shift(1)
            .alias("prev_signal")
            .forward_fill()
            .fill_null(strategy="zero"),
            pl.col("price").pct_change().alias("returns").fill_null(strategy="zero"),
        )
        .with_columns(
            (pl.col("signal") - pl.col("prev_signal"))
            .alias("trade")
            .fill_null(strategy="zero")
        )
        .with_columns(
            (
                pl.col("prev_signal") * pl.col("returns")
                - pl.col("trade").abs() * FEES / 100
            )
            .alias("pnl")
            .fill_null(strategy="zero")
        )
        .with_columns(
            pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero")
        )
        .collect()
    )


def _matched(df: pl.DataFrame) -> int:
    return df.filter(pl.col("signal") != 0).height


# --- regression: epoch-aligned signals must be untouched ------------------------


@pytest.mark.parametrize(
    "interval",
    [
        timedelta(minutes=5),
        timedelta(minutes=15),
        timedelta(hours=1),
        timedelta(hours=4),
        timedelta(days=1),
    ],
)
def test_aligned_grid_is_bit_identical_to_pre_phase_evaluator(interval):
    signal = _signal(interval)
    assert _pre_phase_pipeline(signal.lazy(), interval).equals(
        _eval(signal.lazy(), interval)
    )


@pytest.mark.parametrize("interval", ["1h", "1d", "1w", "1mo"])
def test_calendar_string_intervals_are_untouched(interval):
    # No fixed length to take a modulo of, so the phase must be skipped entirely.
    signal = _signal(timedelta(hours=1))
    assert _pre_phase_pipeline(signal.lazy(), interval).equals(
        _eval(signal.lazy(), interval)
    )


def test_aligned_signal_derives_zero_phase():
    for interval in (timedelta(minutes=15), timedelta(hours=1), timedelta(days=1)):
        assert grid_phase(START, interval) == timedelta(0)


# --- new behaviour: signals off the epoch grid ---------------------------------


@pytest.mark.parametrize(
    "interval,phase",
    [
        (timedelta(minutes=15), timedelta(minutes=4)),  # :04 / :19 / :34 / :49
        (timedelta(minutes=15), timedelta(minutes=7)),
        (timedelta(minutes=15), timedelta(minutes=4, seconds=30)),
        (timedelta(hours=1), timedelta(minutes=23)),
        (timedelta(hours=4), timedelta(minutes=11)),
        (timedelta(days=1), timedelta(hours=9, minutes=30)),
    ],
)
def test_offset_grid_matches_every_signal(interval, phase):
    signal = _signal(interval, phase)
    out = _eval(signal.lazy(), interval)

    assert grid_phase(START + phase, interval) == phase
    assert out.height > 0
    assert _matched(out) == out.height
    assert out["trade"].abs().sum() > 0


@pytest.mark.parametrize("phase", [timedelta(minutes=4), timedelta(minutes=7)])
def test_offset_grid_would_silently_zero_without_phase(phase):
    """Guards the bug this exists to fix: the old pipeline drops every signal, silently."""
    interval = timedelta(minutes=15)
    signal = _signal(interval, phase)

    stale = _pre_phase_pipeline(signal.lazy(), interval)
    assert stale.height > 0
    assert _matched(stale) == 0  # no exception, no warning — just a dead backtest
    assert stale["equity"][-1] == 0.0

    assert _matched(_eval(signal.lazy(), interval)) == stale.height


def test_offset_and_aligned_grids_agree_economically():
    # Same strategy, same prices, grid shifted by 4m: pnl should be close, not equal.
    interval = timedelta(minutes=15)
    aligned = _eval(_signal(interval).lazy(), interval)
    offset = _eval(_signal(interval, timedelta(minutes=4)).lazy(), interval)

    assert aligned.height == offset.height
    assert aligned["trade"].abs().sum() == offset["trade"].abs().sum()
    assert offset["equity"][-1] == pytest.approx(aligned["equity"][-1], rel=0.05)


# --- grid_phase itself ---------------------------------------------------------


@pytest.mark.parametrize(
    "first_start_time,interval,expected",
    [
        (
            datetime(2024, 1, 3, 9, 4, tzinfo=timezone.utc),
            timedelta(minutes=15),
            timedelta(minutes=4),
        ),
        (
            datetime(2024, 1, 3, 9, 4, 30, tzinfo=timezone.utc),
            timedelta(minutes=15),
            timedelta(minutes=4, seconds=30),
        ),
        (
            datetime(2024, 1, 3, 9, 4),
            timedelta(minutes=15),
            timedelta(minutes=4),
        ),  # naive -> UTC
        (
            datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc),
            timedelta(minutes=15),
            timedelta(0),
        ),
        (
            datetime(2024, 1, 3, 9, 4, tzinfo=timezone.utc),
            "15m",
            timedelta(0),
        ),  # str has no length
        (
            datetime(2024, 1, 3, 9, 4, tzinfo=timezone.utc),
            timedelta(0),
            timedelta(0),
        ),  # duplicate timestamps
        (None, timedelta(minutes=15), timedelta(0)),  # empty signal frame
    ],
)
def test_grid_phase(first_start_time, interval, expected):
    assert grid_phase(first_start_time, interval) == expected


def test_naive_and_aware_timestamps_agree():
    aware = datetime(2024, 1, 3, 9, 4, tzinfo=timezone.utc)
    assert grid_phase(aware, timedelta(minutes=15)) == grid_phase(
        aware.replace(tzinfo=None), timedelta(minutes=15)
    )


# --- deriving inside eval(): the frame it reads may be empty or genuinely lazy --


def test_empty_signal_frame_does_not_raise():
    empty = pl.DataFrame(
        schema={"start_time": PRICES.schema["start_time"], "signal": pl.Float64}
    )
    out = _eval(empty.lazy(), timedelta(hours=1))

    assert out.height > 0
    assert _matched(out) == 0


def test_lazy_signal_pipeline_resolves_phase():
    # Not just .lazy() on a materialised frame — eval() must resolve the phase
    # through an unexecuted plan.
    interval = timedelta(minutes=15)
    lazy_signal = (
        _signal(interval, timedelta(minutes=4))
        .lazy()
        .with_columns(pl.col("signal") * 0.5)
        .filter(pl.col("start_time") >= START)
    )
    out = _eval(lazy_signal, interval)

    assert _matched(out) == out.height
    assert out["signal"].abs().max() == 0.5


# --- phase drift mid-history must be a hard stop, not a silent partial join -----


def _drifting_signal(
    step: timedelta, phase: timedelta, drift: timedelta
) -> pl.DataFrame:
    """Signal that starts on `phase` then shifts to `phase + drift` half way through."""
    ts, k = [], 0
    while START + phase + k * step < END:
        at = START + phase + k * step
        ts.append(at if at < START + timedelta(days=DAYS // 2) else at + drift)
        k += 1
    return pl.DataFrame(
        {
            "start_time": ts,
            "signal": [1.0 if (j // 5) % 2 == 0 else -1.0 for j in range(len(ts))],
        }
    )


@pytest.mark.parametrize(
    "step,phase,drift",
    [
        (timedelta(minutes=15), timedelta(minutes=4), timedelta(minutes=1)),
        (timedelta(minutes=15), timedelta(0), timedelta(minutes=4)),
        (timedelta(hours=1), timedelta(minutes=23), timedelta(minutes=-2)),
    ],
)
def test_phase_drift_raises(step, phase, drift):
    signal = _drifting_signal(step, phase, drift)
    with pytest.raises(ValueError, match="drifts mid-history"):
        _eval(signal.lazy(), step)


def test_phase_drift_logs_error(caplog):
    signal = _drifting_signal(
        timedelta(minutes=15), timedelta(minutes=4), timedelta(minutes=1)
    )
    with caplog.at_level(logging.ERROR, logger="adrs.performance.evaluator"):
        with pytest.raises(ValueError):
            _eval(signal.lazy(), timedelta(minutes=15))

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    message = caplog.records[0].getMessage()
    assert "BTC" in message
    assert "0:04:00" in message and "0:05:00" in message


def test_phase_drift_reports_every_distinct_phase():
    signal = _drifting_signal(
        timedelta(minutes=15), timedelta(minutes=4), timedelta(minutes=1)
    )
    drift = phase_drift(signal["start_time"], timedelta(minutes=15))
    assert drift == [timedelta(minutes=4), timedelta(minutes=5)]


def test_constant_phase_reports_no_drift():
    for phase in (timedelta(0), timedelta(minutes=4), timedelta(minutes=7)):
        signal = _signal(timedelta(minutes=15), phase)
        assert phase_drift(signal["start_time"], timedelta(minutes=15)) == []


def test_gaps_are_not_drift():
    # A missing bar keeps the phase intact, so it must stay allowed.
    signal = _signal(timedelta(minutes=15), timedelta(minutes=4)).filter(
        pl.col("start_time").dt.hour() != 3
    )
    assert phase_drift(signal["start_time"], timedelta(minutes=15)) == []
    assert _matched(_eval(signal.lazy(), timedelta(minutes=15))) > 0


@pytest.mark.parametrize("interval", ["15m", "1h", "1w"])
def test_drift_check_skipped_for_calendar_intervals(interval):
    # No fixed length, so there is no phase to compare against.
    signal = _drifting_signal(
        timedelta(hours=1), timedelta(minutes=23), timedelta(minutes=1)
    )
    assert phase_drift(signal["start_time"], interval) == []


def test_drift_check_skipped_for_degenerate_inputs():
    signal = _signal(timedelta(minutes=15), timedelta(minutes=4))
    assert phase_drift(signal["start_time"], timedelta(0)) == []
    assert phase_drift(signal["start_time"].head(1), timedelta(minutes=15)) == []
    assert phase_drift(signal["start_time"].head(0), timedelta(minutes=15)) == []

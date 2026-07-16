"""Lookahead-bias regression tests for the backtest engines.

Timing convention under test:
- Evaluator: a signal row labeled T is decided at the close of bar
  [T, T+interval); positions are lagged one bar so it earns from T+interval.
- generate_signal_df: relabels bar-open labels to decision times (bar close).
- Portfolio.backtest: signal_df timestamps are decision times; a signal at T
  earns only from T (+ execution delay) onward.
- `price_shift` / `shift_backtest_candle_minute`: execution delay in minutes
  after the decision, identical meaning in both engines.
"""

import logging
import polars as pl
import pytest
from datetime import datetime, timedelta, timezone

from adrs.types import Topic, SortedDataList
from adrs.data import DataInfo, DataColumn, Datamap
from adrs.performance.evaluator import Evaluator
from adrs.portfolio import Portfolio
from adrs.utils import infer_interval

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
PRICE_TOPIC = "bybit-linear|candle?symbol=TESTUSDT&interval=1m"


def minute(i: int) -> datetime:
    return T0 + timedelta(minutes=i)


def make_price_datamap(closes: list[float]) -> tuple[Evaluator, Datamap, DataInfo]:
    info = DataInfo(
        topic=PRICE_TOPIC,
        columns=[DataColumn(src="close", dst="price")],
        lookback_size=0,
    )
    datamap = Datamap(data_infos=[info])
    datamap.map[Topic.from_str(PRICE_TOPIC)] = SortedDataList(
        [{"start_time": minute(i), "close": c} for i, c in enumerate(closes)]
    )
    evaluator = Evaluator(assets={"TEST": info})
    return evaluator, datamap, info


def eval_signals(
    closes: list[float],
    signals: dict[int, int],
    price_shift: int = 0,
    n_minutes: int | None = None,
) -> pl.DataFrame:
    """Run Evaluator.eval over 1m bars with signals at given minute labels."""
    evaluator, datamap, _ = make_price_datamap(closes)
    n = n_minutes if n_minutes is not None else len(closes)
    signal_lf = pl.LazyFrame(
        {
            "start_time": [minute(i) for i in signals.keys()],
            "signal": list(signals.values()),
        }
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
    )
    return evaluator.eval(
        signal_lf=signal_lf,
        base_asset="TEST",
        datamap=datamap,
        start_time=T0,
        end_time=minute(n),
        fees=0.0,
        interval=timedelta(minutes=1),
        price_shift=price_shift,
    ).collect()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# closes: 100 for bars 0-4, 200 from bar 5 on — the jump happens inside bar 5
JUMP_CLOSES = [100.0] * 5 + [200.0] * 5


def test_evaluator_signal_decided_after_jump_earns_nothing():
    # signal labeled bar 5 is decided at the close of bar 5 — AFTER the jump.
    # A lookahead-biased engine would credit it with the 100→200 move.
    df = eval_signals(JUMP_CLOSES, signals={5: 1})
    assert df["pnl"].sum() == pytest.approx(0.0)


def test_evaluator_signal_decided_before_jump_captures_it():
    # signal labeled bar 4 is decided at the close of bar 4 (price still 100),
    # so it legitimately rides the jump that happens during bar 5
    df = eval_signals(JUMP_CLOSES, signals={4: 1})
    assert df["pnl"].sum() == pytest.approx(1.0)
    # and the profit lands exactly on bar 5
    assert df.filter(pl.col("start_time") == minute(5))["pnl"][0] == pytest.approx(1.0)


def test_evaluator_price_column_is_honest():
    # `price` must be the true close of each bar regardless of delay
    for shift in (0, 2):
        df = eval_signals(JUMP_CLOSES, signals={4: 1}, price_shift=shift)
        assert df.filter(pl.col("start_time") == minute(4))["price"][0] == 100.0
        assert df.filter(pl.col("start_time") == minute(5))["price"][0] == 200.0


def test_evaluator_execution_delay_moves_the_fill():
    # with a 2-minute delay the bar-4 signal fills at the price observed at
    # (close of bar 4) + 2m = minute 7 — already 200, so the jump profit is gone
    df = eval_signals(JUMP_CLOSES, signals={4: 1}, price_shift=2)
    assert df["pnl"].sum() == pytest.approx(0.0)


def test_evaluator_rows_needing_future_fills_are_dropped():
    # a 2-minute delay makes the last 2 bars' fills unobservable — dropped,
    # never filled with stale prices
    df = eval_signals(JUMP_CLOSES, signals={4: 1}, price_shift=2)
    assert df["start_time"].max() == minute(7)


def test_evaluator_warns_on_misaligned_signals(caplog):
    # signals offset by 30s never match a bar label: they must be reported,
    # not silently produce a flat backtest
    evaluator, datamap, _ = make_price_datamap(JUMP_CLOSES)
    signal_lf = pl.LazyFrame(
        {
            "start_time": [minute(4) + timedelta(seconds=30)],
            "signal": [1],
        }
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
    )
    with caplog.at_level(logging.WARNING):
        df = evaluator.eval(
            signal_lf=signal_lf,
            base_asset="TEST",
            datamap=datamap,
            start_time=T0,
            end_time=minute(10),
            fees=0.0,
            interval=timedelta(minutes=1),
        ).collect()
    assert "do not align" in caplog.text
    assert df["pnl"].sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# infer_interval
# ---------------------------------------------------------------------------


def test_infer_interval_ignores_trailing_gap():
    times = pl.Series([minute(i) for i in [0, 1, 2, 3, 4, 10]])
    assert infer_interval(times) == timedelta(minutes=1)


def test_infer_interval_needs_two_timestamps():
    with pytest.raises(ValueError):
        infer_interval(pl.Series([minute(0)]))


# ---------------------------------------------------------------------------
# Portfolio.backtest
# ---------------------------------------------------------------------------


def make_portfolio(
    signal_rows: dict[str, list],
    metadata_rows: list[dict],
    weights: dict[str, float],
) -> Portfolio:
    return Portfolio(
        id="test",
        signal_df=pl.DataFrame(signal_rows).with_columns(
            pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
        ),
        metadata_df=pl.DataFrame(metadata_rows),
        weight_df=pl.DataFrame(
            {"custom_id": list(weights.keys()), "weights": list(weights.values())}
        ),
    )


def prices_frame(cols: dict[str, list[float]], n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"start_time": [minute(i) for i in range(n)], **cols}
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
    )


META = {
    "custom_id": "a1",
    "base_asset": "BTC",
    "shift_backtest_candle_minute": 0,
    "fees": 0.0,
}


def test_portfolio_slow_alpha_no_lookahead():
    # A 1h-cadence alpha on a 1m price grid — the historical bug: its signal
    # used to earn from (bar open + 1 minute), leaking up to one bar of future.
    # Price jumps at minute 90 (inside the alpha's 1:00-2:00 bar); the signal
    # decided at 2:00 (decision-time label) must earn NOTHING from that jump.
    n = 180
    closes = [100.0] * 90 + [200.0] * 90
    portfolio = make_portfolio(
        signal_rows={"start_time": [minute(120)], "a1": [1]},
        metadata_rows=[META],
        weights={"a1": 1.0},
    )
    _, df = portfolio.backtest(prices_frame({"BTC": closes}, n))
    assert df["pnl"].sum() == pytest.approx(0.0)


def test_portfolio_signal_earns_from_decision_time():
    # jump at minute 121, decision at minute 120 → legitimately captured
    n = 180
    closes = [100.0] * 121 + [200.0] * 59
    portfolio = make_portfolio(
        signal_rows={"start_time": [minute(120)], "a1": [1]},
        metadata_rows=[META],
        weights={"a1": 1.0},
    )
    _, df = portfolio.backtest(prices_frame({"BTC": closes}, n))
    assert df["pnl"].sum() == pytest.approx(1.0)


def test_portfolio_execution_delay_moves_the_fill():
    # same as above but 5 minutes of execution delay: fill happens at minute
    # 125, after the jump — profit gone
    n = 180
    closes = [100.0] * 121 + [200.0] * 59
    meta = {**META, "shift_backtest_candle_minute": 5}
    portfolio = make_portfolio(
        signal_rows={"start_time": [minute(120)], "a1": [1]},
        metadata_rows=[meta],
        weights={"a1": 1.0},
    )
    _, df = portfolio.backtest(prices_frame({"BTC": closes}, n))
    assert df["pnl"].sum() == pytest.approx(0.0)


def test_portfolio_pnl_not_double_counted_on_mismatched_grids():
    # ETH is missing minute 5; the merge across alphas must fill that hole
    # with 0 pnl for the ETH alpha, not repeat its last realized pnl
    n = 10
    btc = [100.0] * n
    eth = [100.0 * (1.02**i) for i in range(n)]
    eth_prices = prices_frame({"ETH": eth}, n).filter(pl.col("start_time") != minute(5))
    prices = prices_frame({"BTC": btc}, n).join(
        eth_prices, on="start_time", how="full", coalesce=True
    )

    portfolio = make_portfolio(
        signal_rows={"start_time": [minute(0)], "a1": [1], "a2": [1]},
        metadata_rows=[
            META,
            {**META, "custom_id": "a2", "base_asset": "ETH"},
        ],
        weights={"a1": 1.0, "a2": 1.0},
    )
    _, df = portfolio.backtest(prices)

    # expected a2 pnl: position 1 over ETH's own return series (with the gap
    # spanned by a single pct_change step), computed independently
    eth_series = eth_prices["ETH"]
    expected = (eth_series.pct_change().fill_null(0.0)).sum()
    assert df["a2_pnl"].sum() == pytest.approx(expected)
    # BTC leg is flat — total portfolio pnl must equal the ETH leg alone
    assert df["pnl"].sum() == pytest.approx(expected)

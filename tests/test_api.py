"""API-surface tests: decoupled evaluate_signals, execution_delay_minute
deprecation path, optional metrics, Portfolio frame validation and the
param-aware signal cache fingerprint."""

import logging
import polars as pl
import pytest
from datetime import timedelta

from adrs.alpha import Alpha
from adrs.data import DataProcessor
from adrs.portfolio import Portfolio
from adrs.execution.backtest import _alpha_fingerprint
from adrs.performance.evaluator import evaluate_signals

from test_lookahead import JUMP_CLOSES, T0, make_price_datamap, minute, prices_frame


def _signal_lf(signals: dict[int, int]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "start_time": [minute(i) for i in signals.keys()],
            "signal": list(signals.values()),
        }
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
    )


# ---------------------------------------------------------------------------
# evaluate_signals — plain-frame core, no Datamap/DataInfo/Topic required
# ---------------------------------------------------------------------------


def test_evaluate_signals_takes_plain_price_frame():
    prices = pl.DataFrame(
        {
            "start_time": [minute(i) for i in range(len(JUMP_CLOSES))],
            "price": JUMP_CLOSES,
        }
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
    )
    df = evaluate_signals(
        signal_lf=_signal_lf({4: 1}),
        prices=prices,
        raw_interval=timedelta(minutes=1),
        start_time=T0,
        end_time=minute(10),
        fees=0.0,
        interval=timedelta(minutes=1),
    ).collect()
    assert df["pnl"].sum() == pytest.approx(1.0)


def test_eval_price_shift_deprecated_but_equivalent():
    evaluator, datamap, _ = make_price_datamap(JUMP_CLOSES)

    def run(**kw) -> pl.DataFrame:
        return evaluator.eval(
            signal_lf=_signal_lf({4: 1}),
            base_asset="TEST",
            datamap=datamap,
            start_time=T0,
            end_time=minute(10),
            fees=0.0,
            interval=timedelta(minutes=1),
            **kw,
        ).collect()

    with pytest.warns(DeprecationWarning, match="price_shift is deprecated"):
        old = run(price_shift=2)
    new = run(execution_delay_minute=2)
    assert old.equals(new)


def test_eval_rejects_both_delay_kwargs():
    evaluator, datamap, _ = make_price_datamap(JUMP_CLOSES)
    with pytest.raises(ValueError, match="not both"):
        evaluator.eval(
            signal_lf=_signal_lf({4: 1}),
            base_asset="TEST",
            datamap=datamap,
            start_time=T0,
            end_time=minute(10),
            fees=0.0,
            interval=timedelta(minutes=1),
            price_shift=2,
            execution_delay_minute=2,
        )


# ---------------------------------------------------------------------------
# Alpha.backtest — compute_metrics=False
# ---------------------------------------------------------------------------


class PassthroughAlpha(Alpha):
    def __init__(self, threshold: float = 0.5):
        super().__init__(
            id="passthrough", data_infos=[], data_processor=DataProcessor()
        )
        self.threshold = threshold

    def next(self, data_df: pl.DataFrame) -> pl.DataFrame:
        return data_df.select(pl.col("start_time"), pl.col("signal"))


def test_backtest_compute_metrics_false_returns_none():
    evaluator, datamap, _ = make_price_datamap(JUMP_CLOSES)
    data_df = pl.DataFrame(
        {
            "start_time": [minute(i) for i in range(10)],
            "signal": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    ).with_columns(
        pl.col("start_time").dt.replace_time_zone("UTC").dt.cast_time_unit("ms")
    )
    performance, df = PassthroughAlpha().backtest(
        evaluator=evaluator,
        base_asset="TEST",
        datamap=datamap,
        start_time=T0,
        end_time=minute(10),
        fees=0.0,
        data_df=data_df,
        compute_metrics=False,
    )
    assert performance is None
    assert df["pnl"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Portfolio frame validation
# ---------------------------------------------------------------------------

VALID_METADATA = pl.DataFrame(
    {
        "custom_id": ["a1"],
        "base_asset": ["BTC"],
        "fees": [0.0],
        "shift_backtest_candle_minute": [0],
    }
)
VALID_WEIGHTS = pl.DataFrame({"custom_id": ["a1"], "weights": [1.0]})
VALID_SIGNALS = prices_frame({"a1": [1.0]}, 1).rename({"a1": "a1"})


def test_portfolio_rejects_missing_metadata_columns():
    with pytest.raises(ValueError, match="metadata_df is missing"):
        Portfolio(
            id="p",
            signal_df=VALID_SIGNALS,
            metadata_df=VALID_METADATA.drop("fees"),
            weight_df=VALID_WEIGHTS,
        )


def test_portfolio_rejects_missing_weights():
    with pytest.raises(ValueError, match="no weights for alphas"):
        Portfolio(
            id="p",
            signal_df=VALID_SIGNALS,
            metadata_df=VALID_METADATA,
            weight_df=pl.DataFrame({"custom_id": ["other"], "weights": [1.0]}),
        )


def test_portfolio_warns_on_missing_signal_column(caplog):
    with caplog.at_level(logging.WARNING):
        Portfolio(
            id="p",
            signal_df=VALID_SIGNALS.drop("a1").with_columns(
                pl.lit(1).alias("unrelated")
            ),
            metadata_df=VALID_METADATA,
            weight_df=VALID_WEIGHTS,
        )
    assert "no column for alphas" in caplog.text


# ---------------------------------------------------------------------------
# signal cache fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_changes_with_alpha_params():
    meta = VALID_METADATA.with_columns(pl.lit("passthrough").alias("custom_id"))
    fp_a = _alpha_fingerprint([PassthroughAlpha(threshold=0.5)], meta)
    fp_b = _alpha_fingerprint([PassthroughAlpha(threshold=0.9)], meta)
    fp_a2 = _alpha_fingerprint([PassthroughAlpha(threshold=0.5)], meta)
    assert fp_a != fp_b
    assert fp_a == fp_a2


def test_fingerprint_changes_with_metadata():
    meta = VALID_METADATA.with_columns(pl.lit("passthrough").alias("custom_id"))
    meta_delayed = meta.with_columns(pl.lit(10).alias("shift_backtest_candle_minute"))
    alphas = [PassthroughAlpha()]
    assert _alpha_fingerprint(alphas, meta) != _alpha_fingerprint(alphas, meta_delayed)

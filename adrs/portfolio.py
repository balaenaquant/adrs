import logging
import polars as pl

from typing import Protocol, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, ConfigDict

from adrs.utils import infer_interval
from adrs.performance.metric import Ratio, Drawdown, Trade

logger = logging.getLogger(__name__)

METADATA_REQUIRED_COLUMNS = (
    "custom_id",
    "base_asset",
    "fees",
    "shift_backtest_candle_minute",
)


class TradePerformance(BaseModel):
    largest_loss: float
    num_datapoints: int
    num_trades: int
    avg_holding_time_in_seconds: float
    long_trades: int
    short_trades: int
    win_trades: int
    lose_trades: int
    win_streak: int
    lose_streak: int
    win_rate: float

    model_config = ConfigDict(extra="allow")


class PortfolioPerformance(BaseModel):
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    cagr: float
    annualized_return: float
    total_return: float
    min_cumu: float
    start_time: datetime
    end_time: datetime
    max_drawdown: float
    max_drawdown_percentage: float
    max_drawdown_start_date: datetime
    max_drawdown_end_date: datetime
    max_drawdown_recover_date: datetime
    max_drawdown_max_duration_in_days: float
    trades: dict[str, TradePerformance]
    metadata: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class WeightAllocator(Protocol):
    def weights(self) -> pl.DataFrame: ...


class MeanWeightAllocator:
    def __init__(
        self,
        signal_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
    ) -> None:
        self.signal_df = signal_df
        self.metadata_df = metadata_df

    def weights(self) -> pl.DataFrame:
        n = len(self.metadata_df)
        return self.metadata_df.select("custom_id").with_columns(
            pl.lit(1.0 / n).alias("weights")
        )


class Portfolio:
    def __init__(
        self,
        id: str,
        signal_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
        weight_df: pl.DataFrame,
    ) -> None:
        self._validate_frames(signal_df, metadata_df, weight_df)
        self.id = id
        self.signal_df = signal_df
        self.metadata_df = metadata_df
        self.weight_df = weight_df
        self.lastest_signal: dict[str, int] = {}

    @staticmethod
    def _validate_frames(
        signal_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
        weight_df: pl.DataFrame,
    ) -> None:
        """Fail fast on malformed frames — the stringly-typed columns are
        otherwise only discovered deep inside backtest()/get_signal()."""
        missing = [c for c in METADATA_REQUIRED_COLUMNS if c not in metadata_df.columns]
        if missing:
            raise ValueError(f"metadata_df is missing required columns {missing}")
        missing = [c for c in ("custom_id", "weights") if c not in weight_df.columns]
        if missing:
            raise ValueError(f"weight_df is missing required columns {missing}")
        if "start_time" not in signal_df.columns:
            raise ValueError("signal_df is missing the 'start_time' column")

        ids = metadata_df["custom_id"].to_list()
        weight_ids = set(weight_df["custom_id"].to_list())
        unweighted = [i for i in ids if i not in weight_ids]
        if unweighted:
            raise ValueError(f"weight_df has no weights for alphas {unweighted}")
        # a missing signal column is not fatal live (update_signal can feed
        # it), but a backtest would raise — surface it early
        unsignaled = [i for i in ids if i not in signal_df.columns]
        if unsignaled:
            logger.warning(
                "signal_df has no column for alphas %s — backtests will fail "
                "unless signals are provided via update_signal()",
                unsignaled,
            )

    def update_signal(self, id: str, signal: int):
        self.lastest_signal[id] = signal

    def update_weights(self, weight_df: pl.DataFrame):
        self.weight_df = weight_df

    def get_signal(self) -> pl.DataFrame:
        rows = []
        for custom_id in self.metadata_df["custom_id"].to_list():
            if custom_id in self.lastest_signal:
                signal = self.lastest_signal[custom_id]
            else:
                col = self.signal_df[custom_id]
                if col.is_empty():
                    raise Exception(
                        f"Alpha ID {custom_id} was not initialized in portfolio"
                    )
                signal = col[-1]

            rows.append({"custom_id": custom_id, "signal": signal})

        signal = pl.DataFrame(rows)
        return (
            self.metadata_df.join(signal, on="custom_id", how="left")
            .join(self.weight_df, on="custom_id", how="left")
            .with_columns(
                (pl.col("signal") * pl.col("weights")).alias("weighted_signal")
            )
            .group_by("base_asset")
            .agg(pl.col("weighted_signal").sum())
        )

    def backtest(
        self, prices_df: pl.DataFrame
    ) -> tuple[PortfolioPerformance, pl.DataFrame]:
        """Backtest the portfolio against a wide price frame (one column per
        base asset, raw candle cadence — e.g. 1m closes).

        Timing convention (no lookahead):
        - `signal_df` timestamps are DECISION times: a signal row labeled T was
          decided at T and is actionable from T onward (this is what
          `generate_signal_df` produces — alpha bar labels shifted forward by
          one alpha bar to their close).
        - `shift_backtest_candle_minute` is the execution delay in MINUTES
          after the decision: fills happen at the raw price observed at
          (decision time + delay). It is a pure delay; the candle-close
          alignment is already encoded in the signal timestamps.
        - `fees` (metadata column) is in PERCENT of notional per unit of
          turnover: pnl subtracts |Δsignal| * fees / 100, so fees=0.035
          charges 3.5 bps per full position flip leg.
        """
        # check if prices include all base asset
        base_assets = self.metadata_df["base_asset"].unique().to_list()
        price_cols = [c for c in prices_df.columns if c != "start_time"]
        missing_cols = [a for a in base_assets if a not in price_cols]
        if missing_cols:
            raise ValueError(
                f"Price DF did not include the following asset prices {missing_cols}"
            )

        prices_df = prices_df.sort("start_time")
        # a raw candle labeled t covers [t, t+raw_interval); its close is only
        # observable at t+raw_interval
        raw_interval = infer_interval(prices_df["start_time"])

        merged_pnl: pl.DataFrame | None = None
        asset_merged: dict[str, pl.DataFrame | None] = {}
        for alpha in self.metadata_df.iter_rows(named=True):
            custom_id: str = alpha["custom_id"]
            fees: float = alpha["fees"]
            delay = timedelta(minutes=alpha["shift_backtest_candle_minute"])
            base_asset: str = alpha["base_asset"]
            alpha_weight: float = float(
                self.weight_df.filter(pl.col("custom_id") == custom_id)["weights"][0]
            )

            alpha_prices = (
                prices_df.select(
                    pl.col("start_time"),
                    pl.col(base_asset).alias("price"),
                )
                .drop_nulls()
                .sort("start_time")
            )
            last_close = alpha_prices["start_time"].max() + raw_interval  # type: ignore[operator]

            # execution price: last raw close observable at exec_time
            exec_prices = alpha_prices.select(
                (pl.col("start_time") + raw_interval).alias("close_time"),
                pl.col("price").alias("exec_price"),
            )

            alpha_signals = self.signal_df.select(
                pl.col("start_time"),
                pl.col(custom_id).alias("signal"),
            )

            pnl_col = f"{custom_id}_pnl"
            sig_col = f"{custom_id}_sig"
            alpha_full_df = (
                alpha_prices.with_columns(
                    (pl.col("start_time") + raw_interval + delay).alias("exec_time")
                )
                .join_asof(
                    exec_prices,
                    left_on="exec_time",
                    right_on="close_time",
                    strategy="backward",
                )
                # a fill scheduled after the last observable raw close would
                # need future data — drop instead of using a stale price
                .filter(pl.col("exec_time") <= last_close)
                .join(alpha_signals, on="start_time", how="left")
                .with_columns(
                    pl.col("signal").forward_fill().fill_null(strategy="zero"),
                    # exec_price returns over [t + delay, t + raw_interval + delay]
                    pl.col("exec_price")
                    .pct_change()
                    .alias("returns")
                    .fill_null(strategy="zero"),
                )
                .with_columns(
                    pl.col("signal")
                    .shift(1)
                    .fill_null(strategy="zero")
                    .alias("prev_signal")
                )
                .with_columns(
                    (pl.col("signal") - pl.col("prev_signal"))
                    .alias("trade")
                    .fill_null(strategy="zero")
                )
                .with_columns(
                    # `signal` (not `prev_signal`): signal timestamps are
                    # decision times, and the returns window of row t only
                    # starts at t + delay — the position decided at t earns
                    # exactly from its own fill onward, never before.
                    (
                        pl.col("signal") * pl.col("returns")
                        - pl.col("trade").abs() * fees / 100
                    ).alias("pnl")
                )
            )

            # accumulate per-asset weighted signal + price for Trade stats
            alpha_asset_df = alpha_full_df.select(
                pl.col("start_time"),
                pl.col("price"),
                (pl.col("signal") * alpha_weight).alias(sig_col),
                (pl.col("pnl") * alpha_weight).alias(pnl_col),
            )
            if base_asset not in asset_merged or asset_merged[base_asset] is None:
                asset_merged[base_asset] = alpha_asset_df
            else:
                asset_merged[base_asset] = (
                    asset_merged[base_asset]
                    .join(
                        alpha_asset_df.drop("price"),
                        on="start_time",
                        how="full",
                        coalesce=True,
                    )
                    .sort("start_time")
                    .with_columns(
                        # signal is a state — carry it across timestamps this
                        # alpha's grid doesn't have; pnl is a flow — a missing
                        # timestamp means no pnl, never a repeat of the last one
                        pl.col(sig_col).forward_fill(),
                        pl.col(pnl_col).fill_null(0.0),
                    )
                )

            alpha_df = alpha_full_df.select(
                pl.col("start_time"),
                (pl.col("pnl") * alpha_weight).alias(pnl_col),
            )

            if merged_pnl is None:
                merged_pnl = alpha_df
            else:
                # pnl is a flow — fill gaps with 0, never forward-fill (that
                # would double-count the last realized pnl at timestamps this
                # alpha's grid doesn't have)
                merged_pnl = (
                    merged_pnl.join(
                        alpha_df, on="start_time", how="full", coalesce=True
                    )
                    .sort("start_time")
                    .with_columns(pl.col(pnl_col).fill_null(0.0))
                )

        trade_performances: dict[str, TradePerformance] = {}
        for base_asset, asset_df in asset_merged.items():
            assert asset_df is not None
            sig_cols = [c for c in asset_df.columns if c.endswith("_sig")]
            pnl_cols_asset = [c for c in asset_df.columns if c.endswith("_pnl")]
            asset_trade_df = (
                asset_df.select(
                    pl.col("start_time"),
                    pl.col("price"),
                    pl.sum_horizontal(sig_cols).alias("signal"),
                    pl.sum_horizontal(pnl_cols_asset).alias("pnl"),
                )
                .sort("start_time")
                .with_columns(
                    pl.col("signal")
                    .shift(1)
                    .alias("prev_signal")
                    .forward_fill()
                    .fill_null(strategy="zero"),
                    pl.col("signal").diff().alias("trade").fill_null(strategy="zero"),
                )
            )
            trade_result = Trade().compute(asset_trade_df)
            trade_performances[base_asset] = TradePerformance(**trade_result)

        assert merged_pnl is not None
        pnl_cols = [c for c in merged_pnl.columns if c.endswith("_pnl")]
        performance_df = (
            merged_pnl.select(
                pl.col("start_time"),
                pl.sum_horizontal(pnl_cols).alias("pnl"),
                *[pl.col(c).fill_null(0.0) for c in pnl_cols],
            )
            .sort("start_time")
            .with_columns(
                pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero"),
            )
        )

        start_time = performance_df["start_time"].min()
        end_time = performance_df["start_time"].max()
        assert isinstance(start_time, datetime) and isinstance(end_time, datetime)
        performance: dict[str, Any] = {
            "start_time": start_time,
            "end_time": end_time,
            "trades": trade_performances,
            "metadata": {},
        }
        for metric in [Ratio(), Drawdown()]:
            performance = {**performance, **metric.compute(performance_df)}

        return PortfolioPerformance.model_validate(performance), performance_df

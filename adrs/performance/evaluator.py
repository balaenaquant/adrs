import logging
import polars as pl
from datetime import datetime, timedelta

from adrs.types import Topic
from adrs.data.datamap import DataInfo, Datamap

logger = logging.getLogger(__name__)


def has_column(info: DataInfo, col: str) -> bool:
    for column in info.columns:
        if column.dst == col:
            return True
    return False


class Evaluator:
    def __init__(self, assets: dict[str, DataInfo]):
        for k, v in assets.items():
            if not has_column(info=v, col="price"):
                raise ValueError(f"Asset {k} must have a 'dst' column of 'price'")

        self.assets = assets

    def eval(
        self,
        signal_lf: pl.LazyFrame,
        base_asset: str,
        datamap: Datamap,
        start_time: datetime,
        end_time: datetime,
        fees: float,
        interval: str | timedelta,
        price_shift: int = 0,
        output_columns: list[pl.Expr] = [pl.all()],
    ):
        """Evaluate a signal frame against prices.

        Timing convention (no lookahead):
        - A signal row labeled T is decided at the close of bar [T, T+interval),
          i.e. it becomes actionable at real time T+interval.
        - Positions are lagged one bar (`prev_signal`), so the signal decided at
          the close of bar T only earns returns from bar T+interval onward.
        - `price_shift` is the execution delay in MINUTES after the bar close:
          fills happen at the raw price observed at (bar close + price_shift).
          It is a pure delay — the one-bar signal lag above already accounts for
          the candle only being known at its close. 0 means "fill at the close
          price the signal was computed from".

        Output columns:
        - `price`: the true close of each bar (honest timestamps, use for
          plots/trade stats).
        - `exec_price`: the fill price (raw price at bar close + delay); pnl is
          computed from `exec_price` returns.
        """
        if price_shift < 0:
            raise ValueError("price_shift must be non-negative")

        if base_asset not in self.assets:
            raise ValueError(f"Base asset {base_asset} not found in configured assets")

        info = self.assets[base_asset]
        if Topic.from_str(info.topic) not in datamap.keys():
            raise ValueError(f"Data for base asset {base_asset} not found in datamap")

        raw_interval = Topic.from_str(info.topic).interval()
        if raw_interval is None:
            raise ValueError(
                f"Price topic for base asset {base_asset} must declare an interval"
            )

        delay = timedelta(minutes=price_shift)
        prices_df = datamap.get(info).sort("start_time")
        # a raw candle labeled t covers [t, t+raw_interval); its close is only
        # observable at t+raw_interval
        last_close = prices_df["start_time"].max() + raw_interval  # type: ignore[operator]

        if isinstance(interval, str):
            bucket_close = pl.col("start_time").dt.offset_by(interval)
        else:
            bucket_close = pl.col("start_time") + interval

        # true close of each signal-cadence bar, labeled by bar start
        bars = (
            prices_df.lazy()
            .group_by_dynamic(index_column="start_time", every=interval)
            .agg(pl.col("price").last())
            .drop_nulls()
            .with_columns((bucket_close + delay).alias("exec_time"))
        )

        # execution price: last raw close observable at exec_time
        exec_prices = prices_df.lazy().select(
            (pl.col("start_time") + raw_interval).alias("close_time"),
            pl.col("price").alias("exec_price"),
        )

        # warn on signals that don't align with any price bar — they are
        # silently dropped by the left join below and would otherwise look like
        # a legitimate flat backtest
        unmatched = (
            signal_lf.filter(
                pl.col("start_time").is_between(start_time, end_time, closed="left")
            )
            .join(bars.select("start_time"), on="start_time", how="anti")
            .select(pl.len())
            .collect()
            .item()
        )
        if unmatched:
            logger.warning(
                "%d signal rows for base asset %s do not align with any price bar "
                "(every=%s) and will be dropped from the backtest",
                unmatched,
                base_asset,
                interval,
            )

        df = (
            bars.join_asof(
                exec_prices,
                left_on="exec_time",
                right_on="close_time",
                strategy="backward",
            )
            # a fill scheduled after the last observable raw close would need
            # future data — drop instead of filling with a stale price
            .filter(pl.col("exec_time") <= last_close)
            .join(signal_lf, how="left", on="start_time")
            .filter(
                pl.col("start_time").is_between(start_time, end_time, closed="left")
            )
            .with_columns(
                pl.col("signal").forward_fill().fill_null(strategy="zero"),
                pl.col("signal")
                .shift(1)
                .alias("prev_signal")
                .forward_fill()
                .fill_null(strategy="zero"),
                pl.col("exec_price")
                .pct_change()
                .alias("returns")
                .fill_null(strategy="zero"),
            )
            .with_columns(
                (pl.col("signal") - pl.col("prev_signal"))
                .alias("trade")
                .fill_null(strategy="zero")
            )
            .with_columns(
                (
                    pl.col("prev_signal") * pl.col("returns")
                    - pl.col("trade").abs() * fees / 100
                )
                .alias("pnl")
                .fill_null(strategy="zero"),
            )
            .with_columns(
                pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero")
            )
        )
        return df.select(*output_columns)

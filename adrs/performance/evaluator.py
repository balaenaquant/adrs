import logging
import warnings
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


def evaluate_signals(
    signal_lf: pl.LazyFrame | pl.DataFrame,
    prices: pl.DataFrame,
    raw_interval: timedelta,
    start_time: datetime,
    end_time: datetime,
    fees: float,
    interval: str | timedelta,
    execution_delay_minute: int = 0,
    output_columns: list[pl.Expr] = [pl.all()],
) -> pl.LazyFrame:
    """Evaluate a signal frame against a plain price frame.

    This is the decoupled core of the backtest engine — it has no knowledge of
    Datamap/DataInfo/Topic, so it can be fed any `prices` frame with columns
    `start_time` (tz-aware) and `price` (raw candle closes at `raw_interval`
    cadence, labeled by candle open).

    Timing convention (no lookahead):
    - A signal row labeled T is decided at the close of bar [T, T+interval),
      i.e. it becomes actionable at real time T+interval.
    - Positions are lagged one bar (`prev_signal`), so the signal decided at
      the close of bar T only earns returns from bar T+interval onward.
    - `execution_delay_minute` is the execution delay in MINUTES after the bar
      close: fills happen at the raw price observed at (bar close + delay).
      It is a pure delay — the one-bar signal lag above already accounts for
      the candle only being known at its close. 0 means "fill at the close
      price the signal was computed from". Fill-time resolution is limited by
      `raw_interval` (use 1m prices for minute-accurate delays).
    - `fees` is in PERCENT of notional per unit of turnover: pnl subtracts
      |Δsignal| * fees / 100, so fees=0.035 charges 3.5 bps per full position
      flip leg.

    Output columns:
    - `price`: the true close of each bar (honest timestamps, use for
      plots/trade stats).
    - `exec_price`: the fill price (raw price at bar close + delay); pnl is
      computed from `exec_price` returns.
    """
    if execution_delay_minute < 0:
        raise ValueError("execution_delay_minute must be non-negative")

    if isinstance(signal_lf, pl.DataFrame):
        signal_lf = signal_lf.lazy()

    delay = timedelta(minutes=execution_delay_minute)
    prices = prices.sort("start_time")
    # a raw candle labeled t covers [t, t+raw_interval); its close is only
    # observable at t+raw_interval
    last_close = prices["start_time"].max() + raw_interval  # type: ignore[operator]

    if isinstance(interval, str):
        bucket_close = pl.col("start_time").dt.offset_by(interval)
    else:
        bucket_close = pl.col("start_time") + interval

    # true close of each signal-cadence bar, labeled by bar start
    bars = (
        prices.lazy()
        .group_by_dynamic(index_column="start_time", every=interval)
        .agg(pl.col("price").last())
        .drop_nulls()
        .with_columns((bucket_close + delay).alias("exec_time"))
    )

    # execution price: last raw close observable at exec_time
    exec_prices = prices.lazy().select(
        (pl.col("start_time") + raw_interval).alias("close_time"),
        pl.col("price").alias("exec_price"),
    )

    # warn on signals that don't align with any price bar — they are silently
    # dropped by the left join below and would otherwise look like a
    # legitimate flat backtest
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
            "%d signal rows do not align with any price bar (every=%s) "
            "and will be dropped from the backtest",
            unmatched,
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
        .filter(pl.col("start_time").is_between(start_time, end_time, closed="left"))
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
        price_shift: int | None = None,
        output_columns: list[pl.Expr] = [pl.all()],
        execution_delay_minute: int | None = None,
    ):
        """Resolve prices for `base_asset` from the datamap and run
        `evaluate_signals` (see its docstring for the timing convention,
        fee units and output columns).

        `price_shift` is a deprecated alias of `execution_delay_minute`
        (execution delay in minutes after the bar close).
        """
        if price_shift is not None and execution_delay_minute is not None:
            raise ValueError(
                "pass either execution_delay_minute or the deprecated "
                "price_shift, not both"
            )
        if price_shift is not None:
            warnings.warn(
                "price_shift is deprecated, use execution_delay_minute "
                "(same meaning: execution delay in minutes after bar close)",
                DeprecationWarning,
                stacklevel=2,
            )
            execution_delay_minute = price_shift

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

        return evaluate_signals(
            signal_lf=signal_lf,
            prices=datamap.get(info),
            raw_interval=raw_interval,
            start_time=start_time,
            end_time=end_time,
            fees=fees,
            interval=interval,
            execution_delay_minute=execution_delay_minute or 0,
            output_columns=output_columns,
        )

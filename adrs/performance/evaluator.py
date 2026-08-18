import logging
import polars as pl
from datetime import datetime, timedelta, timezone

from adrs.types import Topic
from adrs.data.datamap import DataInfo, Datamap

logger = logging.getLogger(__name__)


def grid_phase(
    first_start_time: datetime | None, interval: str | timedelta
) -> timedelta:
    """
    Get offset of signal for group_by_dynamic(offset=...)
    """
    if first_start_time is None or not isinstance(interval, timedelta):
        return timedelta(0)
    interval_s = interval.total_seconds()
    if interval_s <= 0:
        return timedelta(0)
    if first_start_time.tzinfo is None:
        first_start_time = first_start_time.replace(tzinfo=timezone.utc)
    return timedelta(seconds=first_start_time.timestamp() % interval_s)


def phase_drift(start_times: pl.Series, interval: str | timedelta) -> list[timedelta]:
    """Distinct grid phases in `start_times`, empty when the phase is constant."""
    if not isinstance(interval, timedelta) or start_times.len() < 2:
        return []
    interval_us = interval // timedelta(microseconds=1)
    if interval_us <= 0:
        return []
    phases = start_times.dt.epoch("us") % interval_us
    if phases.n_unique() < 2:
        return []
    return [timedelta(microseconds=int(p)) for p in phases.unique().sort()]


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
        if price_shift < 0:
            raise ValueError("price_shift must be non-negative")

        if base_asset not in self.assets:
            raise ValueError(f"Base asset {base_asset} not found in configured assets")

        info = self.assets[base_asset]
        if Topic.from_str(info.topic) not in datamap.keys():
            raise ValueError(f"Data for base asset {base_asset} not found in datamap")

        prices_lf = (
            datamap.get(info).lazy().with_columns(pl.col("price").shift(-price_shift))
        )

        start_times = signal_lf.select(pl.col("start_time")).collect()["start_time"]
        grid_offset = grid_phase(start_times.min(), interval)

        if drift := phase_drift(start_times, interval):
            shown = ", ".join(str(p) for p in drift[:5])
            more = f" (+{len(drift) - 5} more)" if len(drift) > 5 else ""
            logger.error(
                "Signal grid phase for base asset %s is not constant: %d distinct "
                "phases [%s]%s against interval %s. Prices can only be resampled onto "
                "one phase (%s), so rows on the others would be dropped by the join "
                "and silently zeroed.",
                base_asset,
                len(drift),
                shown,
                more,
                interval,
                grid_offset,
            )
            raise ValueError(
                f"Signal grid phase for base asset {base_asset} drifts mid-history: "
                f"{len(drift)} distinct phases [{shown}]{more} for interval {interval}"
            )

        df = (
            prices_lf.group_by_dynamic(
                index_column="start_time", every=interval, offset=grid_offset
            )
            .agg(pl.col("price").last())
            .drop_nulls()
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
                pl.col("price")
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

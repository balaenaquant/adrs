import polars as pl
from datetime import datetime, timedelta

from adrs.types import Topic
from adrs.data.datamap import DataInfo, Datamap


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

        # exec_time is the actual timestamp of the (shifted) price each row is
        # evaluated against. price_shift moves prices by ROWS of the native
        # grid, so across data gaps exec_time can exceed start_time + shift
        # minutes — it is the ground truth for execution-time alignment
        # downstream (e.g. deshifting), where reconstructing it from the shift
        # value alone is approximate.
        prices_lf = (
            datamap.get(info)
            .lazy()
            .with_columns(
                pl.col("price").shift(-price_shift),
                pl.col("start_time").shift(-price_shift).alias("exec_time"),
            )
        )

        df = (
            prices_lf.group_by_dynamic(index_column="start_time", every=interval)
            .agg(pl.col("price").last(), pl.col("exec_time").last())
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

import polars as pl
from datetime import datetime, timedelta

from adrs.data.datamap import DataInfo, Datamap


def has_column(info: DataInfo, col: str) -> bool:
    for column in info.columns:
        if column.dst == col:
            return True
    return False


class Evaluator:
    def __init__(
        self,
        assets: dict[str, DataInfo],
    ):
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
        if info not in datamap.keys():
            raise ValueError(f"Data for base asset {base_asset} not found in datamap")

        prices_lf = (
            datamap.get(info).lazy().with_columns(pl.col("price").shift(-price_shift))
        )

        df = (
            prices_lf.group_by_dynamic(index_column="start_time", every=interval)
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
                .abs()
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

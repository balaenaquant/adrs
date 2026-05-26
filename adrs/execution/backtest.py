import polars as pl

from datetime import datetime

from adrs import Alpha
from adrs.performance.evaluator import Evaluator, Datamap


def generate_signal_df(
    alphas: list[Alpha],
    metadata_df: pl.DataFrame,
    evaluator: Evaluator,
    datamap: Datamap,
    start_time: datetime,
    end_time: datetime,
) -> pl.DataFrame | None:
    signal_df: pl.DataFrame | None = None
    for alpha in alphas:
        alpha_meta = metadata_df.filter(pl.col("custom_id") == alpha.id)
        base_asset = alpha_meta["base_asset"][0]
        _, df = alpha.backtest(
            evaluator=evaluator,
            base_asset=base_asset,
            datamap=datamap,
            start_time=start_time,
            end_time=end_time,
            fees=alpha_meta["fees"][0],
            price_shift=alpha_meta["shift_backtest_candle_minute"][0],
        )
        alpha_signal = df.select(
            pl.col("start_time"),
            pl.col("signal").alias(alpha.id),
        )
        if signal_df is None:
            signal_df = alpha_signal
        else:
            signal_df = (
                signal_df.join(alpha_signal, on="start_time", how="full", coalesce=True)
                .sort("start_time")
                .with_columns(pl.col(alpha.id).forward_fill())
            )
    return signal_df

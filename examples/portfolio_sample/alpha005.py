import polars as pl

from adrs import Alpha
from adrs.data import DataInfo, DataColumn, DataProcessor


class Alpha005(Alpha):
    """ETH RSI mean reversion."""

    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(
            id="alpha_005_eth_rsi_test",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=ETHUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close")],
                    lookback_size=window + 1,
                ),
            ],
            data_processor=DataProcessor(),
        )
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def next(self, data_df: pl.DataFrame) -> pl.DataFrame:
        df = data_df.with_columns(
            pl.col("close").diff().alias("delta")
        ).with_columns(
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"),
        ).with_columns(
            pl.col("gain").rolling_mean(self.window).alias("avg_gain"),
            pl.col("loss").rolling_mean(self.window).alias("avg_loss"),
        ).with_columns(
            (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("rsi")
        ).filter(pl.col("rsi").is_finite()).with_columns(
            pl.when(pl.col("rsi") <= self.oversold)
            .then(1)
            .when(pl.col("rsi") >= self.overbought)
            .then(-1)
            .when(pl.col("rsi").is_between(45, 55))
            .then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(strategy="zero")
            .alias("signal")
        )
        return df.select("start_time", "signal")

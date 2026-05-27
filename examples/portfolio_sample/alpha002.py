import polars as pl

from adrs import Alpha
from adrs.data import DataInfo, DataColumn, DataProcessor


class Alpha002(Alpha):
    """BTC exchange spread z-score (Binance vs Bybit)."""

    def __init__(self, window: int = 100, entry_threshold: float = 1.5):
        super().__init__(
            id="alpha_002_btc_spread_zscore_test",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=BTCUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_binance")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="bybit-linear|candle?symbol=BTCUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_bybit")],
                    lookback_size=window,
                ),
            ],
            data_processor=DataProcessor(),
        )
        self.window = window
        self.entry_threshold = entry_threshold

    def next(self, data_df: pl.DataFrame) -> pl.DataFrame:
        df = data_df.with_columns(
            (
                (pl.col("close_binance") - pl.col("close_bybit"))
                .rolling_mean(self.window)
                .alias("spread_mean")
            ),
            (
                (pl.col("close_binance") - pl.col("close_bybit"))
                .rolling_std(self.window, ddof=1)
                .alias("spread_std")
            ),
        ).with_columns(
            (
                (pl.col("close_binance") - pl.col("close_bybit") - pl.col("spread_mean"))
                / pl.col("spread_std")
            ).alias("zscore")
        ).filter(pl.col("zscore").is_finite()).with_columns(
            pl.when(pl.col("zscore") >= self.entry_threshold)
            .then(-1)
            .when(pl.col("zscore") <= -self.entry_threshold)
            .then(1)
            .when(pl.col("zscore").abs() < 0.2)
            .then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(strategy="zero")
            .alias("signal")
        )
        return df.select("start_time", "signal")

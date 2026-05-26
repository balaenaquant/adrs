import polars as pl

from adrs import Alpha
from adrs.data import DataInfo, DataColumn, DataProcessor


class Alpha001(Alpha):
    def __init__(self, window: int, entry_threshold: float, exit_threshold: float):
        super().__init__(
            id="alpha_001_testing_test",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=BTCUSDT&interval=1m",
                    columns=[DataColumn(src="close", dst="close_binance_spot")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="bybit-spot|candle?symbol=BTCUSDT&interval=1m",
                    columns=[DataColumn(src="close", dst="close_bybit_spot")],
                    lookback_size=window,
                ),
            ],
            data_processor=DataProcessor(),
        )
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def next(self, data_df):
        spread = pl.col("close_binance_spot") - pl.col("close_bybit_spot")
        df = data_df.with_columns(
            (
                (spread - spread.rolling_mean(self.window))
                / spread.rolling_std(self.window, ddof=1)
            ).alias("zscore")
        ).filter(pl.col("zscore").is_finite())

        z = pl.col("zscore")
        entry = (
            pl.when(z >= self.entry_threshold)
            .then(1)
            .when(z <= -self.entry_threshold)
            .then(-1)
            .otherwise(None)
        )
        prev = entry.forward_fill().fill_null(0)
        signal = (
            pl.when(z >= self.entry_threshold)
            .then(1)
            .when(z <= -self.entry_threshold)
            .then(-1)
            .when(
                ((prev == 1) & (z <= self.exit_threshold))
                | ((prev == -1) & (z >= -self.exit_threshold))
            )
            .then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(0)
        )
        df = df.with_columns(signal.alias("signal"))

        return df

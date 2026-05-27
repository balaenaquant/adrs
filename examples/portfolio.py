import json
import asyncio
import logging
import polars as pl
from typing import override
from datetime import datetime

from adrs import Alpha, DataLoader
from adrs.execution import MeanWeightAllocator, generate_signal_df
from adrs.performance import Evaluator
from adrs.data import DataInfo, DataColumn, DataProcessor, make_datamap
from adrs.report.portfolio import PortfolioReportV1

from adrs.logging import setup_logger


class CoinbasePremiumZScore(Alpha):
    def __init__(
        self,
        asset: str,
        id: str | None = None,
        window: int = 100,
        long_entry_threshold: float | None = None,
        long_exit_threshold: float | None = None,
        short_entry_threshold: float | None = None,
        short_exit_threshold: float | None = None,
    ):
        super().__init__(
            id=id
            if id is not None
            else f"zscore_{'long' if long_entry_threshold is not None else 'short'}_{asset}",
            data_infos=[
                DataInfo(
                    topic=f"binance-spot|candle?symbol={asset}USDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_binance_spot")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic=f"coinbase|candle?symbol={asset}USD&interval=1h",
                    columns=[DataColumn(src="close", dst="close_coinbase")],
                    lookback_size=window,
                ),
            ],
            data_processor=DataProcessor(),
        )
        self.window = window
        self.long_entry_threshold = long_entry_threshold
        self.long_exit_threshold = long_exit_threshold
        self.short_entry_threshold = short_entry_threshold
        self.short_exit_threshold = short_exit_threshold

    @override
    def next(self, data_df: pl.DataFrame):
        # alpha formula
        df = data_df.select(
            pl.col("start_time"),
            (pl.col("close_coinbase") - pl.col("close_binance_spot")).alias("data"),
        )

        # pre-process
        df = df

        # modeling
        df = df.with_columns(
            (
                (pl.col("data") - pl.col("data").rolling_mean(self.window))
                / pl.col("data").rolling_std(self.window, ddof=1)
            ).alias("zscore")
        ).filter(pl.col("zscore").is_finite())

        # signal
        if (
            self.long_entry_threshold is not None
            and self.long_exit_threshold is not None
        ):
            df = df.with_columns(
                pl.when(pl.col("zscore") >= self.long_entry_threshold)
                .then(1)
                .when(pl.col("zscore") <= self.long_exit_threshold)
                .then(0)
                .otherwise(None)
                .forward_fill()
                .fill_null(strategy="zero")
                .alias("signal")
            )
        elif (
            self.short_entry_threshold is not None
            and self.short_exit_threshold is not None
        ):
            df = df.with_columns(
                pl.when(pl.col("zscore") >= self.short_entry_threshold)
                .then(-1)
                .when(pl.col("zscore") <= self.short_exit_threshold)
                .then(0)
                .otherwise(None)
                .forward_fill()
                .fill_null(strategy="zero")
                .alias("signal")
            )
        else:
            raise ValueError("not enough thresholds given")

        print(self.id, df)

        return df


async def main():
    setup_logger(log_level=logging.INFO)

    # some alpha backtest args
    start_time, end_time = (
        datetime.fromisoformat("2025-12-01T00:00:00Z"),
        datetime.fromisoformat("2026-01-03T00:00:00Z"),
    )
    dataloader = DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
    )
    evaluator = Evaluator(
        assets={
            "BTC": DataInfo(
                topic="bybit-linear|candle?symbol=BTCUSDT&interval=1m",
                columns=[DataColumn(src="close", dst="price")],
                lookback_size=0,
            ),
            "ETH": DataInfo(
                topic="binance-spot|candle?symbol=ETHUSDT&interval=1m",
                columns=[DataColumn(src="close", dst="price")],
                lookback_size=0,
            ),
        }
    )

    btc_alphas: list[Alpha] = [
        CoinbasePremiumZScore(
            asset="BTC",
            id="btc_zscore_long",
            window=40,
            long_entry_threshold=0.825,
            long_exit_threshold=-0.825,
        ),
        CoinbasePremiumZScore(
            asset="BTC",
            id="btc_zscore_short",
            window=40,
            short_entry_threshold=-0.825,
            short_exit_threshold=0.825,
        ),
    ]
    eth_alphas: list[Alpha] = [
        CoinbasePremiumZScore(
            asset="ETH",
            id="eth_zscore_long",
            window=40,
            long_entry_threshold=0.825,
            long_exit_threshold=-0.825,
        ),
        CoinbasePremiumZScore(
            asset="ETH",
            id="eth_zscore_short",
            window=40,
            short_entry_threshold=-0.825,
            short_exit_threshold=0.825,
        ),
    ]

    all_alphas = btc_alphas + eth_alphas
    metadata_df = pl.DataFrame(
        {
            "custom_id": [a.id for a in all_alphas],
            "base_asset": ["BTC"] * len(btc_alphas) + ["ETH"] * len(eth_alphas),
            "shift_backtest_candle_minute": [10] * len(all_alphas),
            "fees": [0.035] * len(all_alphas),
        }
    )

    datamap = await make_datamap(
        dataloader=dataloader,
        start_time=start_time,
        end_time=end_time,
        data_infos=btc_alphas[0].data_infos + eth_alphas[0].data_infos,
        evaluator=evaluator,
    )

    signal_df = generate_signal_df(
        alphas=btc_alphas + eth_alphas,
        metadata_df=metadata_df,
        evaluator=evaluator,
        datamap=datamap,
        start_time=start_time,
        end_time=end_time,
    )

    assert signal_df is not None
    weight_df = MeanWeightAllocator(signal_df, metadata_df).weights()

    btc_prices = datamap.get(evaluator.assets["BTC"]).select(
        pl.col("start_time"), pl.col("price").alias("BTC")
    )
    eth_prices = datamap.get(evaluator.assets["ETH"]).select(
        pl.col("start_time"), pl.col("price").alias("ETH")
    )
    prices_df = btc_prices.join(eth_prices, on="start_time", how="full", coalesce=True)

    report = PortfolioReportV1.compute(
        id="TEST_PORTFOLIO",
        signal_df=signal_df,
        metadata_df=metadata_df,
        weight_df=weight_df,
        prices_df=prices_df,
        back_pct=0.7,
    )
    print(report.back.performance)

    # with open("multi_asset_portfolio_v1_report.parquet", "wb") as f:
    #     f.write(report.serialize())


if __name__ == "__main__":
    asyncio.run(main())

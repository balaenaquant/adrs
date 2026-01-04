import json
import asyncio
import logging
import polars as pl
from typing import override
from decimal import Decimal
from datetime import datetime, timedelta

from cybotrade.logging import setup_logger

from adrs.data import DataInfo, DataColumn, DataProcessor, Datamap
from adrs.report.portfolio import MultiAssetPortfolioReportV1
from adrs import Alpha, DataLoader
from adrs.alpha import AlphaBacktestArgs
from adrs.performance import Evaluator
from adrs.portfolio import (
    Portfolio,
    AlphaGroup,
    # MultiAssetPortfolio,
    AlphaPerformances,
    AlphaWeights,
    # PortfolioWeights,
    AssetWeights,
)


class ZScoreLongETH(Alpha):
    def __init__(
        self,
        window: int = 100,
        long_entry_threshold: float = 1.0,
        long_exit_threshold: float = 1.0,
    ):
        super().__init__(
            id="zscore_long_eth",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=ETHUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_binance_spot")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="coinbase|candle?symbol=ETHUSD&interval=1h",
                    columns=[DataColumn(src="close", dst="close_coinbase")],
                    lookback_size=window,
                ),
            ],
        )
        self.window = window
        self.long_entry_threshold = long_entry_threshold
        self.long_exit_threshold = long_exit_threshold


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

        return df


class ZScoreShortETH(Alpha):
    def __init__(
        self,
        window: int = 100,
        short_entry_threshold: float = 1.0,
        short_exit_threshold: float = -1.0,
    ):
        super().__init__(
            id="zscore_short_eth",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=ETHUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_binance_spot")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="coinbase|candle?symbol=ETHUSD&interval=1h",
                    columns=[DataColumn(src="close", dst="close_coinbase")],
                    lookback_size=window,
                ),
            ],
        )
        self.window = window
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
        df = df.with_columns(
            pl.when(pl.col("zscore") <= self.short_entry_threshold)
            .then(-1)
            .when(pl.col("zscore") >= self.short_exit_threshold)
            .then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(strategy="zero")
            .alias("signal")
        )

        return df


class ZScoreLongBTC(Alpha):
    def __init__(
        self,
        window: int = 100,
        long_entry_threshold: float = 1.0,
        long_exit_threshold: float = 1.0,
    ):
        super().__init__(
            id="zscore_long_btc",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=BTCUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_binance_spot")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="coinbase|candle?symbol=BTCUSD&interval=1h",
                    columns=[DataColumn(src="close", dst="close_coinbase")],
                    lookback_size=window,
                ),
            ],
        )
        self.window = window
        self.long_entry_threshold = long_entry_threshold
        self.long_exit_threshold = long_exit_threshold


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

        return df


class ZScoreShortBTC(Alpha):
    def __init__(
        self,
        window: int = 100,
        short_entry_threshold: float = 1.0,
        short_exit_threshold: float = -1.0,
    ):
        super().__init__(
            id="zscore_short_btc",
            data_infos=[
                DataInfo(
                    topic="binance-spot|candle?symbol=BTCUSDT&interval=1h",
                    columns=[DataColumn(src="close", dst="close_binance_spot")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="coinbase|candle?symbol=BTCUSD&interval=1h",
                    columns=[DataColumn(src="close", dst="close_coinbase")],
                    lookback_size=window,
                ),
            ],
        )
        self.window = window
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
        df = df.with_columns(
            pl.when(pl.col("zscore") <= self.short_entry_threshold)
            .then(-1)
            .when(pl.col("zscore") >= self.short_exit_threshold)
            .then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(strategy="zero")
            .alias("signal")
        )

        return df


def mean_allocator(performances: AlphaPerformances) -> AlphaWeights:
    n = len(performances)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {alpha_id: weight for alpha_id in performances.keys()}


def mean_asset_allocator(asset_group: dict[str, AlphaGroup]) -> AssetWeights:
    n = len(asset_group)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {asset: weight for asset in asset_group.keys()}


def eighty_twenty_asset_allocator(asset_group: dict[str, AlphaGroup]) -> AssetWeights:
    if asset_group:
        pass

    return {
        "BTC": Decimal("0.8"),
        "ETH": Decimal("0.2"),
    }


async def main():
    setup_logger(log_level=logging.INFO)

    # some alpha backtest args
    fees = 0.035
    start_time, end_time = (
        datetime.fromisoformat("2020-05-11T00:00:00Z"),
        datetime.fromisoformat("2025-01-01T00:00:00Z"),
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
            )
        }
    )

    # create list of Alpha
    btc_alphas: list[Alpha] = [
        ZScoreLongBTC(
            window=40,
            long_entry_threshold=0.825,
            long_exit_threshold=-0.825,
        ),
        ZScoreShortBTC(
            window=40,
            short_entry_threshold=-0.825,
            short_exit_threshold=0.825,
        ),
    ]
    eth_alphas: list[Alpha] = [
        ZScoreLongETH(
            window=40,
            long_entry_threshold=0.825,
            long_exit_threshold=-0.825,
        ),
        ZScoreShortETH(
            window=40,
            short_entry_threshold=-0.825,
            short_exit_threshold=0.825,
        ),
    ]

    # Setup the datamap for alphas (download data)
    datamap = Datamap()
    for alpha in btc_alphas:
        await datamap.init(
            dataloader=dataloader,
            infos=alpha.data_infos,
            start_time=start_time,
            end_time=end_time,
        )
    for alpha in eth_alphas:
        await datamap.init(
            dataloader=dataloader,
            infos=alpha.data_infos,
            start_time=start_time,
            end_time=end_time,
        )
    # download data with (+1 day offset for candle shift)
    await datamap.init(
        dataloader=dataloader,
        infos=list(evaluator.assets.values()),
        start_time=start_time,
        end_time=end_time + timedelta(days=1),
    )

    # create list of AlphaBacktestArgs
    btc_args: list[AlphaBacktestArgs] = []
    for alpha in btc_alphas:
        data_df = alpha.data_processor.process(datamap)
        if data_df is None:
            raise Exception("Failed to process datamap to get the data_df")
        btc_args.append(AlphaBacktestArgs(
            evaluator=evaluator,
            base_asset="BTC",
            datamap=datamap,
            data_df=data_df,
            start_time=start_time,
            end_time=end_time,
            fees=fees,
            interval=timedelta(hours=1),
            price_shift=10,  # assume 10 minutes delay
        ))
    eth_args: list[AlphaBacktestArgs] = []
    for alpha in eth_alphas:
        data_df = alpha.data_processor.process(datamap)
        if data_df is None:
            raise Exception("Failed to process datamap to get the data_df")
        eth_args.append(AlphaBacktestArgs(
            evaluator=evaluator,
            base_asset="ETH",
            datamap=datamap,
            data_df=data_df,
            start_time=start_time,
            end_time=end_time,
            fees=fees,
            interval=timedelta(hours=1),
            price_shift=10,  # assume 10 minutes delay
        ))

    # create portfolio
    portfolio = Portfolio(
        id="TEST_PORTFOLIO",
        asset_group={
            "BTC": AlphaGroup(
                alphas_list=btc_alphas,
                args_list=btc_args,
                alpha_allocator=mean_allocator,
            ),
            "ETH": AlphaGroup(
                alphas_list=eth_alphas,
                args_list=eth_args,
                alpha_allocator=mean_allocator,
            )
        },
        # asset_allocator=eighty_twenty_asset_allocator,
        start_time=start_time,
        end_time=end_time,
    )

    report = MultiAssetPortfolioReportV1.compute(
        portfolio=portfolio,
        B_start=start_time,
        B_end=end_time,
        F_start=datetime.fromisoformat("2024-07-01T00:00:00Z"),
        F_end=datetime.fromisoformat("2025-07-01T00:00:00Z"),
    )
    logging.info(report.back.performance_df.columns)
    print(report.back.performance_df)
    print(report.back.performance_df.select([c for c in report.back.performance_df.columns if c.endswith("signal")]))
    print(report.back.performance)

    # with open("multi_asset_portfolio_v1_report.parquet", "wb") as f:
    #     f.write(report.serialize())


if __name__ == "__main__":
    asyncio.run(main())

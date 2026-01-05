import json
import asyncio
import logging
import polars as pl
from typing import override
from decimal import Decimal
from datetime import datetime, timedelta

from adrs import Alpha, DataLoader
from adrs.performance import Evaluator
from adrs.utils import backforward_split
from adrs.data import DataInfo, DataColumn, Datamap
from adrs.report.portfolio import PortfolioReportV1
from adrs.portfolio import (
    Portfolio,
    Asset,
    AlphaGroup,
    AlphaPerformances,
    AlphaWeights,
    AssetWeights,
)

from cybotrade.logging import setup_logger


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


def eighty_twenty_asset_allocator(_: dict[str, AlphaGroup]) -> AssetWeights:
    return {
        "BTC": Decimal("0.8"),
        "ETH": Decimal("0.2"),
    }


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

    # create list of Alpha
    btc_alphas: list[Alpha] = [
        CoinbasePremiumZScore(
            asset="BTC",
            window=40,
            long_entry_threshold=0.825,
            long_exit_threshold=-0.825,
        ),
        CoinbasePremiumZScore(
            asset="BTC",
            window=40,
            short_entry_threshold=-0.825,
            short_exit_threshold=0.825,
        ),
    ]
    eth_alphas: list[Alpha] = [
        CoinbasePremiumZScore(
            asset="ETH",
            window=40,
            long_entry_threshold=0.825,
            long_exit_threshold=-0.825,
        ),
        CoinbasePremiumZScore(
            asset="ETH",
            window=40,
            short_entry_threshold=-0.825,
            short_exit_threshold=0.825,
        ),
    ]

    # Setup the datamap for alphas (download data)
    datamap = Datamap()
    await datamap.init(
        dataloader=dataloader,
        infos=btc_alphas[0].data_infos,
        start_time=start_time,
        end_time=end_time,
    )
    for alpha in eth_alphas:
        await datamap.init(
            dataloader=dataloader,
            infos=eth_alphas[0].data_infos,
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

    # create portfolio
    portfolio = Portfolio(
        id="TEST_PORTFOLIO",
        assets=[
            Asset(
                name="BTC",
                alphas=btc_alphas,
                fees=0.035,
                price_shift=10,
                allocator=mean_allocator,
            ),
            Asset(
                name="ETH",
                alphas=eth_alphas,
                fees=0.035,
                price_shift=10,
                allocator=mean_allocator,
            ),
        ],
        evaluator=evaluator,
        datamap=datamap,
        start_time=start_time,
        end_time=end_time,
        asset_allocator=lambda _: {
            "BTC": Decimal("0.8"),
            "ETH": Decimal("0.2"),
        },
    )

    B_start, B_end, F_start, F_end = backforward_split(
        start_time=start_time, end_time=end_time, size=(0.7, 0.3)
    )
    report = PortfolioReportV1.compute(
        portfolio=portfolio,
        B_start=B_start,
        B_end=B_end,
        F_start=F_start,
        F_end=F_end,
    )
    logging.info(report.back.performance_df.columns)
    print(report.back.performance_df)
    print(
        report.back.performance_df.select(
            [c for c in report.back.performance_df.columns if c.endswith("signal")]
        )
    )
    print(report.back.performance)

    # with open("multi_asset_portfolio_v1_report.parquet", "wb") as f:
    #     f.write(report.serialize())


if __name__ == "__main__":
    asyncio.run(main())

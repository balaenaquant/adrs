import json
import asyncio
import logging
import polars as pl
from decimal import Decimal
from datetime import datetime
from typing import override, cast

from cybotrade.logging import setup_logger

from adrs.data import DataInfo, DataColumn, UniformDataProcessor
from adrs.report.portfolio import MultiAssetPortfolioReportV1
from adrs import Alpha, Environment, DataLoader, AlphaConfig
from adrs.performance import Evaluator
from adrs.signal import Long, Short
from adrs.signal import Signal
from adrs.model import ZScore
from adrs.portfolio import (
    Portfolio,
    MultiAssetPortfolio,
    AlphaPerformances,
    AlphaWeights,
    PortfolioWeights,
)


class ZScoreLong(Alpha):
    def __init__(
        self,
        window: int = 100,
        long_entry_thres: float = 1.0,
        long_exit_thres: float = 1.0,
    ):
        super().__init__()
        self.window = window
        self.long_entry_thres = long_entry_thres
        self.long_exit_thres = long_exit_thres

    @staticmethod
    def id():
        return cast(str, "zscore_long")

    @staticmethod
    def parameters_description():
        return {
            "rolling_window": "(int) rolling window of data",
            "long_entry_threshold": "(float) threshold for entering a long position",
            "long_exit_threshold": "(float) threshold for exiting a long position",
        }

    @override
    def next(self, datas_df: pl.DataFrame):
        # alpha formula
        datas_df = datas_df.select(
            pl.col("start_time"),
            (pl.col("close_coinbase") - pl.col("close_binance_spot")).alias("data"),
        )

        # pre-process
        df = datas_df

        # modeling
        model = ZScore(window=self.window)
        output = model.eval(df["data"])[0]
        df = df.with_columns(output.alias("data")).drop_nulls()

        # signal
        signals = Long(
            long_entry_thres=self.long_entry_thres,
            long_exit_thres=self.long_exit_thres,
        ).generate(
            df["data"].to_numpy(), df["start_time"].dt.epoch(time_unit="ms").to_numpy()
        )
        df = df.with_columns(signal=pl.Series(signals))

        return signals[-1] if len(signals) > 0 else Signal.NONE, df


class ZScoreShort(Alpha):
    def __init__(
        self,
        window: int = 100,
        short_entry_thres: float = 1.0,
        short_exit_thres: float = -1.0,
    ):
        super().__init__()
        self.window = window
        self.short_entry_thres = short_entry_thres
        self.short_exit_thres = short_exit_thres

    @staticmethod
    def id():
        return cast(str, "zscore_short")

    @staticmethod
    def parameters_description():
        return {
            "rolling_window": "(int) rolling window of data",
            "short_entry_threshold": "(float) threshold for entering a long position",
            "short_exit_threshold": "(float) threshold for exiting a long position",
        }

    @override
    def next(self, datas_df: pl.DataFrame):
        # alpha formula
        datas_df = datas_df.select(
            pl.col("start_time"),
            (pl.col("close_coinbase") - pl.col("close_binance_spot")).alias("data"),
        )

        # pre-process
        df = datas_df

        # modeling
        model = ZScore(window=self.window)
        output = model.eval(df["data"])[0]
        df = df.with_columns(output.alias("data")).drop_nulls()

        # signal
        signals = Short(
            short_entry_thres=self.short_entry_thres,
            short_exit_thres=self.short_exit_thres,
        ).generate(
            df["data"].to_numpy(), df["start_time"].dt.epoch(time_unit="ms").to_numpy()
        )
        df = df.with_columns(signal=pl.Series(signals))

        return signals[-1] if len(signals) > 0 else Signal.NONE, df


class ZScoreLongETH(ZScoreLong):
    @staticmethod
    def id():  # type: ignore
        return "zscore_long_eth"


class ZScoreShortETH(ZScoreShort):
    @staticmethod
    def id():  # type: ignore
        return "zscore_short_eth"


def make_alpha_config(
    dataloader: DataLoader,
    environment: Environment,
    start_time: datetime,
    end_time: datetime,
    base_asset: str = "BTC",
):
    return AlphaConfig(
        base_asset=base_asset,
        data_infos=[
            DataInfo(
                topic=f"binance-spot|candle?symbol={base_asset}USDT&interval=1h",
                columns=[DataColumn(src="close", dst="close_binance_spot")],
                lookback_size=100,
            ),
            DataInfo(
                topic=f"coinbase|candle?symbol={base_asset}USD&interval=1h",
                columns=[DataColumn(src="close", dst="close_coinbase")],
                lookback_size=100,
            ),
        ],
        dataloader=dataloader,
        data_processor=UniformDataProcessor(),
        start_time=start_time,
        end_time=end_time,
        environment=environment,
    )


BTC_ALPHA_CONFIG = make_alpha_config(
    dataloader=DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
    ),
    environment=Environment.BACKTEST,
    base_asset="BTC",
    start_time=datetime.fromisoformat("2020-06-01T00:00:00Z"),
    end_time=datetime.fromisoformat("2025-07-01T00:00:00Z"),
)
ETH_ALPHA_CONFIG = make_alpha_config(
    dataloader=DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
    ),
    environment=Environment.BACKTEST,
    base_asset="ETH",
    start_time=datetime.fromisoformat("2020-06-01T00:00:00Z"),
    end_time=datetime.fromisoformat("2025-07-01T00:00:00Z"),
)
btc_evaluator = Evaluator(fees=0.035, candle_shift=2)
eth_evaluator = Evaluator(fees=0.035, candle_shift=2)
alphas: list[Alpha] = [
    ZScoreLong(),
    ZScoreShort(),
]


def mean_allocator(performances: AlphaPerformances) -> AlphaWeights:
    n = len(performances)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {alpha_id: weight for alpha_id in performances.keys()}


def mean_portfolio_allocator(portfolios: dict[str, Portfolio]) -> PortfolioWeights:
    for base_asset, portfolio in portfolios.items():
        perf, df = portfolio.backtest()

    return {"BTC": Decimal("0.8"), "ETH": Decimal("0.2")}


async def main():
    setup_logger(log_level=logging.INFO)

    eth_alphas: list[Alpha] = [
        ZScoreLongETH(),
        ZScoreShortETH(),
    ]

    for alpha in alphas:
        await alpha.init(BTC_ALPHA_CONFIG, btc_evaluator)
        alpha.backtest()

    for alpha in eth_alphas:
        await alpha.init(ETH_ALPHA_CONFIG, eth_evaluator)
        alpha.backtest()

    start_time, end_time = (
        datetime.fromisoformat("2020-06-01T00:00:00Z"),
        datetime.fromisoformat("2024-07-01T00:00:00Z"),
    )

    portfolio = MultiAssetPortfolio(
        id="TEST",
        portfolios=[
            Portfolio(
                id="ETH_TEST",
                alphas=eth_alphas,
                allocator=mean_allocator,
                start_time=start_time,
                end_time=end_time,
            ),
            Portfolio(
                id="BTC_TEST",
                alphas=alphas,
                allocator=mean_allocator,
                start_time=start_time,
                end_time=end_time,
            ),
        ],
        allocator=mean_portfolio_allocator,
    )
    report = MultiAssetPortfolioReportV1.compute(
        portfolio=portfolio,
        B_start=start_time,
        B_end=end_time,
        F_start=datetime.fromisoformat("2024-07-01T00:00:00Z"),
        F_end=datetime.fromisoformat("2025-07-01T00:00:00Z"),
    )
    logging.info(report.back.performance_df.columns)

    # with open("multi_asset_portfolio_v1_report.parquet", "wb") as f:
    #     f.write(report.serialize())


if __name__ == "__main__":
    asyncio.run(main())

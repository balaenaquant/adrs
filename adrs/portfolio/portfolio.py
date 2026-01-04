import math
import logging
import polars as pl
import numpy as np
import pandera.polars as pa
from decimal import Decimal
from typing import Callable, cast, Any
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from pandera.typing.polars import DataFrame
from pandera.engines.polars_engine import DateTime, Float64

from adrs.types import Performance
from adrs.alpha import Alpha, AlphaBacktestArgs
from adrs.performance.metric import Ratio, Trade, Drawdown
from adrs.performance.metric.metric import Metrics


logger = logging.getLogger(__name__)


AlphaPerformances = dict[str, tuple["Performance", "pl.DataFrame"]]
AlphaWeights = dict[str, "Decimal"]
AlphaWeightAllocator = Callable[["AlphaPerformances"], "AlphaWeights"]

AssetWeights = dict[str, "Decimal"]
AssetGroup = dict[str, "AlphaGroup"]
AssetPerformances = dict[str, tuple["Performance", "pl.DataFrame"]]
AssetWeightAllocator = Callable[[dict[str, "AlphaGroup"]], "AssetWeights"]


class TradePerformance(BaseModel):
    largest_loss: float
    num_datapoints: int
    num_trades: int
    avg_holding_time_in_seconds: float
    long_trades: int
    short_trades: int
    win_trades: int
    lose_trades: int
    win_streak: int
    lose_streak: int
    win_rate: float

    model_config = ConfigDict(extra="allow")


class MultiAssetPortfolioPerformance(BaseModel):
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    cagr: float
    annualized_return: float
    total_return: float
    min_cumu: float
    start_time: datetime
    end_time: datetime
    max_drawdown: float
    max_drawdown_percentage: float
    max_drawdown_start_date: datetime
    max_drawdown_end_date: datetime
    max_drawdown_recover_date: datetime
    max_drawdown_max_duration_in_days: float
    trades: dict[str, TradePerformance]
    metadata: dict[str, Any]

    model_config = ConfigDict(extra="allow")


class MultiAssetPortfolioPerformanceDF(pa.DataFrameModel):
    start_time: DateTime = pa.Field(
        dtype_kwargs={"time_unit": "ms", "time_zone": "UTC"}
    )
    data: Float64 = pa.Field(nullable=True)
    signal: Float64 = pa.Field(coerce=True)
    prev_signal: Float64 = pa.Field(coerce=True)
    trade: Float64 = pa.Field(coerce=True)
    pnl: Float64
    equity: Float64


def single_alpha_allocator(performances: AlphaPerformances) -> AlphaWeights:
    alpha_id = (next(iter(performances.items())))[0]
    return {alpha_id: Decimal(1.0)}


def mean_alpha_allocator(performances: AlphaPerformances) -> AlphaWeights:
    n = len(performances)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {alpha_id: weight for alpha_id in performances.keys()}


class AlphaGroup:
    alphas_list: list[Alpha]
    args_list: list[AlphaBacktestArgs]
    alpha_allocator: AlphaWeightAllocator
    alpha_weights: AlphaWeights
    performances: AlphaPerformances
    metrics: list[Metrics]

    def __init__(
        self,
        alphas_list: list[Alpha],
        args_list: list[AlphaBacktestArgs],
        alpha_allocator: AlphaWeightAllocator | None = None,
        alpha_weights: AlphaWeights | None = None,
    ):
        self.alphas_list = alphas_list
        self.args_list = args_list
        self.performances: AlphaPerformances = {}
        self.metrics = [Ratio(), Trade(), Drawdown()]

        if alpha_allocator is None:
            if len(alphas_list) == 1:
                self.alpha_allocator = single_alpha_allocator
            else:
                self.alpha_allocator = mean_alpha_allocator
                logger.info("alpha group: using default mean alpha allocator")
        else:
            self.alpha_allocator = alpha_allocator

        if alpha_weights is None:
            self.alpha_weights = {}
        else:
            self.alpha_weights = alpha_weights

        if len(self.alphas_list) != len(self.args_list):
            raise ValueError("alphas list and args list must have the same length")

    
    def backtest_alphas(self):
        for alpha, args in zip(self.alphas_list, self.args_list, strict=True):
            self.performances[alpha.id] = alpha.backtest(**args)


    def set_weights(self):
        self.alpha_weights = self.alpha_allocator(self.performances)


def single_asset_allocator(asset_group: dict[str, AlphaGroup]) -> AssetWeights:
    base_asset = (next(iter(asset_group.items())))[0]
    return {base_asset: Decimal(1.0)}


def mean_asset_allocator(asset_group: dict[str, AlphaGroup]) -> AssetWeights:
    n = len(asset_group)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {asset: weight for asset in asset_group.keys()}


class Portfolio:
    id: str
    asset_group: AssetGroup
    start_time: datetime
    end_time: datetime
    asset_weights: AssetWeights
    asset_allocator: AssetWeightAllocator
    performances: AssetPerformances
    metrics: list[Metrics]

    def __init__(
        self,
        id: str,
        asset_group: AssetGroup,
        start_time: datetime,
        end_time: datetime,
        asset_weights: AssetWeights | None = None,
        asset_allocator: AssetWeightAllocator | None = None,
    ):
        self.id = id
        self.asset_group = asset_group
        self.start_time = start_time
        self.end_time = end_time
        self.performances: AssetPerformances = {}
        self.metrics = [Ratio(), Drawdown()]

        if asset_allocator is None:
            if len(asset_group) == 1:
                self.asset_allocator = single_asset_allocator
            else:
                self.asset_allocator = mean_asset_allocator
                logger.info("portfolio: using default mean asset allocator")
        else:
            self.asset_allocator = asset_allocator

        if asset_weights is None:
            self.asset_weights = {}
        else:
            self.asset_weights = asset_weights

        for asset, alpha_group in self.asset_group.items():
            # Check if there is at least one alpha in each asset group
            if len(alpha_group.alphas_list) == 0:
                raise ValueError(f"{asset} group must have at least one alpha")
            # Check if the base asset is the same across alphas in same asset group
            base_asset = (alpha_group.args_list[0])["base_asset"]
            for args in alpha_group.args_list:
                if args["base_asset"] != base_asset:
                    raise ValueError(f"All alphas in the {asset} group must have the same base asset")

        if self.asset_weights and set(self.asset_weights) != set(self.asset_group):
            raise ValueError("asset weights keys must match asset group keys")


    def backtest(
        self,
    ) -> tuple[
        MultiAssetPortfolioPerformance, DataFrame[MultiAssetPortfolioPerformanceDF]
    ]:
        for asset, alpha_group in self.asset_group.items():
            if len(alpha_group.performances) == 0:
                alpha_group.backtest_alphas()

            if len(alpha_group.alpha_weights) == 0:
                alpha_group.set_weights()

            if not math.isclose(sum(alpha_group.alpha_weights.values()), 1.0):
                raise Exception(f"Alpha weights in {asset} group must sum to 1.0")

        if len(self.performances) == 0:
            self.performances = self.backtest_asset_group()

        if len(self.asset_weights) == 0:
            self.set_weights()

        if not math.isclose(sum(self.asset_weights.values()), 1.0):
            raise Exception(f"Asset weights must sum to 1.0")

        # Combine the performances
        merged_df: pl.DataFrame | None = None
        trade_performances: dict[str, TradePerformance] = {}
        for asset, (performance, df) in self.performances.items():
            trade_performances[asset] = TradePerformance(
                largest_loss=performance.largest_loss,
                num_datapoints=performance.num_datapoints,
                num_trades=performance.num_trades,
                avg_holding_time_in_seconds=performance.avg_holding_time_in_seconds,
                long_trades=performance.long_trades,
                short_trades=performance.short_trades,
                win_trades=performance.win_trades,
                lose_trades=performance.lose_trades,
                win_streak=performance.win_streak,
                lose_streak=performance.lose_streak,
                win_rate=performance.win_rate,
            )

            weight = self.asset_weights.get(asset, Decimal(1.0))
            pnl_col = f"{asset}_pnl"
            signal_col = f"{asset}_signal"
            query = [
                pl.col("start_time"),
                pl.col("price").alias(f"{asset}_price"),
                (pl.col("pnl") * weight).alias(pnl_col),  # NOTE: Use Decimal
                (pl.col("signal").cast(pl.Float64) * weight).alias(signal_col),
            ]

            if merged_df is None:
                merged_df = df.select(query)
            else:
                merged_df = (
                    merged_df.join(
                        df.select(query),
                        on="start_time",
                        how="full",
                    )
                    .drop(["start_time_right"])
                    .with_columns(
                        pl.col(pnl_col).forward_fill(),
                        pl.col(signal_col).forward_fill(),
                    )
                )

        merged_df = cast(pl.DataFrame, merged_df)

        performance_df = merged_df.select(
            pl.col("start_time"),
            *[pl.col(f"{asset}_price") for asset in self.asset_group],
            *[pl.col(f"{asset}_signal") for asset in self.asset_group],
            pl.lit(None).alias("data").cast(pl.Float64),
            pl.sum_horizontal(
                [pl.col(c) for c in merged_df.columns if c.endswith("_pnl")]
            ).alias("pnl"),
            pl.sum_horizontal(
                [pl.col(c) for c in merged_df.columns if c.endswith("_signal")]
            ).alias("signal"),
        ).with_columns(
            pl.col("signal")
            .shift(1)
            .alias("prev_signal")
            .forward_fill()
            .fill_null(strategy="zero"),
            pl.col("signal").diff().alias("trade").fill_null(strategy="zero"),
            pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero"),
        )
        # .drop(
        #     "signal"
        # )

        trade_performances["total"] = TradePerformance(
            largest_loss=min(
                map(lambda t: t.largest_loss, trade_performances.values())
            ),
            num_datapoints=max(
                map(lambda t: t.num_datapoints, trade_performances.values())
            ),
            num_trades=sum(map(lambda t: t.num_trades, trade_performances.values())),
            avg_holding_time_in_seconds=float(
                np.mean(
                    list(
                        map(
                            lambda t: t.avg_holding_time_in_seconds,
                            trade_performances.values(),
                        )
                    )
                )
            ),
            long_trades=sum(map(lambda t: t.long_trades, trade_performances.values())),
            short_trades=sum(
                map(lambda t: t.short_trades, trade_performances.values())
            ),
            win_trades=sum(map(lambda t: t.win_trades, trade_performances.values())),
            lose_trades=sum(map(lambda t: t.lose_trades, trade_performances.values())),
            win_streak=min(map(lambda t: t.win_streak, trade_performances.values())),
            lose_streak=max(map(lambda t: t.lose_streak, trade_performances.values())),
            win_rate=sum(map(lambda t: t.win_trades, trade_performances.values()))
            / sum(map(lambda t: t.num_trades, trade_performances.values())),
        )
        performance = {
            "trades": trade_performances,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": {},
        }

        for metric in self.metrics:
            result = metric.compute(performance_df)
            performance.update(result)

        return (
            MultiAssetPortfolioPerformance.model_validate(performance),
            MultiAssetPortfolioPerformanceDF.validate(performance_df)
        )


    def backtest_asset_group(self) -> AssetPerformances:
        asset_performances: AssetPerformances = {}

        for asset, alpha_group in self.asset_group.items():
            merged_df: pl.DataFrame | None = None
            for alpha_id, (_, df) in alpha_group.performances.items():
                if alpha_group.alpha_weights is None:
                    weight = Decimal(1.0)
                else:
                    weight = alpha_group.alpha_weights[alpha_id]

                pnl_col = f"{alpha_id}_pnl"
                signal_col = f"{alpha_id}_signal"
                query = [
                    pl.col("start_time"),
                    pl.col("price"),
                    (pl.col("pnl") * weight).alias(pnl_col),
                    (pl.col("signal").cast(pl.Float64) * weight).alias(signal_col)
                ]

                if merged_df is None:
                    merged_df = df.select(query)
                else:
                    merged_df = (
                        merged_df.join(
                            df.select(query),
                            on="start_time",
                            how="full",
                        )
                        .drop(["start_time_right", "price_right"])
                        .with_columns(
                            pl.col(pnl_col).forward_fill(),
                            pl.col(signal_col).forward_fill(),
                        )
                    )

            merged_df = cast(pl.DataFrame, merged_df)

            pnl_cols = [c for c in merged_df.columns if c.endswith("_pnl")]
            signal_cols = [c for c in merged_df.columns if c.endswith("_signal")]

            # normalized signal to -1.0 to 1.0
            max_val = merged_df.select(pl.sum_horizontal(signal_cols).abs().max()).item()
            if max_val == 0:
                max_val = 1
            
            performance_df = merged_df.select(
                pl.col("start_time"),
                pl.col("price"),
                pl.lit(None).alias("data").cast(pl.Float64),
                pl.sum_horizontal(pnl_cols).alias("pnl"),
                *signal_cols,
            ).with_columns(
                (pl.sum_horizontal(signal_cols) / max_val).clip(-1.0, 1.0).alias("signal"),
            ).with_columns(
                pl.col("signal")
                .shift(1)
                .alias("prev_signal")
                .forward_fill()
                .fill_null(strategy="zero"),
                pl.col("price").pct_change().alias("returns").fill_null(strategy="zero"),
                pl.col("signal").diff().alias("trade").fill_null(strategy="zero"),
                pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero"),
            ) 

            performance: dict[str, Any] = {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "metadata": {},
            }
            for metric in alpha_group.metrics:
                result = metric.compute(performance_df)
                performance = {**performance, **result}

            asset_performances[asset] = (Performance.model_validate(performance), performance_df)

        return asset_performances


    def set_weights(self):
        self.asset_weights = self.asset_allocator(self.asset_group)

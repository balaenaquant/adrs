import math
import logging
import polars as pl
import numpy as np
import pandera.polars as pa
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, cast, Any
from pydantic import BaseModel, ConfigDict
from pandera.typing.polars import DataFrame
from pandera.engines.polars_engine import DateTime, Float64

from adrs.data import Datamap
from adrs.types import Performance
from adrs.alpha import Alpha, AlphaBacktestArgs
from adrs.performance.evaluator import Evaluator
from adrs.performance.metric import Ratio, Drawdown, Trade, Metrics


logger = logging.getLogger(__name__)


AlphaPerformances = dict[str, tuple[Performance, pl.DataFrame]]
AlphaWeights = dict[str, Decimal]
AlphaWeightAllocator = Callable[[AlphaPerformances], AlphaWeights]

AssetWeights = dict[str, Decimal]
AssetPerformances = dict[str, tuple[Performance, pl.DataFrame]]


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


class PortfolioPerformance(BaseModel):
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


class PortfolioPerformanceDF(pa.DataFrameModel):
    start_time: DateTime = pa.Field(
        dtype_kwargs={"time_unit": "ms", "time_zone": "UTC"}
    )
    data: Float64 = pa.Field(nullable=True)
    signal: Float64 = pa.Field(coerce=True)
    prev_signal: Float64 = pa.Field(coerce=True)
    trade: Float64 = pa.Field(coerce=True)
    pnl: Float64
    equity: Float64


def mean_alpha_allocator(performances: AlphaPerformances) -> AlphaWeights:
    n = len(performances)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {alpha_id: weight for alpha_id in performances.keys()}


@dataclass
class Asset:
    name: str
    alphas: list[Alpha]
    fees: float
    price_shift: int = 0
    allocator: AlphaWeightAllocator = mean_alpha_allocator


class AlphaGroup:
    base_asset: str
    alphas: list[tuple[Alpha, AlphaBacktestArgs]]
    alpha_allocator: AlphaWeightAllocator
    alpha_weights: dict[str, Decimal]
    performances: AlphaPerformances

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        base_asset: str,
        alphas: list[Alpha],
        datamap: Datamap,
        evaluator: Evaluator,
        fees: float,
        price_shift: int,
        alpha_allocator: AlphaWeightAllocator = mean_alpha_allocator,
    ):
        self.base_asset = base_asset
        self.alphas = list(
            map(
                lambda a: (
                    a,
                    AlphaBacktestArgs(
                        evaluator=evaluator,
                        base_asset=base_asset,
                        datamap=datamap,
                        start_time=start_time,
                        end_time=end_time,
                        fees=fees,
                        price_shift=price_shift,
                    ),
                ),
                alphas,
            )
        )
        self.performances = {}
        self.alpha_allocator = alpha_allocator
        self.alpha_weights = {}

    def backtest_alphas(self):
        for alpha, args in self.alphas:
            self.performances[alpha.id] = alpha.backtest(**args)
        self.alpha_weights = self.alpha_allocator(self.performances)


AssetWeightAllocator = Callable[[dict[str, AlphaGroup]], AssetWeights]


def mean_asset_allocator(asset_group: dict[str, AlphaGroup]) -> AssetWeights:
    n = len(asset_group)
    if n == 0:
        return {}
    weight = Decimal("1.0") / n
    return {asset: weight for asset in asset_group.keys()}


class Portfolio:
    id: str
    start_time: datetime
    end_time: datetime
    asset_group: dict[str, AlphaGroup]
    asset_weights: dict[str, Decimal]
    asset_allocator: AssetWeightAllocator
    performances: AssetPerformances
    metrics: list[Metrics]

    def __init__(
        self,
        id: str,
        start_time: datetime,
        end_time: datetime,
        datamap: Datamap,
        evaluator: Evaluator,
        assets: list[Asset],
        asset_allocator: AssetWeightAllocator = mean_asset_allocator,
    ):
        self.id = id
        self.asset_group = {
            a.name: AlphaGroup(
                start_time=start_time,
                end_time=end_time,
                base_asset=a.name,
                alphas=a.alphas,
                datamap=datamap,
                evaluator=evaluator,
                fees=a.fees,
                price_shift=a.price_shift,
                alpha_allocator=a.allocator,
            )
            for a in assets
        }
        self.start_time = start_time
        self.end_time = end_time
        self.performances: AssetPerformances = {}
        self.metrics = [Ratio(), Drawdown()]
        self.asset_allocator = asset_allocator
        self.asset_weights = {}

        for asset, alpha_group in self.asset_group.items():
            # Check if there is at least one alpha in each asset group
            if len(alpha_group.alphas) == 0:
                raise ValueError(f"{asset} group must have at least one alpha")

        if self.asset_weights and set(self.asset_weights) != set(self.asset_group):
            raise ValueError("asset weights keys must match asset group keys")

    def backtest(
        self,
    ) -> tuple[PortfolioPerformance, DataFrame[PortfolioPerformanceDF]]:
        if len(self.performances) == 0:
            self.performances = self.backtest_asset()
            self.asset_weights = self.asset_allocator(self.asset_group)

        if not math.isclose(sum(self.asset_weights.values()), 1.0):
            raise Exception("Asset weights must sum to 1.0")

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
            PortfolioPerformance.model_validate(performance),
            PortfolioPerformanceDF.validate(performance_df),
        )

    def backtest_asset(self) -> AssetPerformances:
        asset_performances: AssetPerformances = {}

        for asset, alpha_group in self.asset_group.items():
            # Make sure preconditions are correct
            if len(alpha_group.performances) == 0:
                alpha_group.backtest_alphas()
            if not math.isclose(sum(alpha_group.alpha_weights.values()), 1.0):
                raise Exception(f"Alpha weights in {asset} group must sum to 1.0")

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
                        .drop(["start_time_right", "price_right"])
                        .with_columns(
                            pl.col(pnl_col).forward_fill(),
                            pl.col(signal_col).forward_fill(),
                        )
                    )

            merged_df = cast(pl.DataFrame, merged_df)

            pnl_cols = [c for c in merged_df.columns if c.endswith("_pnl")]
            signal_cols = [c for c in merged_df.columns if c.endswith("_signal")]

            performance_df = merged_df.select(
                pl.col("start_time"),
                pl.col("price"),
                pl.lit(None).alias("data").cast(pl.Float64),
                pl.sum_horizontal(pnl_cols).alias("pnl"),
                pl.sum_horizontal(signal_cols).alias("signal"),
            ).with_columns(
                pl.col("signal")
                .shift(1)
                .alias("prev_signal")
                .forward_fill()
                .fill_null(strategy="zero"),
                pl.col("price")
                .pct_change()
                .alias("returns")
                .fill_null(strategy="zero"),
                pl.col("signal").diff().alias("trade").fill_null(strategy="zero"),
                pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero"),
            )

            performance: dict[str, Any] = {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "metadata": {},
            }
            for metric in [Ratio(), Drawdown(), Trade()]:
                result = metric.compute(performance_df)
                performance = {**performance, **result}

            asset_performances[asset] = (
                Performance.model_validate(performance),
                performance_df,
            )

        return asset_performances

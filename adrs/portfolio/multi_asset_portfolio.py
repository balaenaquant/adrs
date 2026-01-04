# import math
# import logging
# import numpy as np
# import polars as pl
# import pandera.polars as pa
# from decimal import Decimal
# from datetime import datetime
# from typing import Callable, Any, cast
# from pydantic import BaseModel, ConfigDict
# from pandera.typing.polars import DataFrame
# from pandera.engines.polars_engine import DateTime, Float64
#
# from adrs.types import Performance
# from adrs.portfolio import Portfolio
# from adrs.performance.metric import Metrics, Ratio, Drawdown
#
# logger = logging.getLogger(__name__)
#
#
# class TradePerformance(BaseModel):
#     largest_loss: float
#     num_datapoints: int
#     num_trades: int
#     avg_holding_time_in_seconds: float
#     long_trades: int
#     short_trades: int
#     win_trades: int
#     lose_trades: int
#     win_streak: int
#     lose_streak: int
#     win_rate: float
#
#     model_config = ConfigDict(extra="allow")
#
#
# # class MultiAssetPortfolioPerformance(BaseModel):
# #     sharpe_ratio: float
# #     calmar_ratio: float
# #     sortino_ratio: float
# #     cagr: float
# #     annualized_return: float
# #     total_return: float
# #     min_cumu: float
# #     start_time: datetime
# #     end_time: datetime
# #     max_drawdown: float
# #     max_drawdown_percentage: float
# #     max_drawdown_start_date: datetime
# #     max_drawdown_end_date: datetime
# #     max_drawdown_recover_date: datetime
# #     max_drawdown_max_duration_in_days: float
# #     trades: dict[str, TradePerformance]
# #     metadata: dict[str, Any]
# #
# #     model_config = ConfigDict(extra="allow")
# #
# #
# # class MultiAssetPortfolioPerformanceDF(pa.DataFrameModel):
# #     start_time: DateTime = pa.Field(
# #         dtype_kwargs={"time_unit": "ms", "time_zone": "UTC"}
# #     )
# #     data: Float64 = pa.Field(nullable=True)
# #     signal: Float64 = pa.Field(coerce=True)
# #     prev_signal: Float64 = pa.Field(coerce=True)
# #     trade: Float64 = pa.Field(coerce=True)
# #     pnl: Float64
# #     equity: Float64
#
#
# PortfolioWeights = dict[str, Decimal]
# PortfolioWeightAllocator = Callable[[dict[str, Portfolio]], PortfolioWeights]
#
#
# class MultiAssetPortfolio:
#     def __init__(
#         self,
#         id: str,
#         portfolios: list[Portfolio],
#         allocator: PortfolioWeightAllocator,
#         metrics: list[Metrics] = [Ratio(), Drawdown()],
#         start_time: datetime | None = None,
#         end_time: datetime | None = None,
#     ):
#         self.id = id
#         self.portfolios = {p.base_asset: p for p in portfolios}
#         self.alphas = {alpha.id: alpha for p in portfolios for alpha in p.alphas}
#         self.allocator = allocator
#         self.metrics = metrics
#         self.performances: dict[str, tuple[Performance, pl.DataFrame]] = {}
#         self.weights: PortfolioWeights = {}
#         self.start_time = start_time
#         self.end_time = end_time
#
#     def _validate_portfolio(self):
#         # Make sure all alphas are unique
#         all_ids = [id for p in self.portfolios.values() for id in p.alpha_ids()]
#         if len(all_ids) != len(set(all_ids)):
#             raise ValueError("All alphas must have unique ids")
#         logger.info(f"validation: all {len(all_ids)} alphas have unique ids ✅")
#
#         # Allocate weights if not already done
#         if len(self.weights) == 0:
#             self.weights = self.allocator(self.portfolios)
#
#         # Make sure weights sum to 1.0
#         if not math.isclose(sum(self.weights.values()), 1.0):
#             raise Exception("Weights must sum up to 1.0")
#         logger.info(
#             f"validation: all {len(self.portfolios)} portfolio weights sum up to 1.0 ✅"
#         )
#
#         # Make sure alpha performances are available
#         # TODO: This is not working right now due to breaking ADRS API
#         # if len(self.performances) == 0:
#         #     for base_asset, portfolio in self.portfolios.items():
#         #         self.performances[base_asset] = portfolio.backtest()
#         logger.info(
#             f"validation: all {len(all_ids)} alphas' backtest performances are ready ✅"
#         )
#
#     def backtest(
#         self,
#     ) -> tuple[
#         MultiAssetPortfolioPerformance, DataFrame[MultiAssetPortfolioPerformanceDF]
#     ]:
#         self._validate_portfolio()
#
#         # Combine the performances
#         trade_performances: dict[str, TradePerformance] = {}
#         merged_df: pl.DataFrame | None = None
#         for base_asset, (performance, df) in self.performances.items():
#             trade_performances[base_asset] = TradePerformance(
#                 largest_loss=performance.largest_loss,
#                 num_datapoints=performance.num_datapoints,
#                 num_trades=performance.num_trades,
#                 avg_holding_time_in_seconds=performance.avg_holding_time_in_seconds,
#                 long_trades=performance.long_trades,
#                 short_trades=performance.short_trades,
#                 win_trades=performance.win_trades,
#                 lose_trades=performance.lose_trades,
#                 win_streak=performance.win_streak,
#                 lose_streak=performance.lose_streak,
#                 win_rate=performance.win_rate,
#             )
#
#             weight = self.weights[base_asset]
#             pnl_col = f"{base_asset}_pnl"
#             signal_col = f"{base_asset}_signal"
#             query = [
#                 pl.col("start_time"),
#                 pl.col("price").alias(f"{base_asset}_price"),
#                 (pl.col("pnl") * weight).alias(pnl_col),  # NOTE: Use Decimal
#                 (pl.col("signal").cast(pl.Float64) * weight).alias(signal_col),
#             ]
#
#             if merged_df is None:
#                 merged_df = df.select(query)
#             else:
#                 merged_df = (
#                     merged_df.join(
#                         df.select(query),
#                         on="start_time",
#                         how="full",
#                     )
#                     .drop(["start_time_right"])
#                     .with_columns(
#                         pl.col(pnl_col).forward_fill(),
#                         pl.col(signal_col).forward_fill(),
#                     )
#                 )
#
#         merged_df = cast(pl.DataFrame, merged_df)
#         performance_df = merged_df.select(
#             pl.col("start_time"),
#             *[pl.col(f"{base_asset}_price") for base_asset in self.portfolios],
#             *[pl.col(f"{base_asset}_signal") for base_asset in self.portfolios],
#             pl.lit(None).alias("data").cast(pl.Float64),
#             pl.sum_horizontal(
#                 [pl.col(c) for c in merged_df.columns if c.endswith("_pnl")]
#             ).alias("pnl"),
#             pl.sum_horizontal(
#                 [pl.col(c) for c in merged_df.columns if c.endswith("_signal")]
#             ).alias("signal"),
#         ).with_columns(
#             pl.col("signal")
#             .shift(1)
#             .alias("prev_signal")
#             .forward_fill()
#             .fill_null(strategy="zero"),
#             pl.col("signal").diff().alias("trade").fill_null(strategy="zero"),
#             pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero"),
#         )
#
#         trade_performances["total"] = TradePerformance(
#             largest_loss=min(
#                 map(lambda t: t.largest_loss, trade_performances.values())
#             ),
#             num_datapoints=max(
#                 map(lambda t: t.num_datapoints, trade_performances.values())
#             ),
#             num_trades=sum(map(lambda t: t.num_trades, trade_performances.values())),
#             avg_holding_time_in_seconds=float(
#                 np.mean(
#                     list(
#                         map(
#                             lambda t: t.avg_holding_time_in_seconds,
#                             trade_performances.values(),
#                         )
#                     )
#                 )
#             ),
#             long_trades=sum(map(lambda t: t.long_trades, trade_performances.values())),
#             short_trades=sum(
#                 map(lambda t: t.short_trades, trade_performances.values())
#             ),
#             win_trades=sum(map(lambda t: t.win_trades, trade_performances.values())),
#             lose_trades=sum(map(lambda t: t.lose_trades, trade_performances.values())),
#             win_streak=min(map(lambda t: t.win_streak, trade_performances.values())),
#             lose_streak=max(map(lambda t: t.lose_streak, trade_performances.values())),
#             win_rate=sum(map(lambda t: t.win_trades, trade_performances.values()))
#             / sum(map(lambda t: t.num_trades, trade_performances.values())),
#         )
#         performance = {
#             "trades": trade_performances,
#             "start_time": self.start_time,
#             "end_time": self.end_time,
#             "metadata": {},
#         }
#         for metric in self.metrics:
#             result = metric.compute(performance_df)
#             performance = {**performance, **result}
#
#         return MultiAssetPortfolioPerformance.model_validate(
#             performance
#         ), MultiAssetPortfolioPerformanceDF.validate(performance_df)

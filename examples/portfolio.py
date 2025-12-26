# import json
# import asyncio
# import logging
# import polars as pl
# from typing import override
# from decimal import Decimal
# from datetime import datetime

# from cybotrade.logging import setup_logger

# from adrs.data import DataInfo, DataColumn, DataProcessor
# from adrs.report.portfolio import MultiAssetPortfolioReportV1
# from adrs import Alpha, DataLoader
# from adrs.performance import Evaluator
# from adrs.portfolio import (
#     Portfolio,
#     MultiAssetPortfolio,
#     AlphaPerformances,
#     AlphaWeights,
#     PortfolioWeights,
# )


# class CoinbaseBinancePremiumAlpha(Alpha):
#     def __init__(
#         self, window: int, long_entry_threshold: float, long_exit_threshold: float
#     ) -> None:
#         super().__init__(
#             id="coinbase_binance_premium_zscore",
#             data_infos=[
#                 DataInfo(
#                     topic="binance-spot|candle?symbol=BTCUSDT&interval=1h",
#                     columns=[DataColumn(src="close", dst="close_binance_spot")],
#                     lookback_size=window,
#                 ),
#                 DataInfo(
#                     topic="coinbase|candle?symbol=BTCUSD&interval=1h",
#                     columns=[DataColumn(src="close", dst="close_coinbase")],
#                     lookback_size=window,
#                 ),
#             ],
#         )
#         self.window = window
#         self.long_entry_threshold = long_entry_threshold
#         self.long_exit_threshold = long_exit_threshold

#     @override
#     def next(self, data_df: pl.DataFrame):
#         # alpha formula
#         df = data_df.select(
#             pl.col("start_time"),
#             (pl.col("close_coinbase") - pl.col("close_binance_spot")).alias("data"),
#         )

#         # pre-process
#         df = df

#         # modeling
#         df = df.with_columns(
#             (
#                 (pl.col("data") - pl.col("data").rolling_mean(self.window))
#                 / pl.col("data").rolling_std(self.window, ddof=1)
#             ).alias("zscore")
#         ).filter(pl.col("zscore").is_finite())

#         # signal
#         df = df.with_columns(
#             pl.when(pl.col("zscore") >= self.long_entry_threshold)
#             .then(1)
#             .when(pl.col("zscore") <= self.long_exit_threshold)
#             .then(0)
#             .otherwise(None)
#             .forward_fill()
#             .fill_null(strategy="zero")
#             .alias("signal")
#         )

#         return df


# # BTC_ALPHA_CONFIG = make_alpha_config(
# #     dataloader=DataLoader(
# #         data_dir="outdir",
# #         credentials=json.load(open("credentials.json")),
# #     ),
# #     environment=Environment.BACKTEST,
# #     base_asset="BTC",
# #     start_time=datetime.fromisoformat("2020-06-01T00:00:00Z"),
# #     end_time=datetime.fromisoformat("2025-07-01T00:00:00Z"),
# # )
# # ETH_ALPHA_CONFIG = make_alpha_config(
# #     dataloader=DataLoader(
# #         data_dir="outdir",
# #         credentials=json.load(open("credentials.json")),
# #     ),
# #     environment=Environment.BACKTEST,
# #     base_asset="ETH",
# #     start_time=datetime.fromisoformat("2020-06-01T00:00:00Z"),
# #     end_time=datetime.fromisoformat("2025-07-01T00:00:00Z"),
# # )
# # btc_evaluator = Evaluator(fees=0.035, candle_shift=2)
# # eth_evaluator = Evaluator(fees=0.035, candle_shift=2)
# # alphas: list[Alpha] = [
# #     ZScoreLong(),
# #     ZScoreShort(),
# # ]


# def mean_allocator(performances: AlphaPerformances) -> AlphaWeights:
#     n = len(performances)
#     if n == 0:
#         return {}
#     weight = Decimal("1.0") / n
#     return {alpha_id: weight for alpha_id in performances.keys()}


# def mean_portfolio_allocator(portfolios: dict[str, Portfolio]) -> PortfolioWeights:
#     for base_asset, portfolio in portfolios.items():
#         perf, df = portfolio.backtest()

#     return {"BTC": Decimal("0.8"), "ETH": Decimal("0.2")}


# async def main():
#     setup_logger(log_level=logging.INFO)

#     eth_alphas: list[Alpha] = [
#         ZScoreLongETH(),
#         ZScoreShortETH(),
#     ]

#     for alpha in alphas:
#         await alpha.init(BTC_ALPHA_CONFIG, btc_evaluator)
#         alpha.backtest()

#     for alpha in eth_alphas:
#         await alpha.init(ETH_ALPHA_CONFIG, eth_evaluator)
#         alpha.backtest()

#     start_time, end_time = (
#         datetime.fromisoformat("2020-06-01T00:00:00Z"),
#         datetime.fromisoformat("2024-07-01T00:00:00Z"),
#     )

#     portfolio = MultiAssetPortfolio(
#         id="TEST",
#         portfolios=[
#             Portfolio(
#                 id="ETH_TEST",
#                 alphas=eth_alphas,
#                 allocator=mean_allocator,
#                 start_time=start_time,
#                 end_time=end_time,
#             ),
#             Portfolio(
#                 id="BTC_TEST",
#                 alphas=alphas,
#                 allocator=mean_allocator,
#                 start_time=start_time,
#                 end_time=end_time,
#             ),
#         ],
#         allocator=mean_portfolio_allocator,
#     )
#     report = MultiAssetPortfolioReportV1.compute(
#         portfolio=portfolio,
#         B_start=start_time,
#         B_end=end_time,
#         F_start=datetime.fromisoformat("2024-07-01T00:00:00Z"),
#         F_end=datetime.fromisoformat("2025-07-01T00:00:00Z"),
#     )
#     logging.info(report.back.performance_df.columns)

#     # with open("multi_asset_portfolio_v1_report.parquet", "wb") as f:
#     #     f.write(report.serialize())


# if __name__ == "__main__":
#     asyncio.run(main())

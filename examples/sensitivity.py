import json
import asyncio
import logging
import polars as pl
from datetime import datetime
from typing import override, cast

from adrs.data import DataInfo, DataColumn, UniformDataProcessor
from adrs import Alpha, AlphaConfig, Environment, DataLoader
from adrs.tests import Sensitivity, SensitivityParameter
from adrs.utils import backforward_split
from adrs.performance import Evaluator
from adrs.report import AlphaReportV1
from adrs.signal import Long, Signal
from adrs.model import ZScore

from cybotrade.logging import setup_logger


class MockAlpha(Alpha):
    def __init__(
        self,
        window: int,
        long_entry_thres: float,
        long_exit_thres: float,
    ):
        super().__init__()
        self.window = window
        self.long_entry_thres = long_entry_thres
        self.long_exit_thres = long_exit_thres

    @staticmethod
    def id():
        return cast(str, "TEST_SENSITIVITY")

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


async def main():
    setup_logger(log_level=logging.INFO)

    rolling_window = 40
    start_time, end_time = (
        datetime.fromisoformat("2020-05-11T00:00:00Z"),
        datetime.fromisoformat("2025-01-01T00:00:00Z"),
    )
    fees = 0.035

    dataloader = DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
    )
    config = AlphaConfig(
        base_asset="BTC",
        data_infos=[
            DataInfo(
                topic="binance-spot|candle?symbol=BTCUSDT&interval=1h",
                columns=[DataColumn(src="close", dst="close_binance_spot")],
                lookback_size=rolling_window,
            ),
            DataInfo(
                topic="coinbase|candle?symbol=BTCUSD&interval=1h",
                columns=[DataColumn(src="close", dst="close_coinbase")],
                lookback_size=rolling_window,
            ),
        ],
        dataloader=dataloader,
        data_processor=UniformDataProcessor(),
        start_time=start_time,
        end_time=end_time,
        environment=Environment.BACKTEST,
    )
    evaluator = Evaluator(fees=fees, candle_shift=10)

    alpha = MockAlpha(
        window=rolling_window,
        long_entry_thres=0.825,
        long_exit_thres=-0.825,
    )
    await alpha.init(config=config, evaluator=evaluator)

    B_start, B_end, F_start, F_end = backforward_split(
        start_time=start_time, end_time=end_time, size=(0.7, 0.3)
    )
    sensitivity = Sensitivity(
        alpha=alpha,
        parameters={
            "window": SensitivityParameter(min_val=10, min_gap=25),
            "long_entry_thres": SensitivityParameter(min_val=0.1),
        },
        gap_percent=0.15,
    )

    report = AlphaReportV1.compute(alpha, B_start, B_end, F_start, F_end, sensitivity)
    print(f"The report is {len(report.serialize()) / 1_000_000}mb")
    print(list(map(lambda a: (a[0], a[1].sharpe_ratio), report.back.sensitivity)))
    print(report.back.performance.sharpe_ratio)
    print(report.back.sensitivity_sr_summary)


if __name__ == "__main__":
    asyncio.run(main())

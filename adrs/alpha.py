import polars as pl
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, TypedDict, Unpack, NotRequired

from adrs.types import Performance
from adrs.performance import Evaluator
from adrs.data import Datamap, DataInfo, DataProcessor
from adrs.performance.metric import Ratio, Drawdown, Trade


class AlphaBacktestArgs(TypedDict):
    evaluator: Evaluator
    base_asset: str
    datamap: Datamap
    data_df: pl.DataFrame
    start_time: datetime
    end_time: datetime
    fees: float
    interval: str | timedelta
    price_shift: NotRequired[int]
    output_columns: NotRequired[list[pl.Expr]]


class Alpha:
    def __init__(
        self,
        id: str,
        data_infos: list[DataInfo],
        data_processor: DataProcessor = DataProcessor(),
    ):
        self.id = id
        self.data_infos = data_infos
        self.data_processor = data_processor
        self.data_processor.data_infos = data_infos

    @abstractmethod
    def next(self, data_df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError("Every alpha must implement its `next` method")

    def backtest(
        self, **kwargs: Unpack[AlphaBacktestArgs]
    ) -> tuple[Performance, pl.DataFrame]:
        evaluator = kwargs["evaluator"]
        base_asset = kwargs["base_asset"]
        datamap = kwargs["datamap"]
        data_df = kwargs["data_df"]
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]
        fees = kwargs["fees"]
        interval = kwargs["interval"]
        price_shift = kwargs.get("price_shift", 0)
        output_columns = kwargs.get("output_columns", [pl.all()])

        df = self.next(data_df)

        # Verify that the signal is valid
        if "signal" not in df.schema.keys():
            raise ValueError(
                "`next()` method must return a DataFrame with the column 'signal'"
            )
        if not df.schema["signal"].is_numeric():
            raise ValueError(
                "DataFrame returned from `next()` must have a 'signal' column that is numeric"
            )
        if (
            df.select("signal").min().to_numpy().ravel()[0] < -1
            or df.select("signal").max().to_numpy().ravel()[0] > 1
        ):
            raise ValueError(
                "DataFrame returned from `next()` must have a 'signal' column that is between [-1, 1]"
            )

        pdf = evaluator.eval(
            signal_lf=df.lazy(),
            base_asset=base_asset,
            datamap=datamap,
            start_time=start_time,
            end_time=end_time,
            fees=fees,
            interval=interval,
            price_shift=price_shift,
            output_columns=output_columns,
        ).collect(engine="in-memory")

        # Compute the metrics
        performance: dict[str, Any] = {
            "start_time": start_time,
            "end_time": end_time,
            "metadata": {},
        }
        for metric in [Ratio(), Trade(), Drawdown()]:
            result = metric.compute(pdf)
            performance = {**performance, **result}

        return Performance.model_validate(performance), pdf

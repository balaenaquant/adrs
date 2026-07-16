import warnings
import polars as pl
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, TypedDict, Unpack, NotRequired

from adrs.types import Performance
from adrs.utils import infer_interval
from adrs.performance import Evaluator
from adrs.data import Datamap, DataInfo, DataProcessor
from adrs.performance.metric import Ratio, Drawdown, Trade


class AlphaBacktestArgs(TypedDict):
    evaluator: Evaluator
    base_asset: str
    datamap: Datamap
    start_time: datetime
    end_time: datetime
    # percent of notional per unit of turnover: 0.035 charges 3.5 bps per
    # full position flip leg
    fees: float
    data_df: NotRequired[pl.DataFrame]
    # execution delay after the bar close (fills at the raw price observed at
    # bar close + delay)
    execution_delay: NotRequired[timedelta]
    # deprecated alias of execution_delay, in minutes
    price_shift: NotRequired[int]
    output_columns: NotRequired[list[pl.Expr]]
    # skip metric computation (returns None for Performance) — useful when
    # only the signal/pnl frame is needed, e.g. in search loops
    compute_metrics: NotRequired[bool]


class Alpha:
    def __init__(
        self,
        id: str,
        data_infos: list[DataInfo],
        data_processor: DataProcessor,
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
    ) -> tuple[Performance | None, pl.DataFrame]:
        evaluator = kwargs["evaluator"]
        base_asset = kwargs["base_asset"]
        datamap = kwargs["datamap"]
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]
        fees = kwargs["fees"]
        # don't use .get's default — it would run the (potentially expensive)
        # processor even when data_df is supplied
        data_df = kwargs.get("data_df")
        if data_df is None:
            data_df = self.data_processor.process(datamap)
        price_shift = kwargs.get("price_shift")
        execution_delay = kwargs.get("execution_delay")
        if price_shift is not None and execution_delay is not None:
            raise ValueError(
                "pass either execution_delay or the deprecated price_shift, not both"
            )
        if price_shift is not None:
            warnings.warn(
                "price_shift is deprecated, use execution_delay "
                "(a timedelta: the execution delay after bar close)",
                DeprecationWarning,
                stacklevel=2,
            )
            execution_delay = timedelta(minutes=price_shift)
        output_columns = kwargs.get("output_columns", [pl.all()])
        compute_metrics = kwargs.get("compute_metrics", True)

        if data_df is None:
            raise ValueError("data_df received is None")

        df = self.next(data_df)
        interval = infer_interval(df["start_time"])

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
            execution_delay=execution_delay or timedelta(0),
            output_columns=output_columns,
        ).collect(engine="in-memory")

        if not compute_metrics:
            return None, pdf

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

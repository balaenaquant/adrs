import io
import math
import pickle
import logging
import polars as pl
from typing import cast, Any, Self
from datetime import datetime, timedelta
from pandera.typing.polars import DataFrame

from flow import DataLoader

from cybotrade import Topic

from adrs.types import SignalDF, PriceDF, PerformanceDF, Performance

from .metric import Metrics, Ratio, Trade, Drawdown


class Evaluator:
    prices_df: DataFrame[PriceDF] | None = None
    metrics: list[Metrics] = []
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        fees: float,
        candle_shift: int = 0,
        metrics: list[Metrics] = [Ratio(), Trade(), Drawdown()],
    ) -> None:
        self.fees = fees
        self.candle_shift = candle_shift
        self.metrics = metrics

    async def init(
        self,
        dataloader: DataLoader,
        candle_topic: str,
        start_time: datetime,
        end_time: datetime,
    ):
        # find previous loaded prices (no need to load if we have the data already loaded)
        prev_candle_topic, prices_df_start_time, prices_df_end_time = (
            self.candle_topic if hasattr(self, "candle_topic") else None,
            self.prices_df.row(0, named=True)["start_time"]
            if self.prices_df is not None
            else None,
            self.prices_df.row(-1, named=True)["start_time"]
            if self.prices_df is not None
            else None,
        )

        self.dataloader = dataloader
        self.candle_topic = Topic.from_str(candle_topic)
        self.start_time, self.end_time = start_time, end_time

        candle_interval = self.candle_topic.interval()
        if candle_interval is None:
            raise Exception(f"Topic {candle_topic} does not have an interval")

        # NOTE: need more data due to candles being shifted
        offset = math.ceil(candle_interval / timedelta(days=1)) * timedelta(days=1)
        self.prices_end_time = end_time if self.candle_shift == 0 else end_time + offset

        # skip if that data range is covered
        prices_df_start_time = (
            prices_df_start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            if prices_df_start_time is not None
            else prices_df_start_time
        )
        prices_df_end_time = (
            (prices_df_end_time + (self.candle_shift + 1) * candle_interval).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if prices_df_end_time is not None
            else prices_df_end_time
        )
        logging.debug(
            f"prev_candle_topic: {prev_candle_topic}, self.candle_topic: {self.candle_topic}"
        )
        logging.debug(
            f"prices_df_start_time: {prices_df_start_time}, self.start_time: {self.start_time}"
        )
        logging.debug(
            f"prices_df_end_time: {prices_df_end_time}, self.end_time: {self.end_time}"
        )
        if (
            prev_candle_topic is not None
            and prices_df_start_time is not None
            and prices_df_end_time is not None
            and prev_candle_topic == self.candle_topic
            and prices_df_start_time <= self.start_time
            and prices_df_end_time >= self.end_time
        ):
            return

        await self.load_prices()

    async def load_prices(self):
        logging.info(
            f"Loading {self.candle_topic} from {self.start_time} to {self.end_time}"
        )
        df = await self.dataloader.load(
            topic=str(self.candle_topic),
            start_time=self.start_time,
            end_time=self.prices_end_time,
        )
        prices_df = (
            df.with_columns(pl.col("start_time").dt.replace_time_zone(time_zone="UTC"))
            .select(
                pl.col("start_time"),
                pl.col("close").alias("price").shift(-self.candle_shift),
            )
            .drop_nulls()
        )
        self.prices_df = PriceDF.validate(prices_df)
        logging.info(f"Loaded {len(prices_df)} candles for {self.candle_topic}")

    def eval(
        self, signal_df: DataFrame[SignalDF]
    ) -> tuple[Performance, DataFrame[PerformanceDF]]:
        SignalDF.validate(signal_df)

        if self.prices_df is None:
            raise Exception("Prices are not loaded yet. Call init() first.")

        # determine the interval of signals
        interval = signal_df["start_time"].diff().last()
        if not isinstance(interval, timedelta):
            raise Exception("signal_df does not have an interval in between data")

        # calculate the pnl of each interval
        df = (
            cast(pl.DataFrame, self.prices_df)
            .group_by_dynamic(index_column="start_time", every=interval)
            .agg(pl.col("price").last())
            .join(signal_df, how="left", left_on="start_time", right_on="start_time")
            .filter(
                pl.col("start_time").is_between(
                    self.start_time, self.end_time, closed="left"
                )
            )
            .with_columns(
                pl.col("signal").forward_fill().fill_null(strategy="zero"),
                pl.col("signal")
                .shift(1)
                .alias("prev_signal")
                .forward_fill()
                .fill_null(strategy="zero"),
                pl.col("price")
                .pct_change()
                .alias("returns")
                .fill_null(strategy="zero"),
            )
            .with_columns(
                pl.col("signal").diff().alias("trade").fill_null(strategy="zero")
            )
            .with_columns(
                (
                    pl.col("prev_signal") * pl.col("returns")
                    - pl.col("trade").abs() * self.fees / 100
                )
                .alias("pnl")
                .fill_null(strategy="zero"),
            )
            .with_columns(
                pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero")
            )
        )
        df = PerformanceDF.validate(df)

        # Compute the metrics
        performance: dict[str, Any] = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": {},
        }
        for metric in self.metrics:
            result = metric.compute(df)
            performance = {**performance, **result}

        return Performance.model_validate(performance), df

    def write_ipc(self) -> bytes:
        """Serialize Evaluator instance to bytes."""
        payload: dict[str, bytes | None] = {}

        # serialize the Polars DataFrame separately
        if self.prices_df is not None:
            buf = io.BytesIO()
            self.prices_df.write_ipc(buf)
            payload["prices_df"] = buf.getvalue()
        else:
            payload["prices_df"] = None

        # serialize the rest of the Python objects
        py_objects = {
            "fees": self.fees,
            "candle_shift": self.candle_shift,
            "candle_topic": self.candle_topic,
            "metrics": self.metrics,
            "dataloader": self.dataloader,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        payload["py_objects"] = pickle.dumps(py_objects)

        return pickle.dumps(payload)  # top-level payload

    @classmethod
    def read_ipc(cls, raw: bytes) -> Self:
        """Deserialize bytes into an Evaluator instance."""
        payload = pickle.loads(raw)

        # restore Python objects
        py_objects = pickle.loads(payload["py_objects"])
        evaluator = cls(
            fees=py_objects["fees"],
            candle_shift=py_objects["candle_shift"],
            metrics=py_objects["metrics"],
        )
        evaluator.candle_topic = py_objects["candle_topic"]
        evaluator.dataloader = py_objects["dataloader"]
        evaluator.start_time = py_objects["start_time"]
        evaluator.end_time = py_objects["end_time"]

        # restore Polars DataFrame
        if payload["prices_df"] is not None:
            buf = io.BytesIO(payload["prices_df"])
            evaluator.prices_df = PriceDF.validate(pl.read_ipc(buf))
        else:
            evaluator.prices_df = None

        return evaluator

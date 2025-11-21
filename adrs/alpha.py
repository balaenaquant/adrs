import inspect
import logging
import polars as pl
from enum import Enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import cast, override, Any
from pandera.typing.polars import DataFrame
from datetime import datetime, timezone, timedelta

from flow import DataLoader

from cybotrade import Topic
from cybotrade.io import Event, EventType
from cybotrade.strategy import BaseStrategy

from cybotrade_datasource import query_paginated, Data

from adrs.http import HttpClient
from adrs.data import (
    Datamap,
    SimulationDataEvent,
    DataProcessor,
    DataInfo,
)
from adrs.data.event import NoopDataEvent
from adrs.signal import Signal
from adrs.performance import Evaluator
from adrs.types import SignalDF, PerformanceDF, Performance

logger = logging.getLogger(__name__)


def derive_candle_topic(base_asset: str) -> str:
    return f"{'bybit-linear' if base_asset == 'BTC' else 'binance-linear'}|candle?symbol={base_asset}USDT&interval=1m"


class AlphaKind(str, Enum):
    """
    Kind of the alpha.
    """

    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class Environment(str, Enum):
    """
    Environment for the alpha.
    """

    BACKTEST = "backtest"
    SIMULATION = "simulation"
    LIVE = "live"


@dataclass
class AlphaConfig:
    base_asset: str
    data_infos: list[DataInfo]
    data_processor: DataProcessor
    environment: Environment

    # for backtest
    start_time: datetime
    end_time: datetime
    dataloader: DataLoader | None = None
    backtest_should_lookback: bool = False

    # for live
    datasource_api_key: str | None = None
    http_client: HttpClient | None = None


class Alpha(BaseStrategy):
    # The configuration for the alpha.
    config: AlphaConfig

    # Kind of the alpha, whether it trade long-only, short-only or both.
    kind: AlphaKind

    # To hold additional information about the alpha
    metadata: dict[str, Any] = {}

    # The latest time that the alphas has ever seen.
    # In live environment, this is actual time whereas in simulation environment
    # it is the current_time() of last received data.
    timestamp: datetime

    # The evaluator which holds price data to evaluate the performance of this alpha.
    evaluator: Evaluator

    # A custom implementation of datamap that is different from Cybotrade's implementation
    datamap: Datamap  # type: ignore

    def __init__(self):
        # The data that the alpha is working on.
        self.df: pl.DataFrame | None = None

        # Performance (only available in BACKTEST and SIMULTATION)
        self.performance: Performance | None = None
        self.performance_df: DataFrame[PerformanceDF] | None = None

        # HTTP client (only available in LIVE)
        self.http: HttpClient | None = None

        # Current signal (default to NONE)
        self.signal: Signal = Signal.NONE

    async def init(
        self,
        config: AlphaConfig,
        evaluator: Evaluator | None = None,
        kind: AlphaKind = AlphaKind.BOTH,
        datamap: Datamap | None = None,
    ):
        topics = list(map(lambda info: Topic.from_str(info.topic), config.data_infos))

        # Make sure there's no duplicated topics
        if len(topics) != len(set(topics)):
            logging.error(
                f"[{self.id()}] config.data_infos has duplicated info: {config.data_infos}"
            )
            raise Exception("There is duplicated Topic in `data_infos`")

        # Setup strategy stuff if it is not BACKTEST
        if config.environment != Environment.BACKTEST:
            super().__init__()

        self.kind = kind if getattr(self, "kind", None) is None else self.kind
        self.config = config
        self.datasource_topics = topics  # type: ignore
        self.timestamp = (
            config.start_time
            if config.start_time is not None
            else datetime.now(tz=timezone.utc)
        )
        if datamap is None:
            self.datamap = Datamap(data_infos=config.data_infos)  # type: ignore
        else:
            self.datamap = datamap  # type: ignore
        self.data_processor = config.data_processor
        self.data_processor.set_data_infos(config.data_infos)
        if evaluator is not None:
            if (
                config.dataloader is None
                or config.start_time is None
                or config.end_time is None
            ):
                raise ValueError(
                    "dataloader, start_time and end_time must be provided when using an evaluator"
                )
            self.evaluator = evaluator
            await self.evaluator.init(
                dataloader=config.dataloader,
                candle_topic=derive_candle_topic(base_asset=config.base_asset),
                start_time=config.start_time,
                end_time=config.end_time,
            )

        # load datamap if is backtest
        if config.environment == Environment.BACKTEST:
            await self.init_datamap_backtest()
        if config.environment == Environment.LIVE:
            if config.datasource_api_key is None:
                raise Exception("'datasource_api_key' is required for live")
            self.datasource_api_key = config.datasource_api_key

            if config.http_client is None:
                raise Exception("'http_client' is required for live")
            self.http = config.http_client

            self.lookback_size = max(
                self.config.data_infos, key=lambda x: x.lookback_size
            ).lookback_size
            if self.lookback_size <= 0:
                raise Exception("Lookback size must be a positive integer")

    def update_start_end_time(self, start_time: datetime, end_time: datetime):
        self.config.start_time, self.config.end_time = start_time, end_time
        self.evaluator.start_time, self.evaluator.end_time = start_time, end_time

    @staticmethod
    @abstractmethod
    def id() -> str:
        """A name or identifier for the alpha."""
        raise NotImplementedError("All alpha should have an identifier.")

    @staticmethod
    @abstractmethod
    def parameters_description() -> dict[str, str]:
        """Parameters for the alpha."""
        raise NotImplementedError("All alpha should specify its parameters.")

    @abstractmethod
    async def on_datasource_event(self, topic: Topic, data: Data):
        # Ready when all number of data points in the datamap reaches the lookback size.
        info = next(
            info
            for info in self.config.data_infos
            if Topic.from_str(info.topic) == topic
        )
        await self.resync(info)
        self.datamap.update(info=info, data=data)
        if not self.datamap.is_ready():
            logger.info(
                f"[COLLECT_DATA] 🔃 {topic} | {len(self.datamap[info])}/{info.lookback_size}"
            )
            return

        # Check if we are getting delayed data (only topic that has an interval will be checked,
        # tick data will result in `last_closed_time=None` hence always valid).
        # This check will always be correct in Environment.SIMULATION
        start_time = data["start_time"].replace(tzinfo=timezone.utc)
        last_closed_time = self.last_closed_time(topic, start_time)
        if last_closed_time != start_time:
            msg = f"[WRONG_DATA] ❌ Last closed time {last_closed_time} does not match current time {start_time} for topic {topic}"
            logger.warning(
                msg
            ) if self.config.environment == Environment.LIVE else logger.error(msg)
            return
        else:
            logger.info(
                f"[CORRECT_DATA] ✅ Last closed time {last_closed_time} matches current time {start_time} for topic {topic}"
            )

        # Process the data (users can override this so they can do custom logic here)
        df = self.data_processor.process(
            datamap=self.datamap,
            last_closed_time=last_closed_time,
        )
        if df is None:
            return

        # Inference for the next signal
        signal, df = self.next(df)

        # NOTE: The returned type from `next` could be a `np.int8` not an instance of the `Signal` enum, hence
        #       explicitly instatiate it here.
        signal = Signal(signal)

        # Update df
        self.df = df if self.df is None else self.df.extend(df)

        if self.config.environment == Environment.LIVE:  # in LIVE environment
            if len(self.df) >= len(df):  # preserve up to lookback size
                self.df = self.df[1:]

        match self.config.environment:
            case Environment.BACKTEST:
                raise Exception("Alpha in backtest environment should not reach here.")
            case Environment.SIMULATION | Environment.LIVE:
                self.logger.info(f"[SIGNAL] {signal}")
                await self.on_signal(signal)  # handle the signal

    @abstractmethod
    async def on_signal(self, signal: Signal):
        if self.http is None:
            return

        response = await self.http.post(
            json={"id": self.id(), "side": signal.to_order_side()}
        )
        self.logger.info(f"[SERVER RESPONSE] {response}")

    @abstractmethod
    def next(self, datas_df: pl.DataFrame) -> tuple[Signal, pl.DataFrame]:
        raise NotImplementedError("Alpha does not implement next()")

    @abstractmethod
    async def on_end(self):
        self.on_shutdown()

    @override
    async def on_event(self, event: Event):
        match cast(str, event.event_type):
            case EventType.DatasourceUpdate:
                topic = Topic.from_str(str(event.data["topic"]))
                for d in event.data["data"]:
                    await self.on_datasource_event(topic=topic, data=d)
            case EventType.Subscribed:
                if not event.data["success"]:
                    raise Exception(event.data["message"])
                self.logger.info(event.data["message"])
            case "end":
                await self.on_end()
            case _:
                self.logger.error("unknown event")

    @override
    def on_init_datamap(self, topic: Topic, data: list[Data]):
        info = next(
            info
            for info in self.config.data_infos
            if Topic.from_str(info.topic) == topic
        )
        self.datamap.update_df(info, pl.DataFrame(data))

    async def init_datamap_backtest(self):
        env = self.config.environment

        if env != Environment.BACKTEST:
            raise Exception(
                f"init_datamap_backtest is only valid for environment {env}"
            )

        if self.config.dataloader is None:
            raise ValueError(f"'dataloader' must be provided for {env}")
        if self.config.start_time is None or self.config.end_time is None:
            raise ValueError(f"'start_time' and 'end_time' must be provided for {env}")
        if self.datasource_topics is None:
            raise ValueError("'self.datasource_topics' cannot be None")

        for topic in self.datasource_topics:
            info = next(
                info
                for info in self.config.data_infos
                if Topic.from_str(info.topic) == topic
            )
            interval = topic.interval()
            if interval is None:
                raise Exception(f"Topic {topic} does not have an interval")

            start_time = (
                self.config.start_time - interval * info.lookback_size
                if self.config.backtest_should_lookback
                else self.config.start_time
            ).replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = self.config.end_time

            # Skip if have already loaded before
            if (
                info in self.datamap.map
                and self.datamap[info].row(0, named=True)["start_time"] <= start_time
                and self.datamap[info].row(-1, named=True)["start_time"]
                >= topic.last_closed_time_relative(
                    self.config.end_time, is_collect=False
                )
            ):
                continue

            # Explicitly override today's data because it might be incomplete
            if end_time.date() == datetime.now(tz=timezone.utc).date():
                end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info(
                    f"Loading data for topic {topic} from {start_time} to {end_time}"
                )
                df = await self.config.dataloader.load(
                    topic=str(topic),
                    start_time=start_time,
                    end_time=end_time,
                )
                today_df = await self.config.dataloader.load(
                    topic=str(topic),
                    start_time=end_time,
                    end_time=end_time + timedelta(days=1),
                    override_existing=True,
                )
                df = df.extend(today_df)
            else:
                logger.info(
                    f"Loading data for topic {topic} from {start_time} to {end_time}"
                )
                df = await self.config.dataloader.load(
                    topic=str(topic),
                    start_time=start_time,
                    end_time=end_time,
                )
            self.datamap.update_df(info=info, df=df)
            logger.info(f"Loaded {len(df)} datapoints for topic {topic}")

    async def resync(self, info: DataInfo):
        """Resync to get the latest data"""
        if self.datasource_api_key is None:
            raise Exception("datasource_api_key should not be None here")
        data = await query_paginated(
            self.datasource_api_key, str(info.topic), limit=info.lookback_size
        )
        self.datamap.update_df(info, DataFrame(data))
        logger.info(
            f"Data for {info.topic} with {len(self.datamap[info])} data points has been resync"
        )

    def evaluate(self):
        """Evaluate the performance of the alpha. This method should be ran after backtest has concluded."""
        if self.evaluator is None:
            raise Exception("self.evaluator should not be None")
        if self.df is None:
            raise Exception("self.df should not be None")
        self.performance, self.performance_df = self.evaluator.eval(
            signal_df=SignalDF.validate(self.df)
        )

        # inject more information into the metadata
        self.performance.metadata = {
            **self.performance.metadata,
            **self.metadata,
            "kind": self.kind,
            "params": {
                name: getattr(self, name)
                for name in inspect.signature(self.__init__).parameters.keys()
            },
        }

        return self.performance, self.performance_df

    def current_time(self, topic: Topic, start_time: datetime) -> datetime:
        """Get the current time based on the environment."""
        match self.config.environment:
            case Environment.BACKTEST | Environment.SIMULATION:
                topic_interval = topic.interval()
                interval = topic_interval if topic_interval is not None else timedelta()
                # In a simulated environment, we only have the data's start time as a solid baseline.
                # Hence, we assume the current time to be the moment when we receive the data.
                current_time = start_time + interval
            case Environment.LIVE:
                # In a live environment, you would typically use the actual current time.
                current_time = datetime.now(tz=timezone.utc)
            case env:
                raise ValueError(f"Unsupported environment: {env}")

        self.timestamp = max(self.timestamp, current_time)  # update timestamp

        return current_time

    def last_closed_time(self, topic: Topic, start_time: datetime) -> datetime | None:
        """Get the last closed time based on the environment."""
        return topic.last_closed_time_relative(
            timestamp=self.current_time(topic, start_time),
            is_collect=True if self.config.environment == Environment.LIVE else False,
        )

    def backtest(self):
        # process the data
        datas_df = self.data_processor.process(
            datamap=self.datamap,
            last_closed_time=None,
        )
        if datas_df is None:
            raise Exception("datas_df is None after process in backtest")

        # filter the data by start_time and end_time
        datas_df = datas_df.filter(
            (pl.col("start_time") >= self.config.start_time)
            & (pl.col("start_time") < self.config.end_time)
        )

        # runs the logic (signal generation)
        self.signal, self.df = self.next(datas_df)

    async def run(
        self,
    ):
        if self.datasource_topics is None:
            raise ValueError("'self.datasource_topics' cannot be None")

        env = self.config.environment

        match env:
            case Environment.BACKTEST | Environment.SIMULATION:
                if self.config.dataloader is None:
                    raise ValueError(f"'dataloader' must be provided for {env}")
                if self.config.start_time is None or self.config.end_time is None:
                    raise ValueError(
                        f"'start_time' and 'end_time' must be provided for {env}"
                    )
                if self.evaluator is not None:
                    await self.evaluator.init(
                        dataloader=self.config.dataloader,
                        candle_topic=derive_candle_topic(
                            base_asset=self.config.base_asset
                        ),
                        start_time=self.config.start_time,
                        end_time=self.config.end_time,
                    )

                if env == Environment.BACKTEST:
                    self.backtest()
                else:
                    await self.start(
                        events=SimulationDataEvent(
                            dataloader=self.config.dataloader,
                            topics=self.datasource_topics,
                            start_time=self.config.start_time,
                            end_time=self.config.end_time,
                        )
                    )

                # ends the alpha (performance are computed here)
                await self.on_end()
            case Environment.LIVE:
                # For live we don't need to do anything since Cybotrade handles the
                # event handling with upstream Cybotrade Datasource.
                await self.start(events=NoopDataEvent())

    def get_config(self):
        return self.config

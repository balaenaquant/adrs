import time
import json
import signal
import asyncio
import logging
import traceback
import numpy as np

from nats_client import Msg
from functools import reduce
from typing import TypedDict, cast
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP

from aion import Scheduler, Trigger

from adrs.alpha import Alpha
from adrs.logging import setup_logger
from adrs.io.event import Event, EventType
from adrs.execution.portfolio import Portfolio
from adrs.types import Topic, CollectedData, Data
from adrs.data import (
    MetricStream,
    Datamap,
    DataInfo,
    DataLoader,
    DatasourceStream,
    MetricBuilder,
)


logger = logging.getLogger(__name__)


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


def acq_info(topic: str, infos: list[DataInfo]) -> list[DataInfo]:
    return cast(
        list[DataInfo],
        list(
            filter(
                lambda di: Topic.from_str(di.topic) == Topic.from_str(topic),
                infos,
            )
        ),
    )


def flatten_json(value, _prefix: str = "") -> dict:
    flattened_map = {}

    if not isinstance(value, dict):
        return flattened_map

    for key, val in value.items():
        if isinstance(val, dict):
            nested_map = flatten_json(val)
            for nested_key, nested_val in nested_map.items():
                if key == "o":
                    flattened_map[nested_key] = nested_val
                else:
                    flattened_map[f"{key}_{nested_key}"] = nested_val
        else:
            flattened_map[key] = val

    return flattened_map


class Signal(TypedDict):
    signal: Decimal
    weight: Decimal


class PortfolioExecutor:
    def __init__(
        self,
        portfolio: Portfolio,
        metric_stream: MetricStream,
        aggregate_window: Trigger = Trigger.Cron("*/5 * * * *"),
        max_signal_age: timedelta = timedelta(hours=2),
    ):
        self.portfolio = portfolio
        self.metric_builder = MetricBuilder(metric_stream)
        self.aggregate_window = aggregate_window
        self.scheduler = Scheduler()
        self.max_signal_age: timedelta = max_signal_age

        last_signal_time = self.portfolio.signal_df["start_time"][-1]
        if isinstance(last_signal_time, datetime):
            last_signal_time = (
                last_signal_time.replace(tzinfo=timezone.utc)
                if last_signal_time.tzinfo is None
                else last_signal_time
            )
            age = datetime.now(tz=timezone.utc) - last_signal_time
            if age > max_signal_age:
                raise Exception(
                    f"signal_df is stale: last row is {age} old (>{max_signal_age}). "
                    "Ensure signal_df covers current time"
                )

        setup_logger()

        def handle_signal(signum, _):
            match signum:
                case signal.SIGINT | signal.SIGTERM:
                    self.on_shutdown()
                    exit(0)
                case _:
                    logger.warning(f"Handling signal ({signum}) by doing nothing.")

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def on_shutdown(self):
        logger.warning("[on_shutdown] received SIGINT / SIGTERM, shutdown process")

    async def on_signal(self, msg: Msg):
        alpha_id = msg.subject.split(".")[1]
        payload = json.loads(msg.data.decode())
        signal = int(float(payload["signal"]))

        self.portfolio.update_signal(alpha_id, signal)

        logger.info(f"[on_signal] received signal from {alpha_id} signal: {signal}")

    async def on_aggregate(self):
        try:
            target_assets_df = self.portfolio.get_signal()
            # check if bounds are satisfied
            if not target_assets_df["weighted_signal"].abs().sum() <= 1:
                logger.error("Sum of fund percentage is out of bounds")
                raise Exception("Total assets percentage cannot exceed ±1.0")

            target_assets = dict(
                zip(target_assets_df["base_asset"], target_assets_df["weighted_signal"])
            )
            logger.info(f"[on_aggregate] latest target assets: {target_assets}")

            # create portfolio signal (aegis)
            await self.metric_builder.create_portfolio_signal(
                portfolio_id=self.portfolio.id, signals=target_assets
            )

            # send signal to upstream
            payload = json.dumps(
                {
                    "assets": {
                        asset: str(
                            Decimal(str(signal)).quantize(
                                Decimal("0.01"), rounding=ROUND_HALF_UP
                            )
                        )
                        for asset, signal in target_assets.items()
                    },
                    "timestamp": time.time_ns(),
                }
            ).encode()
            await self.metric_builder.metric_stream.publish(self.portfolio.id, payload)
        except Exception as e:
            logger.warning(f"[on_aggregate] an exception has been raised: {e}")
            await self.metric_builder.create_portfolio_alert(
                portfolio_id=self.portfolio.id,
                title="Exception occured in on_aggregate",
                description=str(e),
                priority=1,
            )

    async def start(self):
        # List of jobs to schedule in the background
        await self.scheduler.schedule(
            handler=self.on_aggregate, trigger=self.aggregate_window
        )

        await self.metric_builder.metric_stream.subscribe(
            "alpha_signal.*", callback=self.on_signal
        )
        await asyncio.gather(self.scheduler.start())


class AlphaExecutor:
    def __init__(
        self,
        alphas: list[Alpha],
        dataloader: DataLoader,
        datasource_api_key: str,
        metric_stream: MetricStream,
        datasource_stream: DatasourceStream,
        init_batch_size: int = 50,
        health_check_trigger: Trigger = Trigger.Cron("*/1 * * * *"),  # every 1 min
    ):
        for alpha in alphas:
            if len(alpha.data_infos) == 0:
                raise ValueError(f"Alpha {alpha.id} has 0 data info")

        self.alphas = alphas
        self.scheduler = Scheduler()
        self.datamap = Datamap(
            data_infos=list(flat_map(lambda a: a.data_infos, self.alphas))
        )
        self.dataloader = dataloader
        self.datasource_api_key = datasource_api_key
        self.aegis = MetricBuilder(metric_stream)
        self.stream = datasource_stream
        self.init_batch_size = init_batch_size
        self.health_check_trigger = health_check_trigger

        setup_logger()

        def handle_signal(signum, _):
            match signum:
                case signal.SIGINT | signal.SIGTERM:
                    self.on_shutdown()
                    exit(0)
                case _:
                    logger.warning(f"Handling signal ({signum}) by doing nothing.")

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    async def on_event(self, event: Event):
        match event.event_type:
            case EventType.DatasourceSubscribed:
                logger.info(f"[on_event] subscribed to datasource: {event.data}")
            case EventType.DatasourceUpdate:
                logger.info(f"[on_event] receive data: {event.data}")
                data = cast(CollectedData, event.data)
                topic = Topic.from_str(data["topic"])

                # update datamap with latest data
                for _data in data["data"]:
                    try:
                        self.datamap.update(topic, cast(Data, flatten_json(_data)))
                    except Exception as e:
                        logger.error(f"Failed to update Datamap due to: {e}")

                # resync data by fetching from rest again
                before_df = self.datamap.map[topic].to_df()
                before_df = before_df.select(sorted(before_df.columns))
                await self.datamap.resync(topic=topic, dataloader=self.dataloader)

                after_df = self.datamap.map[topic].to_df()
                # after_df = self.data_manager.map[topic].to_df()
                after_df = after_df.select(sorted(after_df.columns))
                if not before_df.equals(after_df):
                    for alpha in self.alphas:
                        if topic in alpha.data_infos:
                            await self.aegis.create_alpha_alert(
                                alpha_id=alpha.id,
                                title=f"[resync] [{topic}] different data prior to resyncing",
                                description=f"before resync: {before_df}\nafter resync: {after_df}",
                                priority=3,
                            )
                    logger.warning(
                        f"[resync] [{topic}] different data prior to resyncing"
                    )
                    logger.warning(f"[resync] [{topic}] before resync: {before_df}")
                    logger.warning(f"[resync] [{topic}] after resync: {after_df}")

                # process the data for each alpha that uses that data
                for alpha in self.alphas:
                    if Topic.from_str(data["topic"]) not in map(
                        lambda di: Topic.from_str(di.topic), alpha.data_infos
                    ):
                        continue

                    infos = acq_info(data["topic"], alpha.data_infos)
                    if len(infos) == 0:
                        continue

                    # insert alpha_trigger
                    await self.aegis.create_alpha_trigger(
                        alpha_id=alpha.id, topic=infos[0].topic
                    )

                    for info in infos:
                        # process the datamap into data_df (combine multiple data infos)
                        last_closed_time = Topic.from_str(info.topic).last_closed_time(
                            True
                        )
                        if last_closed_time is None:
                            logger.error(f"{info.topic} has no interval")
                            continue

                        df = alpha.data_processor.process(
                            datamap=self.datamap,
                            last_closed_time=last_closed_time,
                        )
                        if df is None:
                            continue

                        # generate alpha signal based on latest data
                        signal_df = alpha.next(df)
                        if len(signal_df) == 0:
                            logger.error(f"{alpha.id} couldn't generate signal")
                            logger.error(f"{alpha.id} df: {df}")
                            continue

                        logging.info(f"data_df: {df}")
                        logging.info(f"[latest_signal] {signal_df}")
                        signal = np.format_float_positional(
                            signal_df.select("signal").to_numpy().ravel()[-1],
                            precision=2,
                            unique=False,
                        )

                        # send signal
                        payload = json.dumps(
                            {"signal": signal, "timestamp": time.time_ns()}
                        ).encode()
                        # insert alpha_signal (aegis)
                        await self.aegis.create_alpha_signal(
                            alpha_id=alpha.id, signal=signal
                        )
                        # send signal to portfolios
                        await self.aegis.metric_stream.publish(
                            f"alpha_signal.{alpha.id}", payload
                        )

            case _:
                logger.info(
                    f"[on_event] got {event.event_type} event with data: {event.data}"
                )

    def on_shutdown(self):
        logger.warning("[on_shutdown] received SIGINT / SIGTERM, shutdown process")

    async def on_health_check(self):
        try:
            for alpha in self.alphas:
                # Check whether data is not arrived for more than some time
                topics = list(
                    map(lambda di: Topic.from_str(di.topic), alpha.data_infos)
                )
                dfs = list(
                    map(
                        lambda d: d[1].to_df(),
                        filter(
                            lambda d: d[0] in topics,
                            self.datamap.map.items(),
                        ),
                    )
                )
                df = reduce(
                    lambda acc, x: acc.select("start_time").join(
                        x.select("start_time"), on="start_time"
                    ),
                    dfs,
                )
                last_joined_data_time: datetime = df["start_time"][-1]
                last_closed_time = min(
                    map(
                        lambda info: cast(
                            datetime, Topic.from_str(info.topic).last_closed_time(True)
                        ),
                        self.datamap.data_infos,
                    )
                )
                interval = max(
                    map(
                        lambda info: cast(
                            timedelta, Topic.from_str(info.topic).interval()
                        ),
                        self.datamap.data_infos,
                    )
                )

                if last_closed_time - last_joined_data_time >= 2 * interval:
                    logger.warning(
                        f"[on_health_check] [{alpha.id}] Data not arrived for more than {2 * interval}, latest_datapoint: {last_joined_data_time}, should_have: {last_closed_time}"
                    )
                    await self.aegis.create_alpha_alert(
                        alpha_id=alpha.id,
                        title=f"Data not arrived for more than {2 * interval}",
                        description=f"latest_datapoint: {last_joined_data_time}, should_have: {last_closed_time}",
                        priority=1,
                    )
        except Exception as e:
            logging.warning(f"[on_health_check] failed to health check: {e}")

    async def _start_datasource(self):
        topics = list(
            set(
                map(
                    lambda di: Topic.from_str(di.topic),
                    flat_map(lambda a: a.data_infos, self.alphas),
                )
            )
        )

        data_stream = await self.stream.connect(topics)
        if data_stream is None:
            raise Exception("Failed to connect to Datasource WS after max_retries")

        async for data in data_stream:
            try:
                if "data" in data:
                    await self.on_event(
                        Event(
                            event_type=EventType.DatasourceUpdate,
                            orig=data,
                            data=data,
                        )
                    )
                else:
                    await self.on_event(
                        Event(
                            event_type=EventType.DatasourceSubscribed,
                            orig=data,
                            data=data,
                        )
                    )
            except Exception as e:
                logger.warning(f"Datasource WS encountered an exception: {e}")
                traceback.print_exc()

                if "data" in data:
                    data = cast(CollectedData, data)
                    for alpha in self.alphas:
                        if Topic.from_str(data["topic"]) in map(
                            lambda di: Topic.from_str(di.topic), alpha.data_infos
                        ):
                            await self.aegis.create_alpha_alert(
                                alpha_id=alpha.id,
                                title="Exception raised in `on_event`",
                                description=str(e),
                                priority=2,
                            )
                else:
                    for alpha in self.alphas:
                        await self.aegis.create_alpha_alert(
                            alpha_id=alpha.id,
                            title="Exception raised in `on_event`",
                            description=str(e),
                            priority=2,
                        )

    async def start(self):
        logging.info("[init] initializing Datamap")

        topics = {
            Topic.from_str(data_info.topic) for data_info in self.datamap.data_infos
        }
        async with asyncio.TaskGroup() as tg:
            for topic in topics:
                tg.create_task(
                    self.datamap.resync(topic=topic, dataloader=self.dataloader)
                )

        # List of jobs to schedule in the background
        await self.scheduler.schedule(
            handler=self.on_health_check,
            trigger=self.health_check_trigger,
        )

        await asyncio.gather(
            self.scheduler.start(),
            self._start_datasource(),
        )

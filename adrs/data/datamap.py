import io
import asyncio
import pickle
import logging
import polars as pl
from typing import Self
from datetime import datetime, timezone, timedelta

from adrs.types import Topic, Data, SortedDataList
from adrs.data.dataloader import DataLoader

from .types import DataInfo

logger = logging.getLogger(__name__)


def dedup_data_infos_by_max_lookback_size(
    data_infos: list[DataInfo],
) -> list[DataInfo]:
    """Deduplicate DataInfo by keeping the one with the maximum lookback_size for each topic."""
    info_map = {}
    for info in data_infos:
        if info not in info_map:
            info_map[info] = info
        else:
            if info.lookback_size > info_map[info].lookback_size:
                info_map[info] = info
    return list(info_map.values())


class Datamap:
    map: dict[Topic, SortedDataList]
    data_infos: list[DataInfo]  # Dedupped, only holds info with greatest lookback
    topics: set[Topic]

    def __init__(self):
        self.map = {}
        self.data_infos = []

    def get_lookback_size(self, topic: Topic) -> int:
        return max(
            map(
                lambda di: di.lookback_size,
                filter(lambda di: Topic.from_str(di.topic) == topic, self.data_infos),
            )
        )

    def update(self, topic: Topic, data: Data):
        lookback_size = self.get_lookback_size(topic)

        # check for race condition: duplicate data
        if (
            topic in self.map
            and self.map[topic]["start_time"][-1] == data["start_time"]
        ):
            logging.warning(f"Duplicate data for topic {topic} at {data['start_time']}")
            self.map[topic].data[-1] = data
            return

        # maintain the datamap
        if topic not in self.map:
            self.map[topic] = SortedDataList([data])
        else:
            self.map[topic].append(data)
            self.map[topic].data = self.map[topic].data[-lookback_size:]

    def is_ready(self) -> bool:
        has_init_all_df = len(self.topics) == len(self.map.keys())
        has_enough_data = all(
            len(self.map[Topic.from_str(info.topic)]) >= info.lookback_size
            for info in self.data_infos
        )
        return has_init_all_df and has_enough_data

    def write_ipc(self) -> bytes:
        """Serialize the Datamap into bytes (IPC for DataFrames + pickle for metadata)."""
        payload = {"map": {}, "data_infos": self.data_infos}

        for topic, data_list in self.map.items():
            df = data_list.to_df()
            buf = io.BytesIO()
            df.write_ipc(buf)
            payload["map"][topic] = buf.getvalue()  # raw bytes # type: ignore

        return pickle.dumps(payload)

    @classmethod
    def read_ipc(cls, raw: bytes) -> Self:
        """Deserialize from bytes back into a Datamap."""
        payload = pickle.loads(raw)
        obj = cls()

        # rebuild map
        for topic, ipc_bytes in payload["map"].items():
            buf = io.BytesIO(ipc_bytes)
            obj.map[topic] = SortedDataList.from_df(pl.read_ipc(buf))

        return obj

    async def _init(
        self,
        dataloader: DataLoader,
        topic: Topic,
        start_time: datetime,
        end_time: datetime,
        should_lookback: bool = True,
    ):
        interval = topic.interval()
        if interval is None:
            raise Exception(f"Topic {topic} does not have an interval")
        lookback_size = self.get_lookback_size(topic)

        start_time = (
            start_time - interval * lookback_size if should_lookback else start_time
        ).replace(hour=0, minute=0, second=0, microsecond=0)

        last_closed = topic.last_closed_time_relative(end_time, is_collect=False)
        if last_closed is None:
            raise Exception(f"Topic {topic} does not have a last_closed_time_relative")

        # Skip if have already loaded before
        if (
            topic in self.map
            and self.map[topic]
            .data[0]["start_time"]
            .replace(hour=0, minute=0, second=0, microsecond=0)
            <= start_time
            and self.map[topic].data[-1]["start_time"] >= last_closed
        ):
            return

        # Explicitly override today's data because it might be incomplete
        if end_time.date() == datetime.now(tz=timezone.utc).date():
            end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info(
                f"Loading data for topic {topic} from {start_time} to {end_time}"
            )
            df = await dataloader.load(
                topic=str(topic),
                start_time=start_time,
                end_time=end_time,
            )
            today_df = await dataloader.load(
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
            df = await dataloader.load(
                topic=str(topic),
                start_time=start_time,
                end_time=end_time,
            )
        self.map[topic] = SortedDataList.from_df(df)
        logger.info(f"Loaded {len(df)} datapoints for topic {topic}")

    async def init(
        self,
        dataloader: DataLoader,
        infos: list[DataInfo],
        start_time: datetime,
        end_time: datetime,
        should_lookback: bool = True,
    ):
        self.data_infos = dedup_data_infos_by_max_lookback_size(infos + self.data_infos)
        self.topics = {Topic.from_str(data_info.topic) for data_info in self.data_infos}
        async with asyncio.TaskGroup() as tg:
            for topic in self.topics:
                tg.create_task(
                    self._init(dataloader, topic, start_time, end_time, should_lookback)
                )

    def get(self, info: DataInfo) -> pl.DataFrame:
        return (
            self.map[Topic.from_str(info.topic)]
            .to_df()
            .select(
                [
                    "start_time",
                    *(pl.col(col.src).alias(col.dst) for col in info.columns),
                ]
            )
        )

    def __getitem__(self, info: DataInfo) -> pl.DataFrame:
        return self.get(info)

    def keys(self):
        return self.map.keys()

    def values(self):
        return self.map.values()

    def items(self):
        return self.map.items()

    def __len__(self):
        return len(self.map)

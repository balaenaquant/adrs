import io
import pickle
import logging
import polars as pl
from typing import Self
from datetime import datetime, timezone, timedelta

from cybotrade import Topic
from flow import DataLoader
from cybotrade_datasource import Data

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
    map: dict[DataInfo, pl.DataFrame]
    data_infos: list[DataInfo]

    def __init__(self, data_infos: list[DataInfo] = []):
        self.map = {}
        self.data_infos = dedup_data_infos_by_max_lookback_size(data_infos)

    def update_df(self, info: DataInfo, df: pl.DataFrame):
        # make data into a dataframe (replace the start_time as a UTC timestamp)
        df = (
            df.with_columns(
                pl.col("start_time")
                .dt.replace_time_zone(time_zone="UTC")
                .dt.cast_time_unit(time_unit="ms")
            )
            # filter and rename the column based on info
            .select(
                [
                    "start_time",
                    *(pl.col(col.src).alias(col.dst) for col in info.columns),
                ]
            )
        )

        # put into datamap
        self.map[info] = df

        # update data_infos if not exist
        if info not in self.data_infos:
            self.data_infos.append(info)

    def update(self, info: DataInfo, data: Data):
        lookback_size = info.lookback_size

        # make data into a dataframe (replace the start_time as a UTC timestamp)
        df = (
            pl.DataFrame(data)
            .with_columns(
                pl.col("start_time")
                .dt.replace_time_zone(time_zone="UTC")
                .dt.cast_time_unit(time_unit="ms")
            )
            # filter and rename the column based on info
            .select(
                [
                    "start_time",
                    *(pl.col(col.src).alias(col.dst) for col in info.columns),
                ]
            )
        )

        # check for race condition: duplicate data
        if info in self.map and self.map[info]["start_time"][-1] == data["start_time"]:
            logging.warning(
                f"Duplicate data for topic {info.topic} at {data['start_time']}"
            )
            self.map[info] = self.map[info][:-1]
            self.map[info].extend(df)
            return

        # maintain the datamap (push and pop from DataFrame)
        if info not in self.map:
            # put new data
            self.map[info] = df
        else:
            # pop from head
            if len(self.map[info]) >= lookback_size:
                self.map[info] = self.map[info][1:]
            # push to tail
            self.map[info] = self.map[info].extend(df)

    def is_ready(self) -> bool:
        has_init_all_df = len(self.data_infos) == len(self.map.keys())
        has_enough_data = all(
            len(self.map[info]) == info.lookback_size for info in self.data_infos
        )
        return has_init_all_df and has_enough_data

    def write_ipc(self) -> bytes:
        """Serialize the Datamap into bytes (IPC for DataFrames + pickle for metadata)."""
        payload = {"map": {}, "data_infos": self.data_infos}

        for info, df in self.map.items():
            buf = io.BytesIO()
            df.write_ipc(buf)
            payload["map"][info] = buf.getvalue()  # raw bytes # type: ignore

        return pickle.dumps(payload)

    @classmethod
    def read_ipc(cls, raw: bytes) -> Self:
        """Deserialize from bytes back into a Datamap."""
        payload = pickle.loads(raw)
        obj = cls(payload["data_infos"])  # re-init with data_infos

        # rebuild map
        for info, ipc_bytes in payload["map"].items():
            buf = io.BytesIO(ipc_bytes)
            obj.map[info] = pl.read_ipc(buf)

        return obj

    async def _init(
        self,
        dataloader: DataLoader,
        info: DataInfo,
        start_time: datetime,
        end_time: datetime,
        should_lookback: bool = True,
    ):
        topic = Topic.from_str(info.topic)
        interval = topic.interval()
        if interval is None:
            raise Exception(f"Topic {topic} does not have an interval")

        start_time = (
            start_time - interval * info.lookback_size
            if should_lookback
            else start_time
        ).replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = end_time

        # Skip if have already loaded before
        if (
            info in self.map
            and self.map[info]
            .row(0, named=True)["start_time"]
            .replace(hour=0, minute=0, second=0, microsecond=0)
            <= start_time
            and self.map[info].row(-1, named=True)["start_time"]
            >= topic.last_closed_time_relative(end_time, is_collect=False)
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
        self.update_df(info=info, df=df)
        if info not in self.data_infos:
            self.data_infos.append(info)
        logger.info(f"Loaded {len(df)} datapoints for topic {topic}")

    async def init(
        self,
        dataloader: DataLoader,
        infos: list[DataInfo],
        start_time: datetime,
        end_time: datetime,
        should_lookback: bool = True,
    ):
        for info in infos:
            await self._init(dataloader, info, start_time, end_time, should_lookback)

    def get(self, info: DataInfo) -> pl.DataFrame:
        return self.map[info]

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

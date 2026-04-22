import logging
import polars as pl
import polars.selectors as cs

from functools import reduce
from datetime import datetime, timedelta
from typing import cast
from abc import abstractmethod

from adrs.types import Topic

from .types import DataInfo
from .datamap import Datamap


logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, data_infos: list[DataInfo] = []):
        self.data_infos = data_infos

    @abstractmethod
    def process(
        self,
        datamap: Datamap,
        last_closed_time: datetime | None = None,
    ) -> pl.DataFrame | None:
        intervals = list(
            Topic.from_str(info.topic).interval() for info in self.data_infos
        )
        if None in intervals:
            raise Exception(
                "Requires custom implementation of DataProcessor since there are topics that does not have an interval."
            )
        intervals = cast(list[timedelta], intervals)
        if not all(interval == intervals[0] for interval in intervals):
            raise Exception(
                "Requires custom implementation of DataProcessor since there are topics that does not have the same interval."
            )

        lookback_sizes = list(info.lookback_size for info in self.data_infos)
        if not all(
            lookback_size == lookback_sizes[0] for lookback_size in lookback_sizes
        ):
            raise Exception(
                "Requires custom implementation of DataProcessor since there are topics that does not have the same lookback size."
            )

        # Make sure we have collected all topics for current time
        topic_map = {topic: sdl for topic, sdl in datamap.items()}
        for info in self.data_infos:
            if Topic.from_str(info.topic) not in topic_map:
                raise Exception(f"Topic '{info.topic}' not found in datamap.")
        dfs = [
            topic_map[Topic.from_str(info.topic)]
            .to_df()
            .select(
                [
                    "start_time",
                    *(pl.col(col.src).alias(col.dst) for col in info.columns),
                ]
            )
            for info in self.data_infos
        ]

        df = reduce(lambda acc, x: acc.join(x, on="start_time"), dfs)
        last_joined_data_time = df["start_time"][-1]
        if last_closed_time is not None and last_joined_data_time < last_closed_time:
            logging.warning(
                f"[WAIT_DATA] last_closed_time is {last_closed_time} but currently only have all data up to {last_joined_data_time}"
            )
            return None

        missing_data_df = df.upsample(
            time_column="start_time", every=intervals[0]
        ).join(df, on="start_time", how="anti")
        if missing_data_df.height > 0:
            logging.warning(f"Missing data in datas_df: {missing_data_df}")

        return (
            df.drop_nulls()
            .drop_nans()
            .filter(~pl.any_horizontal(cs.numeric().is_infinite()))
        )

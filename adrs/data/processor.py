import logging
import polars as pl
import polars.selectors as cs
from functools import reduce
from datetime import datetime
from abc import abstractmethod

from cybotrade import Topic

from .types import DataInfo
from .datamap import Datamap


logger = logging.getLogger(__name__)


class DataProcessor:
    # The final dataframe (after processing)
    df: pl.DataFrame

    # The data informations given (details about how to resample, preprocess, etc.)
    data_infos: list[DataInfo]

    def set_data_infos(self, data_infos: list[DataInfo]):
        self.data_infos = data_infos

    @abstractmethod
    def process(
        self,
        datamap: Datamap,
        last_closed_time: datetime | None,
    ) -> pl.DataFrame | None:
        raise NotImplementedError("All DataProcessor should implement 'process' method")


class UniformDataProcessor(DataProcessor):
    @abstractmethod
    def process(
        self,
        datamap: Datamap,
        last_closed_time: datetime | None,
    ) -> pl.DataFrame | None:
        intervals = list(
            Topic.from_str(info.topic).interval() for info in self.data_infos
        )
        if None in intervals:
            raise Exception(
                "Requires custom implementation of DataProcessor since there are topics that does not have an interval."
            )
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
        dfs = list(
            map(
                lambda d: d[1],
                filter(lambda d: d[0] in self.data_infos, datamap.items()),
            )
        )
        df = reduce(lambda acc, x: acc.join(x, on="start_time"), dfs)
        last_joined_data_time = df["start_time"][-1]
        if last_closed_time is not None and last_joined_data_time < last_closed_time:
            logging.warning(
                f"[WAIT_DATA] last_closed_time is {last_closed_time} but currently only have all data up to {last_joined_data_time}"
            )
            return None

        return (
            df.drop_nulls()
            .drop_nans()
            .filter(~pl.any_horizontal(cs.numeric().is_infinite()))
        )

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import datetime, timedelta

from .processor import DataProcessor

from .types import DataInfo, DataColumn
from .datamap import Datamap
from .dataloader import DataLoader
from .handler import yfinance_handler
from .cache import Cache
from .connector import DatasourceStream, MetricStream, MetricBuilder

if TYPE_CHECKING:
    from adrs.performance import Evaluator

__all__ = [
    "Datamap",
    "DataProcessor",
    "DataInfo",
    "DataColumn",
    "DataLoader",
    "yfinance_handler",
    "MetricStream",
    "MetricBuilder",
    "DatasourceStream",
    "Cache",
    "DataLoader",
]


async def make_datamap(
    dataloader: DataLoader,
    start_time: datetime,
    end_time: datetime,
    data_infos: list[DataInfo],
    evaluator: Evaluator | None = None,
    evaluator_offset: timedelta = timedelta(days=1),
) -> Datamap:
    """Create a datamap and initialize with given data infos and optionally evaluator."""

    datamap = Datamap()

    # Setup the datamap (download data)
    await datamap.init(
        dataloader=dataloader,
        infos=data_infos,
        start_time=start_time,
        end_time=end_time,
    )

    # download data with (+1 day offset for candle shift)
    if evaluator is not None:
        await datamap.init(
            dataloader=dataloader,
            infos=list(evaluator.assets.values()),
            start_time=start_time,
            end_time=end_time + evaluator_offset,
        )

    return datamap

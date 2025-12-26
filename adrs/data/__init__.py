from .processor import DataProcessor
from .event import SimulationDataEvent
from .types import DataInfo, DataColumn
from .datamap import Datamap
from .dataloader import DataLoader

__all__ = [
    "Datamap",
    "DataProcessor",
    "SimulationDataEvent",
    "DataInfo",
    "DataColumn",
    "DataLoader",
]


from datetime import datetime, timedelta
from adrs.performance import Evaluator


async def make_datamap(
    dataloader: DataLoader,
    start_time: datetime,
    end_time: datetime,
    data_infos: list[DataInfo],
    evaluator: Evaluator | None = None,
    evaluator_offset: timedelta = timedelta(days=1),
) -> Datamap:
    """Create a datamap and initialize with given data infos and optionally evaluator."""

    datamap = Datamap(data_infos)

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

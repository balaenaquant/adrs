import logging

from flow import DataLoader

from datetime import datetime

from cybotrade import Topic
from cybotrade.io import Event, EventType, EventHandler


logger = logging.getLogger(__name__)


class SimulationDataEvent(EventHandler):
    def __init__(
        self,
        dataloader: DataLoader,
        topics: list[Topic],
        start_time: datetime,
        end_time: datetime,
    ):
        self.dataloader = dataloader
        self.topics = topics
        self.start_time = start_time
        self.end_time = end_time

    # NOTE: This method is a no-op because it will be overridden.
    async def on_event(self, event: Event):
        pass

    async def start(self):
        # Load data from the dataloader
        datas = []
        for topic in self.topics:
            logger.info(f"Loading data for topic: {topic}")
            df = await self.dataloader.load(
                topic=str(topic), start_time=self.start_time, end_time=self.end_time
            )
            logger.info(f"Loaded {len(df)} datapoints for topic: {topic}")
            datas += map(lambda d: (topic, d), df.drop("date").to_dicts())
        datas.sort(key=lambda x: x[1]["start_time"])

        for data in datas:
            await self.on_event(
                Event(
                    event_type=EventType.DatasourceUpdate,
                    orig=None,
                    data={"topic": str(data[0]), "data": [data[1]]},
                )
            )


class NoopDataEvent(EventHandler):
    def __init__(self):
        pass

    # NOTE: This method is a no-op because it will be overridden.
    async def on_event(self, event: Event):
        pass

    async def start(self):
        pass

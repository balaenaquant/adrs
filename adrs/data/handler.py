import logging

from datetime import datetime

from adrs.data.datasource import CybotradeDatasource
from adrs.types import Topic
from adrs.data.cache import Cache

logger = logging.getLogger(__name__)


def cybotrade_handler(
    datasource: CybotradeDatasource,
    cache: Cache,
):
    async def handler(topic_str: str, start_time: datetime, end_time: datetime):
        topic = Topic.from_str(topic_str)

        cache.init(topic=topic, start_time=start_time, end_time=end_time)

        datapoints = await cache.download(datasource, topic)
        logger.info("[%s] downloaded %d datapoints", topic, datapoints)

        return await cache.read(topic, start_time, end_time)

    return handler

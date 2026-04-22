from pathlib import Path

from polars import DataFrame
from datetime import datetime
from typing import Optional, Callable, Awaitable

from adrs.data.cache import Cache
from adrs.data.handler import cybotrade_handler
from adrs.data.datasource import DEFAULT_API_URL, CybotradeDatasource

type Handler = Callable[[str, datetime, datetime], Awaitable[DataFrame | None]]


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        credentials: Optional[dict[str, str]] = None,
        format: Optional[str] = None,
        cybotrade_api_url: Optional[str] = None,
        handlers: list[Handler] = [],
    ):
        self.data_dir = data_dir
        self.credentials = credentials
        self.format = format
        self.cybotrade_api_url = cybotrade_api_url
        self.handlers = handlers

    async def load(
        self,
        topic: str,
        start_time: datetime,
        end_time: datetime,
        override_existing: bool = False,
    ) -> DataFrame:
        for handler in self.handlers:
            df = await handler(topic, start_time, end_time)
            if df is not None:
                return df

        # default handler
        if self.credentials is None:
            raise Exception("No credentials given for default handler")

        cybotrade_api_key = self.credentials.get("cybotrade_api_key")
        if cybotrade_api_key is None:
            raise Exception("No cybotrade_api_key given for default handler")

        cache = Cache(
            data_path=Path(self.data_dir),
            format=self.format if self.format else "parquet",
            override_existing=override_existing,
        )

        datasource = CybotradeDatasource(
            api_key=cybotrade_api_key,
            base_url=self.cybotrade_api_url
            if self.cybotrade_api_url
            else DEFAULT_API_URL,
        )

        default_handler = cybotrade_handler(
            datasource=datasource,
            cache=cache,
        )
        return await default_handler(
            topic_str=topic,
            start_time=start_time,
            end_time=end_time,
        )

    def add_handler(self, handler: Handler):
        self.handlers.append(handler)

    def remove_handler(self, handler: Handler):
        self.handlers = list(filter(lambda h: id(h) != id(handler), self.handlers))

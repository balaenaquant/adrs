from pathlib import Path

from polars import DataFrame
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any

from adrs.data.cache import Cache
from adrs.data.handler import datasource_handler
from adrs.data.datasource import (
    DEFAULT_API_URL,
    Datasource,
    CybotradeDatasource,
    ClickhouseDatasource,
)

type Handler = Callable[[str, datetime, datetime], Awaitable[DataFrame | None]]


def parse_credentials(
    credentials: dict[str, Any], cybotrade_api_url: str | None = None
) -> Datasource | None:
    match credentials.get("type"):
        case "clickhouse":
            return ClickhouseDatasource(
                **{k: v for k, v in credentials.items() if k != "type"}
            )
        case None:
            if (
                key := credentials.get("cybotrade_api_key")
                or credentials.get("datasource_api_key")
            ) is not None:
                return CybotradeDatasource(
                    api_key=key,
                    base_url=cybotrade_api_url
                    if cybotrade_api_url
                    else DEFAULT_API_URL,
                )


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        credentials: dict[str, Any],
        format: Optional[str] = None,
        cybotrade_api_url: Optional[str] = None,
        handlers: list[Handler] = [],
    ):
        self.data_dir = data_dir
        self.credentials = credentials
        self.format = format
        self.cybotrade_api_url = cybotrade_api_url
        self.handlers = handlers

        datasource = parse_credentials(credentials, cybotrade_api_url)
        if datasource is None:
            raise Exception("invalid credentials format")
        self.datasource = datasource

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

        cache = Cache(
            data_path=Path(self.data_dir),
            format=self.format if self.format else "parquet",
            override_existing=override_existing,
        )

        default_handler = datasource_handler(datasource=self.datasource, cache=cache)
        return await default_handler(
            topic_str=topic,
            start_time=start_time,
            end_time=end_time,
        )

    def add_handler(self, handler: Handler):
        self.handlers.append(handler)

    def remove_handler(self, handler: Handler):
        self.handlers = list(filter(lambda h: id(h) != id(handler), self.handlers))

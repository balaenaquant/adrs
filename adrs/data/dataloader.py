import flow
from typing import Optional, Callable, Awaitable
from datetime import datetime
from polars import DataFrame

type Handler = Callable[[str, datetime, datetime], Awaitable[DataFrame | None]]


class DataLoader(flow.DataLoader):
    def __new__(
        cls,
        data_dir: str,
        credentials: Optional[dict[str, str]] = None,
        format: Optional[str] = None,
        use_cybotrade_datasource: Optional[bool] = None,
        cybotrade_api_url: Optional[str] = None,
        handlers: list[Handler] = [],
    ):
        instance = super().__new__(
            cls,
            data_dir=data_dir,  # type: ignore
            credentials=credentials,  # type: ignore
            format=format,  # type: ignore
            use_cybotrade_datasource=use_cybotrade_datasource,  # type: ignore
            cybotrade_api_url=cybotrade_api_url,  # type: ignore
        )
        instance.handlers = handlers
        return instance

    # NOTE: This method is only required as type definitions because initialisation is done within `__new__`.
    def __init__(
        self,
        data_dir: str,
        credentials: Optional[dict[str, str]] = None,
        format: Optional[str] = None,
        use_cybotrade_datasource: Optional[bool] = None,
        cybotrade_api_url: Optional[str] = None,
        handlers: list[Handler] = [],
    ):
        pass

    async def load(
        self,
        topic: str,
        start_time: datetime,
        end_time: datetime,
        override_existing: Optional[bool] = None,
    ) -> DataFrame:
        for handler in self.handlers:
            df = await handler(topic, start_time, end_time)
            if df is not None:
                return df

        return await super().load(
            topic,
            start_time,
            end_time,
            override_existing,
        )

    def add_handler(self, handler: Handler):
        self.handlers.append(handler)

    def remove_handler(self, handler: Handler):
        self.handlers = list(filter(lambda h: id(h) != id(handler), self.handlers))

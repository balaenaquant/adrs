import polars as pl
import httpx
import asyncio
import logging

from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from typing import cast

from adrs.types import Data, Topic, SortedDataList

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.datasource.cybotrade.rs"


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class Datasource:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        max_limit: int,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_limit = max_limit

    @abstractmethod
    async def query(
        self,
        topic: Topic | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        flatten: bool = False,
    ) -> list[Data]:
        """Single-page fetch. Called by query_paginated.

        Returned list must contain dicts with at least a `start_time` key as tz-aware datetime.
        """
        raise NotImplementedError()

    async def query_paginated(
        self,
        topic: Topic | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        flatten: bool = False,
    ) -> pl.DataFrame:
        """Paginate across multiple query() calls, returning all rows as one DataFrame.

        Resolves missing args from topic.params before dispatching.
        - Range mode (start_time + end_time): pages forward until end_time.
        - Limit mode (limit only): pages backward from last closed candle.

        Returns DataFrame with start_time as Datetime("ms", "UTC").
        Raises ValueError if neither (start_time, end_time) nor limit can be resolved,
        or if topic has no interval.
        """
        if isinstance(topic, str):
            topic = Topic.from_str(topic)

        if start_time is None:
            if (st := topic.params.get("start_time")) is not None:
                start_time = ms_to_dt(int(st))
        if end_time is None:
            if (st := topic.params.get("end_time")) is not None:
                end_time = ms_to_dt(int(st))
        if limit is None:
            if (lim := topic.params.get("limit")) is not None:
                limit = int(lim)

        if (start_time is None or end_time is None) and limit is None:
            raise ValueError(
                "either (start_time, end_time) or (limit) must be specified"
            )
        interval = topic.interval()
        if interval is None:
            raise ValueError(f"{topic} does not include an interval")
        interval_ms = int(interval.total_seconds() * 1000)
        datas = SortedDataList()

        if start_time and end_time:
            # truncate end_time to interval boundary
            end_ms = (int(end_time.timestamp() * 1000) // interval_ms) * interval_ms
            end_time = ms_to_dt(end_ms)

            current_start = start_time
            total = (end_ms - int(start_time.timestamp() * 1000)) // interval_ms
            iter_n = 1

            while current_start < end_time:
                current_limit = (
                    self.max_limit if topic.is_block() else min(self.max_limit, total)
                )
                current_end_check = current_start + timedelta(
                    milliseconds=current_limit * interval_ms
                )

                logger.debug(
                    "[query %d] fetching %s (start_time: %s, limit: %d)",
                    iter_n,
                    topic,
                    current_start,
                    current_limit,
                )
                resp = await self.query(
                    topic,
                    start_time=current_start,
                    limit=current_limit,
                    flatten=flatten,
                )
                num = len(resp)
                logger.debug("[query %d] %s got %d datapoints", iter_n, topic, num)

                page_end = max(r["start_time"] for r in resp)
                if not datas.data:
                    datas.data = resp
                datas.merge(resp)

                if topic.is_block():
                    current_start = page_end + timedelta(seconds=1)
                else:
                    total = max(0, total - num)
                    current_start = page_end + interval
                    if total == 0:
                        break

                iter_n += 1

                if (
                    not topic.is_block()
                    and current_end_check >= end_time
                    and num < current_limit
                ):
                    break
                elif topic.is_block() and num < (current_limit - 1):
                    break

                await asyncio.sleep(0.1)

            if topic.is_block() and datas.data:
                datas.data = [
                    d for d in datas.data if start_time <= d["start_time"] < end_time
                ]
        else:
            # paginate backwards from last_closed_time
            current_end = topic.last_closed_time(is_collect=False)
            remaining = limit
            iter_n = 1

            while remaining > 0:
                current_limit = min(self.max_limit, remaining)
                logger.debug(
                    "[query %d] fetching %s (end_time: %s, limit: %d)",
                    iter_n,
                    topic,
                    current_end,
                    current_limit,
                )
                resp = await self.query(
                    topic, end_time=current_end, limit=current_limit, flatten=flatten
                )
                num = len(resp)
                logger.debug("[query %d] %s got %d datapoints", iter_n, topic, num)

                page_start = min(r["start_time"] for r in resp)
                remaining -= num
                current_end = (
                    page_start - timedelta(seconds=1)
                    if topic.is_block()
                    else page_start - interval
                )
                if not datas.data:
                    datas.data = resp
                datas.merge(resp)
                iter_n += 1

                if num < current_limit:
                    break
                await asyncio.sleep(0.1)

        if not datas.data:
            return pl.DataFrame()

        return datas.to_df()


class CybotradeDatasource(Datasource):
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_API_URL,
        max_limit: int = 100_000,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_limit=max_limit,
        )
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = httpx.AsyncClient(timeout=120.0)

    async def query(
        self,
        topic: Topic | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        flatten: bool = False,
    ) -> list[Data]:
        if isinstance(topic, str):
            topic = Topic.from_str(topic)
        params: dict[str, str] = dict(topic.params)
        if start_time is not None:
            params["start_time"] = str(int(start_time.timestamp() * 1000))
        if end_time is not None:
            params["end_time"] = str(int(end_time.timestamp() * 1000))
        if limit is not None:
            params["limit"] = str(limit)
        if flatten:
            params["flatten"] = "true"

        url = f"{self.base_url}/{topic.provider}/{topic.endpoint}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.get(
                    url, params=params, headers={"X-API-KEY": self.api_key}
                )
                resp.raise_for_status()
                batch = resp.json()["data"]
                return cast(
                    list[Data],
                    [
                        {**item, "start_time": ms_to_dt(item["start_time"])}
                        for item in batch
                    ],
                )
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                wait = self.backoff_base**attempt
                logger.warning(
                    "[%s] attempt %d/%d failed (%s), retrying in %.1fs",
                    topic,
                    attempt + 1,
                    self.max_retries,
                    type(e).__name__,
                    wait,
                )
                await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]

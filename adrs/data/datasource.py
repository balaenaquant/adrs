import httpx
import asyncio
import logging
import pandas as pd
import polars as pl
import clickhouse_connect

from typing import cast
from abc import abstractmethod
from datetime import datetime, timedelta, timezone

from adrs.types import Data, Topic, SortedDataList
from adrs.data.progress import inner_task

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.datasource.cybotrade.rs"


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class Datasource:
    @abstractmethod
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
        raise NotImplementedError()


class CybotradeDatasource(Datasource):
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_API_URL,
        max_limit: int = 100_000,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_limit = max_limit
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

    async def query_paginated(
        self,
        topic: Topic | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        flatten: bool = False,
    ) -> pl.DataFrame:
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
            end_ms = (int(end_time.timestamp() * 1000) // interval_ms) * interval_ms
            inner_total: int | None = (
                None
                if topic.is_block()
                else max(
                    0,
                    (end_ms - int(start_time.timestamp() * 1000)) // interval_ms,
                )
            )
        else:
            inner_total = None if topic.is_block() else limit

        async with inner_task(str(topic), inner_total) as _bar:
            if start_time and end_time:
                # pyrefly: ignore [unbound-name]
                end_time = ms_to_dt(end_ms)

                current_start = start_time
                total = (end_ms - int(start_time.timestamp() * 1000)) // interval_ms
                iter_n = 1

                while current_start < end_time:
                    current_limit = (
                        self.max_limit
                        if topic.is_block()
                        else min(self.max_limit, total)
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
                    if len(resp) == 0:  # stop pagination if no new data
                        break

                    if "start_time" not in resp[0].keys():
                        raise ValueError(
                            f"{type(self).__name__}.query() does not return 'start_time' in its records"
                        )
                    if not isinstance(resp[0]["start_time"], datetime):
                        raise TypeError(
                            f"'start_time' must be tz-aware datetime, got {type(resp[0]['start_time'])}"
                        )
                    num = len(resp)
                    _bar.advance(num)
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
                        d
                        for d in datas.data
                        if start_time <= d["start_time"] < end_time
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
                        topic,
                        end_time=current_end,
                        limit=current_limit,
                        flatten=flatten,
                    )
                    num = len(resp)
                    _bar.advance(num)
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


class ClickhouseDatasource(Datasource):
    def __init__(self, **kwargs):
        self.connect_params = kwargs
        self.ch = None

    async def _client(self) -> clickhouse_connect.driver.AsyncClient:
        if self.ch is None:
            self.ch = await clickhouse_connect.get_async_client(**self.connect_params)
        return self.ch

    def _make_query_params_filter(self, topic: Topic):
        if topic.provider == "bybit" and topic.endpoint == "double-win":
            return f"query_params LIKE '%interval={topic.params.get('interval')}&product={topic.params.get('currency')}%'"
        else:
            return f"query_params = '{topic.query_params_str()}'"

    def _make_query(self, topic: Topic, start_time: datetime, end_time: datetime):
        return f"""
            SELECT timestamp AS start_time, data FROM `{topic.provider}`.`{topic.endpoint}` FINAL
            WHERE {self._make_query_params_filter(topic)}
                AND timestamp >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}'
                AND timestamp < '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
            ORDER BY timestamp ASC
        """

    # Empirically-set chunk size (not a config knob elsewhere): a single
    # unpaginated multi-year query against a multi-record-per-timestamp
    # topic (full option chains, GEX snapshots -- ~800 nested structs per
    # row) overloaded clickhouse-connect's streaming parser
    # (StreamFailureError: "unrecognized data found in stream", ~5min in,
    # against a 7-year range). A single month (~744 rows * ~800 structs)
    # completed reliably in the same test. Chunking unconditionally (not
    # just for detected multi-record topics) keeps this simple and safe for
    # any topic, at the cost of a few extra round-trips for small
    # single-record ranges that were already fine unchunked.
    _MAX_CHUNK_DAYS = 31

    async def query_paginated(
        self,
        topic: Topic | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        flatten: bool = False,
    ) -> pl.DataFrame:
        if isinstance(topic, str):
            topic = Topic.from_str(topic)
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided")

        if end_time - start_time <= timedelta(days=self._MAX_CHUNK_DAYS):
            return await self._query_one_range(topic, start_time, end_time)

        frames = []
        cur = start_time
        while cur < end_time:
            nxt = min(cur + timedelta(days=self._MAX_CHUNK_DAYS), end_time)
            chunk = await self._query_one_range(topic, cur, nxt)
            if len(chunk):
                frames.append(chunk)
            cur = nxt
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame()

    async def _query_one_range(
        self, topic: Topic, start_time: datetime, end_time: datetime
    ) -> pl.DataFrame:
        ch = await self._client()

        # clickhouse-connect for pyarrow doesn't support JSON column yet,
        # see: https://github.com/ClickHouse/clickhouse-connect/issues/398
        # df = await ch.query_df_arrow(
        #     self._make_query(topic, start_time, end_time), dataframe_library="polars"
        # )

        pandas_df = await ch.query_df(self._make_query(topic, start_time, end_time))
        if pandas_df.empty:
            return pl.DataFrame()

        # Multi-record-per-timestamp topics (full option chains, GEX
        # snapshots): "data" is a LIST of per-instrument dicts, not one
        # flat dict. json_normalize on a list-per-cell Series indexes
        # positionally ("0".."N" columns) -- meaningless here since column
        # identity isn't stable across snapshots (chain composition
        # changes every hour), and it drops the "data" column name that
        # DataColumn(src="data", ...) needs downstream. Keep the list
        # intact; normalize numeric fields to float so polars can infer
        # one consistent Struct schema across all records instead of
        # choking on mixed int/float typing for the same field across
        # instruments.
        #
        # ClickHouse's raw storage preserves the source API's full
        # per-timestamp envelope, which for multi-record topics is itself
        # `{"data": [...]}` -- one extra wrapper level versus what the
        # cybotrade REST proxy client hands back (it unwraps `resp.json()
        # ["data"]` itself before this code ever sees it). Verified directly
        # via a raw SQL query against the real flow.`options/gamma-exposure`
        # table: cells are dicts of the form {"data": [<840 instrument
        # dicts>]}, not the list directly. Unwrap that one level, if present,
        # before checking for the list shape -- a no-op for any topic where
        # "data" is already the list itself.
        def _unwrap(v):
            if (
                isinstance(v, dict)
                and set(v.keys()) == {"data"}
                and isinstance(v["data"], list)
            ):
                return v["data"]
            return v

        first_non_null = _unwrap(
            next((v for v in pandas_df["data"] if v is not None), None)
        )
        is_multi_record = isinstance(first_non_null, list)

        if is_multi_record:

            def _normalize_record(rec: dict) -> dict:
                # Two independent sources of mixed typing for the SAME
                # struct field across different records, both verified
                # directly against real GEX data, not assumed:
                # 1. isinstance(v, (int, float)) doesn't catch numpy scalar
                #    types (np.int64, np.float64) that clickhouse-connect's
                #    driver returns for numeric JSON fields.
                # 2. The underlying JSON itself is inconsistently typed --
                #    the SAME field (e.g. dealerTotalInventory) arrives as a
                #    JSON string in some records and a JSON number in
                #    others. A blanket "leave strings alone" rule preserves
                #    that inconsistency instead of fixing it.
                # Try a numeric cast on everything except None/bool; genuine
                # text fields (currency, exchange, putCall,
                # instrumentNormalized) simply fail the cast and keep their
                # original string value.
                out = {}
                for k, v in rec.items():
                    if v is None or isinstance(v, bool):
                        out[k] = v
                        continue
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        # Some numeric fields serialize missing values as
                        # the literal JSON string "null" (verified: real
                        # data, not a true JSON null) instead of an actual
                        # null -- treat those as None rather than leaving a
                        # stray string in an otherwise-numeric field.
                        if isinstance(v, str) and v.strip().lower() in (
                            "null",
                            "none",
                            "nan",
                        ):
                            out[k] = None
                        else:
                            out[k] = v
                return out

            normalized = pandas_df["data"].map(
                lambda cell: [_normalize_record(r) for r in _unwrap(cell)]
                if cell is not None
                else cell
            )
            pandas_df = pandas_df.drop(columns=["data"])
        else:
            pandas_df = pandas_df.drop(columns=["data"]).join(
                # pyrefly: ignore [bad-argument-type]
                pd.json_normalize(pandas_df["data"])
            )

        for col in pandas_df.columns:
            # Fix dynamic 'object' columns containing mixed PyArrow-unfriendly types
            if pandas_df[col].dtype == "object":
                # Check if it's primarily numeric despite being labeled an 'object'
                converted_numeric = pd.to_numeric(pandas_df[col], errors="coerce")
                if not converted_numeric.isna().all():
                    pandas_df[col] = converted_numeric
                else:
                    # If it's mixed text/strings/None, force everything to string safely
                    pandas_df[col] = (
                        pandas_df[col]
                        .astype(str)
                        .replace("None", None)
                        .replace("nan", None)
                    )

        df = pl.from_pandas(pandas_df)
        if is_multi_record:
            # Build "data" via pl.Series directly from the normalized
            # Python list rather than pl.from_pandas' generic object-column
            # handling -- polars' own type inference over a list of dicts
            # reliably produces List(Struct(...)) once values are
            # type-consistent (verified against real GEX data); pandas'
            # pyarrow-conversion path for nested Python objects is not.
            df = df.with_columns(pl.Series("data", normalized.tolist()))

        return df

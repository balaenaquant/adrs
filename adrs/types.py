import polars as pl
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    model_serializer,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from typing import Any, Self, TypedDict, cast


class Data(TypedDict):
    start_time: datetime


class SortedDataList:
    def __init__(self, datas: list[Data] = []):
        self.data: list[Data] = datas

    def append(self, value: Data):
        self.data.append(value)
        self.sort()

    def sort(self):
        self.data = sorted(self.data, key=lambda d: d["start_time"])

    def merge(self, datas: list[Data]):
        df = pl.DataFrame(datas).with_columns(
            pl.col("start_time")
            .dt.replace_time_zone(time_zone="UTC")
            .dt.cast_time_unit(time_unit="ms")
        )

        cols = set(df.columns) - set(["start_time"])
        merged_df = (
            self.to_df()
            .join(df, how="full", on="start_time", coalesce=True)
            .select(
                "start_time",
                *[pl.coalesce(f"{name}_right", name).alias(name) for name in cols],
            )
            .sort("start_time")
        )

        self.data = cast(list[Data], merged_df.to_dicts())

    def to_df(self):
        return pl.DataFrame(self.data, infer_schema_length=None).with_columns(
            pl.col("start_time")
            .dt.replace_time_zone(time_zone="UTC")
            .dt.cast_time_unit(time_unit="ms")
        )

    @classmethod
    def from_df(cls, df: pl.DataFrame):
        return cls(datas=[Data(**d) for d in df.iter_rows(named=True)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


class Symbol:
    _MAX_LEN = 24

    def __init__(self, s: str) -> None:
        if len(s) > self._MAX_LEN:
            raise ValueError(f"Symbol exceeds max length of {self._MAX_LEN}: {s!r}")
        self._s = s

    @classmethod
    def from_str(cls, s: str) -> "Symbol":
        return cls(s)

    def hash(self) -> int:
        return sum(ord(c) for c in self._s)

    def split(self) -> tuple[str, str] | None:
        for i, c in enumerate(self._s):
            if c in "-/_:":
                return self._s[:i], self._s[i + 1 :]

        s = self._s

        if len(s) >= 5 and s[-5:] in (
            "FDUSD",
            "fdusd",
            "USDTM",
            "usdtm",
            "USDCM",
            "usdcm",
        ):
            return s[:-5], s[-5:]

        if len(s) >= 4 and s[-4:] in ("USDT", "USDC", "usdt", "usdc"):
            return s[:-4], s[-4:]

        if len(s) >= 3 and s[-3:] in ("USD", "usd"):
            return s[:-3], s[-3:]

        return None

    def __str__(self) -> str:
        return self._s

    def __repr__(self) -> str:
        return self._s

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return self._s == other._s
        if isinstance(other, str):
            return self._s == other
        return False

    def __hash__(self) -> int:
        return self.hash()

    def __copy__(self) -> "Symbol":
        return self

    def __deepcopy__(self, memo: dict) -> "Symbol":
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple:
        return (Symbol, (self._s,))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: cls(v) if isinstance(v, str) else v,
            serialization=core_schema.to_string_ser_schema(),
        )


class TopicError(Exception):
    pass


class Topic(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    endpoint: str
    params: dict[str, str]

    def query_params(self) -> dict[str, str]:
        return self.params

    def query_params_str(self) -> str:
        if not self.params:
            return ""
        return "&".join(f"{k}={v}" for k, v in sorted(self.params.items()))

    def endpoint_with_query_params(self) -> str:
        if not self.params:
            return self.endpoint
        query = "&".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.endpoint}?{query}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Topic):
            return False
        return (
            self.provider == other.provider
            and self.endpoint_with_query_params() == other.endpoint_with_query_params()
        )

    def __hash__(self) -> int:
        return hash((self.provider, self.endpoint_with_query_params()))

    def __repr__(self) -> str:
        return f"Topic {{ provider: {self.provider!r}, endpoint: {self.endpoint!r}, params: {dict(sorted(self.params.items()))!r} }}"

    def __str__(self) -> str:
        return f"{self.provider}|{self.endpoint_with_query_params()}"

    def __copy__(self) -> "Topic":
        return self

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "Topic":
        if memo is not None:
            memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple:
        return (Topic.from_str, (str(self),))

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, v: Any) -> Any:
        if isinstance(v, str):
            return cls.from_str(v).__dict__
        if isinstance(v, dict) and "query_params" in v and "params" not in v:
            v = {**v, "params": v["query_params"]}
        return v

    @model_serializer
    def _to_str(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, s: str) -> "Topic":
        tokens = s.split("|")
        if len(tokens) < 2:
            raise TopicError("cannot find provider/endpoint in the topic")
        if len(tokens) > 2:
            raise TopicError("topic must be in the '{provider}|{endpoint}' format")

        provider, endpoint_with_query_params = tokens

        if "?" in endpoint_with_query_params:
            endpoint, query_string = endpoint_with_query_params.split("?", 1)
        else:
            endpoint, query_string = endpoint_with_query_params, ""
        params: dict[str, str] = dict(parse_qsl(query_string))

        if provider == "cryptoquant":
            endpoint = endpoint.replace("flow/", "flows/", 1)

        if params.get("symbol") and endpoint == "candle":
            symbol_str = params["symbol"]
            parsed_symbol = Symbol.from_str(symbol_str)
            result = parsed_symbol.split()
            if result is None:
                raise TopicError(f"invalid symbol '{symbol_str}'")
            base, quote = result
            if "bybit" in provider or "binance" in provider:
                params["symbol"] = f"{base}{quote}"
            elif "coinbase" in provider:
                params["symbol"] = f"{base}-{quote}"

        return cls(provider=provider, endpoint=endpoint, params=params)

    def get_params(self, key: str) -> str:
        if key not in self.params:
            raise TopicError(f"key '{key}' not found in topic '{self}'")
        return self.params[key]

    def interval(self) -> timedelta | None:
        _MINUTE = 60_000
        _HOUR = 60 * _MINUTE
        _DAY = 24 * _HOUR

        def ms(n: int) -> timedelta:
            return timedelta(milliseconds=n)

        match self.provider:
            case "cryptoquant":
                key = "window"
                duration_map = {
                    "min": ms(_MINUTE),
                    "block": ms(10 * _MINUTE),
                    "10min": ms(10 * _MINUTE),
                    "hour": ms(_HOUR),
                    "day": ms(_DAY),
                }
            case "glassnode":
                key = "i"
                duration_map = {
                    "10m": ms(10 * _MINUTE),
                    "1h": ms(_HOUR),
                    "24h": ms(24 * _HOUR),
                    "1w": ms(7 * _DAY),
                    "1month": ms(30 * _DAY),
                }
            case "amberdata":
                match self.endpoint:
                    case "derivatives/analytics/futures-perpetuals/apr-basis/constant-maturities":
                        return ms(15 * _MINUTE)
                    case "derivatives/analytics/trades-flow/gamma-exposures-snapshots":
                        return ms(_HOUR)
                    case (
                        "derivatives/analytics/futures-perpetuals/realized-funding-rates-cumulated"
                        | "derivatives/analytics/futures-perpetuals/funding-rates"
                    ):
                        return ms(8 * _HOUR)
                    case (
                        "derivatives/analytics/trades-flow/decorated-trades"
                        | "derivatives/analytics/trades-flow/options-yields/tradfi"
                        | "derivatives/analytics/trades-flow/option-volumes/tradfi"
                        | "derivatives/analytics/realized-volatility/implied-vs-realized/tradfi"
                        | "derivatives/analytics/realized-volatility/cones/tradfi"
                        | "derivatives/analytics/realized-volatility/cones"
                        | "derivatives/analytics/volatility/svi-minutely"
                        | "derivatives/analytics/options-scanner/top-trades"
                        | "derivatives/analytics/trades-flow/gamma-exposures/normalized-usd"
                        | "derivatives/analytics/trades-flow/block-volumes"
                    ):
                        return ms(_MINUTE)
                key = "timeInterval"
                duration_map = {
                    "day": ms(_DAY),
                    "hour": ms(_HOUR),
                    "minute": ms(_MINUTE),
                }
            case "coinalyze":
                key = "interval"
                duration_map = {
                    "1min": ms(_MINUTE),
                    "5min": ms(5 * _MINUTE),
                    "15min": ms(15 * _MINUTE),
                    "30min": ms(30 * _MINUTE),
                    "1hour": ms(_HOUR),
                    "2hour": ms(2 * _HOUR),
                    "4hour": ms(4 * _HOUR),
                    "6hour": ms(6 * _HOUR),
                    "12hour": ms(12 * _HOUR),
                    "daily": ms(_DAY),
                }
            case _:
                key = "interval"
                duration_map = {
                    "1m": ms(_MINUTE),
                    "3m": ms(3 * _MINUTE),
                    "5m": ms(5 * _MINUTE),
                    "10m": ms(10 * _MINUTE),
                    "15m": ms(15 * _MINUTE),
                    "30m": ms(30 * _MINUTE),
                    "1h": ms(_HOUR),
                    "2h": ms(2 * _HOUR),
                    "4h": ms(4 * _HOUR),
                    "6h": ms(6 * _HOUR),
                    "12h": ms(12 * _HOUR),
                    "1d": ms(_DAY),
                    "3d": ms(3 * _DAY),
                    "1w": ms(7 * _DAY),
                    "1M": ms(30 * _DAY),
                }

        v = self.params.get(key)
        if v is None:
            return None
        return duration_map.get(v)

    def cron(self) -> str | None:
        match self.provider:
            case "cryptoquant":
                match self.params.get("window"):
                    case "min":
                        return "6 * * * * *"
                    case "block":
                        return "1 */10 * * * *"
                    case "10min":
                        return "0 */13 * * * *"
                    case "hour" if "/market-data/" in self.endpoint:
                        return "0 6 * * * *"
                    case "hour":
                        return "0 15 * * * *"
                    case "day":
                        return "0 6 0 * * *"
                    case _:
                        return None
            case "glassnode":
                match self.params.get("i"):
                    case "10m":
                        return "0 */13 * * * *"
                    case "1h":
                        return "0 6 * * * *"
                    case "24h":
                        return "0 6 0 * * *"
                    case "1w":
                        return "0 0 0 * * 1"
                    case "1month":
                        return "0 0 0 1 * *"
                    case _:
                        return None
            case "coinbase":
                match self.params.get("interval"):
                    case "1m":
                        return "6 * * * * *"
                    case "5m":
                        return "25 */5 * * * *"
                    case "15m":
                        return "15 */16 * * * *"
                    case "1h":
                        return "0 7 * * * *"
                    case "6h":
                        return "0 6 */7 * * *"
                    case "1d":
                        return "0 7 0 * * *"
                    case _:
                        return None
            case "amberdata":
                match self.params.get("timeInterval"):
                    case "minute":
                        return "3 * * * * *"
                    case "hour":
                        return "0 3 * * * *"
                    case "day":
                        return "0 3 0 * * *"
                    case _:
                        return None
            case "coinalyze":
                match self.params.get("interval"):
                    case "1min":
                        return "3 * * * * *"
                    case "5min":
                        return "3 */5 * * * *"
                    case "15min":
                        return "3 */15 * * * *"
                    case "30min":
                        return "3 */30 * * * *"
                    case "1hour":
                        return "0 3 * * * *"
                    case "2hour":
                        return "0 3 */2 * * *"
                    case "4hour":
                        return "0 3 */4 * * *"
                    case "6hour":
                        return "0 3 */6 * * *"
                    case "12hour":
                        return "0 3 */12 * * *"
                    case "daily":
                        return "0 3 0 * * *"
                    case _:
                        return None
            case _:
                match self.params.get("interval"):
                    case "1m":
                        return "1 * * * * *"
                    case "3m":
                        return "1 */3 * * * *"
                    case "5m":
                        return "1 */5 * * * *"
                    case "10m":
                        return "1 */10 * * * *"
                    case "15m":
                        return "1 */15 * * * *"
                    case "30m":
                        return "1 */30 * * * *"
                    case "1h":
                        return "1 0 * * * *"
                    case "2h":
                        return "1 0 */2 * * *"
                    case "4h":
                        return "1 0 */4 * * *"
                    case "6h":
                        return "1 0 */6 * * *"
                    case "12h":
                        return "1 */12 * * * *"
                    case "1d":
                        return "1 0 0 * * *"
                    case "3d":
                        return "1 0 * */3 * *"
                    case "1w":
                        return "1 0 0 * * 1"
                    case "1M":
                        return "1 0 0 1 * *"
                    case _:
                        return None

    def delay_ms(self) -> int:
        _SECOND = 1_000
        _MINUTE = 60 * _SECOND
        _HOUR = 60 * _MINUTE

        if self.provider == "cryptoquant" and self.endpoint.startswith(
            "eth/exchange-flows"
        ):
            return 3 * _MINUTE

        match self.provider:
            case "cryptoquant":
                match self.params.get("window"):
                    case "min":
                        return 10 * _SECOND
                    case "block" | "10min":
                        return 30 * _SECOND
                    case _:
                        return _MINUTE
            case "glassnode":
                match self.params.get("i"):
                    case "10m":
                        return 30 * _SECOND
                    case "1w" | "1month":
                        return _HOUR
                    case _:
                        return _MINUTE
            case "amberdata":
                match self.params.get("timeInterval"):
                    case "minute":
                        return 10 * _SECOND
                    case "day":
                        return _HOUR
                    case _:
                        return _MINUTE
            case "coinalyze":
                match self.params.get("interval"):
                    case "1min" | "5min" | "15min" | "30min":
                        return 10 * _SECOND
                    case "1hour" | "2hour" | "4hour" | "6hour" | "12hour":
                        return _MINUTE
                    case "daily":
                        return _HOUR
                    case _:
                        return _MINUTE
            case "coinglass" | "coinbase":
                match self.params.get("interval"):
                    case "1m" | "3m" | "5m":
                        return 10 * _SECOND
                    case "10m" | "15m" | "30m":
                        return 30 * _SECOND
                    case "1h" | "2h" | "4h" | "6h" | "12h":
                        return _MINUTE
                    case "1d" | "3d" | "1w" | "1M":
                        return _HOUR
                    case _:
                        return _MINUTE
            case _:
                return _SECOND

    def is_block(self) -> bool:
        return self.provider == "cryptoquant" and self.params.get("window") == "block"

    def last_closed_time_relative(
        self, timestamp: datetime, is_collect: bool
    ) -> datetime | None:
        interval = self.interval()
        if interval is None:
            return None

        interval_ms = int(interval.total_seconds() * 1000)
        ts_ms = int(timestamp.timestamp() * 1000)
        truncated_ms = (ts_ms // interval_ms) * interval_ms

        difference_ms = (
            2 * interval_ms
            if self.provider == "glassnode"
            and is_collect
            and interval_ms == 60 * 60 * 1000
            else interval_ms
        )

        return datetime.fromtimestamp(
            (truncated_ms - difference_ms) / 1000, tz=timezone.utc
        )

    def last_closed_time(self, is_collect: bool) -> datetime | None:
        return self.last_closed_time_relative(datetime.now(tz=timezone.utc), is_collect)


class Performance(BaseModel):
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    cagr: float
    annualized_return: float
    total_return: float
    min_cumu: float
    largest_win: float
    largest_loss: float
    num_datapoints: int
    num_trades: int
    avg_holding_time_in_seconds: float
    max_holding_time_in_seconds: float
    long_trades: int
    short_trades: int
    win_trades: int
    lose_trades: int
    win_streak: int
    lose_streak: int
    win_rate: float
    start_time: datetime
    end_time: datetime
    max_drawdown: float
    max_drawdown_percentage: float
    max_drawdown_start_date: datetime
    max_drawdown_end_date: datetime
    max_drawdown_recover_date: datetime
    max_drawdown_max_duration_in_days: float
    metadata: dict[str, Any]

    model_config = ConfigDict(extra="allow")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Performance):
            return False
        return self.model_dump(exclude={"metadata"}) == other.model_dump(
            exclude={"metadata"}
        )

    @staticmethod
    def from_df(
        df: pl.DataFrame,
        start_time: datetime,
        end_time: datetime,
        metadata: dict[str, Any] = {},
    ) -> Self:
        from adrs.performance.metric import Ratio, Drawdown, Trade

        return Performance.model_validate(
            {
                **Ratio().compute(df),
                **Drawdown().compute(df),
                **Trade().compute(df),
                "start_time": start_time,
                "end_time": end_time,
                "metadata": metadata,
            }
        )


__all__ = ["Topic", "TopicError", "Symbol", "Data"]

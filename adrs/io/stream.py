import json
import zlib
import time
import asyncio
import logging
import websockets

from datetime import datetime, timezone
from typing import AsyncIterator, AsyncGenerator, Awaitable, Callable, TypeVar, cast

from grpc import StatusCode
from grpc.aio import AioRpcError
from nats_client import NATSClient, Msg
from websockets.exceptions import ConnectionClosed  # type: ignore[import-untyped]

from adrs.types import (
    Topic,
    Message,
    CollectedData,
    Data,
    SubscriptionResponse,
    BoundedSet,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# gRPC codes worth retrying. JetStream rate-limiting (429) surfaces as
# UNAVAILABLE with "Received http2 header with status: 429" in the details;
# RESOURCE_EXHAUSTED also maps to 429 in some gRPC implementations.
_RETRYABLE_CODES = frozenset(
    {
        StatusCode.UNAVAILABLE,
        StatusCode.RESOURCE_EXHAUSTED,
        StatusCode.DEADLINE_EXCEEDED,
    }
)


async def _retry_grpc(
    op: Callable[[], Awaitable[_T]],
    *,
    desc: str,
    max_retries: int,
    retry_backoff: float,
) -> _T:
    """Run `op`, retrying transient gRPC failures with exponential backoff.

    Rate-limited JetStream ops are "rejected-not-processed", so retrying does
    not risk duplicate side effects. Non-retryable codes propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return await op()
        except AioRpcError as e:
            if e.code() not in _RETRYABLE_CODES or attempt == max_retries:
                raise
            delay = retry_backoff * (2**attempt)
            logger.warning(
                "%s transient gRPC failure (%s); retry %d/%d in %.2fs",
                desc,
                e.code(),
                attempt + 1,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # loop either returns or raises


_BINANCE_WS = {
    "binance-spot": "wss://stream.binance.com:9443/ws",
    "binance-futures": "wss://fstream.binance.com/ws",
}

_BYBIT_WS = {
    "bybit-linear": "wss://stream.bybit.com/v5/public/linear",
    "bybit-spot": "wss://stream.bybit.com/v5/public/spot",
    "bybit-inverse": "wss://stream.bybit.com/v5/public/inverse",
}

_BYBIT_INTERVAL = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}
_BYBIT_INTERVAL_INV = {v: k for k, v in _BYBIT_INTERVAL.items()}


def _ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _binance_worker(url: str, topics: list[Topic], queue: asyncio.Queue) -> None:
    streams = [
        f"{t.params['symbol'].lower()}@kline_{t.params['interval']}" for t in topics
    ]
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                await ws.send(
                    json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1})
                )
                await queue.put(
                    SubscriptionResponse(
                        conn_id=url, success=True, message=f"subscribed {streams}"
                    )
                )
                backoff = 1.0
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("e") != "kline":
                        continue
                    k = msg["k"]
                    if not k.get("x"):
                        continue
                    sym, iv = msg["s"], k["i"]
                    topic_str = next(
                        (
                            str(t)
                            for t in topics
                            if t.params["symbol"] == sym and t.params["interval"] == iv
                        ),
                        None,
                    )
                    if topic_str is None:
                        continue
                    await queue.put(
                        CollectedData(
                            topic=topic_str,
                            data=cast(
                                list[Data],
                                [
                                    {
                                        "start_time": _ms(k["t"]),
                                        "open": float(k["o"]),
                                        "high": float(k["h"]),
                                        "low": float(k["l"]),
                                        "close": float(k["c"]),
                                        "volume": float(k["v"]),
                                    }
                                ],
                            ),
                            local_timestamp_ms=_now_ms(),
                            type="delta",
                        )
                    )
        except (ConnectionClosed, OSError, Exception) as e:
            logger.warning("[binance] %s — reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def _bybit_worker(url: str, topics: list[Topic], queue: asyncio.Queue) -> None:
    args = [
        f"kline.{_BYBIT_INTERVAL.get(t.params['interval'], t.params['interval'])}.{t.params['symbol']}"
        for t in topics
    ]
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                await queue.put(
                    SubscriptionResponse(
                        conn_id=url, success=True, message=f"subscribed {args}"
                    )
                )
                backoff = 1.0
                async for raw in ws:
                    msg = json.loads(raw)
                    topic_field = msg.get("topic", "")
                    if not topic_field.startswith("kline."):
                        continue
                    _, bybit_iv, sym = topic_field.split(".", 2)
                    iv = _BYBIT_INTERVAL_INV.get(bybit_iv, bybit_iv)
                    topic_str = next(
                        (
                            str(t)
                            for t in topics
                            if t.params["symbol"] == sym and t.params["interval"] == iv
                        ),
                        None,
                    )
                    if topic_str is None:
                        continue
                    for item in msg.get("data", []):
                        if not item.get("confirm"):
                            continue
                        await queue.put(
                            CollectedData(
                                topic=topic_str,
                                data=cast(
                                    list[Data],
                                    [
                                        {
                                            "start_time": _ms(item["start"]),
                                            "open": float(item["open"]),
                                            "high": float(item["high"]),
                                            "low": float(item["low"]),
                                            "close": float(item["close"]),
                                            "volume": float(item["volume"]),
                                        }
                                    ],
                                ),
                                local_timestamp_ms=_now_ms(),
                                type="delta",
                            )
                        )
        except (ConnectionClosed, OSError, Exception) as e:
            logger.warning("[bybit] %s — reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def _coinbase_worker(url: str, topics: list[Topic], queue: asyncio.Queue) -> None:
    product_ids = list({t.params["symbol"] for t in topics})
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=30) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "subscribe",
                            "product_ids": product_ids,
                            "channel": "candles",
                        }
                    )
                )
                await queue.put(
                    SubscriptionResponse(
                        conn_id=url, success=True, message=f"subscribed {product_ids}"
                    )
                )
                backoff = 1.0
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("channel") != "candles":
                        continue
                    for event in msg.get("events", []):
                        msg_type = (
                            "snapshot" if event.get("type") == "snapshot" else "delta"
                        )
                        for candle in event.get("candles", []):
                            pid = candle["product_id"]
                            topic_str = next(
                                (str(t) for t in topics if t.params["symbol"] == pid),
                                None,
                            )
                            if topic_str is None:
                                continue
                            await queue.put(
                                CollectedData(
                                    topic=topic_str,
                                    data=cast(
                                        list[Data],
                                        [
                                            {
                                                "start_time": datetime.fromtimestamp(
                                                    int(candle["start"]),
                                                    tz=timezone.utc,
                                                ),
                                                "open": float(candle["open"]),
                                                "high": float(candle["high"]),
                                                "low": float(candle["low"]),
                                                "close": float(candle["close"]),
                                                "volume": float(candle["volume"]),
                                            }
                                        ],
                                    ),
                                    local_timestamp_ms=_now_ms(),
                                    type=msg_type,
                                )
                            )
        except (ConnectionClosed, OSError, Exception) as e:
            logger.warning("[coinbase] %s — reconnect in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


class PublicDatasourceStream:
    """DatasourceStream backed by public exchange WebSocket APIs.

    Supports: binance-spot, binance-futures, bybit-linear, bybit-spot,
              bybit-inverse, coinbase.
    """

    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None:
        by_provider: dict[str, list[Topic]] = {}
        for t in topics:
            by_provider.setdefault(t.provider, []).append(t)

        queue: asyncio.Queue[Message] = asyncio.Queue()
        tasks: list[asyncio.Task] = []

        for provider, pts in by_provider.items():
            if provider in _BINANCE_WS:
                tasks.append(
                    asyncio.create_task(
                        _binance_worker(_BINANCE_WS[provider], pts, queue),
                        name=f"ws_{provider}",
                    )
                )
            elif provider in _BYBIT_WS:
                tasks.append(
                    asyncio.create_task(
                        _bybit_worker(_BYBIT_WS[provider], pts, queue),
                        name=f"ws_{provider}",
                    )
                )
            elif provider == "coinbase":
                tasks.append(
                    asyncio.create_task(
                        _coinbase_worker(
                            "wss://advanced-trade-ws.coinbase.com", pts, queue
                        ),
                        name=f"ws_{provider}",
                    )
                )
            else:
                logger.warning(
                    "[PublicDatasourceStream] unsupported provider: %s", provider
                )

        if not tasks:
            return None

        async def _iter() -> AsyncGenerator[Message, None]:
            try:
                while True:
                    yield await queue.get()
            finally:
                for task in tasks:
                    task.cancel()

        return _iter()


def _transform_data_from_timestamp_ms(data: list[Data]) -> list[Data]:
    return list(
        map(
            lambda d: {
                **d,
                "start_time": datetime.fromtimestamp(
                    cast(int, d["start_time"]) / 1000, tz=timezone.utc
                ),
            },
            data,
        )
    )


class PublicNatsDatasourceStream:
    def __init__(
        self,
        flow_nats: NATSClient,
        timeout_secs: int = 5,
        max_timeout_secs: int = 60,
        multiplier: int = 2,
        max_retries: int = 5,
        retry_backoff: float = 0.5,
        log_title: str = "connect",
    ):
        self.flow_nats = flow_nats
        self.timeout_secs = timeout_secs
        self.max_timeout_secs = max_timeout_secs
        self.multiplier = multiplier
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.log_title = log_title

    async def _stream(self, topics: list[Topic]):
        queue: asyncio.Queue[Message] = asyncio.Queue()

        def make_handler(topic: Topic):
            dedup = BoundedSet(10)

            async def handler(msg):
                try:
                    message = json.loads(zlib.decompress(msg.data))
                    message["data"] = _transform_data_from_timestamp_ms(message["data"])

                    if len(message["data"]) == 0:
                        await queue.put(message)
                        return

                    key = (message["topic"], message["data"][0]["start_time"])
                    if dedup.add(key):
                        await queue.put(message)
                    else:
                        logging.warning(
                            f"[stream] received duplicated message on {topic}: {message}"
                        )
                except Exception as e:
                    logging.error(f"[stream] failed to process message on {topic}: {e}")

            return handler

        for topic in topics:
            handler = make_handler(topic)
            await _retry_grpc(
                lambda t=str(topic), cb=handler: self.flow_nats.subscribe(t, cb=cb),
                desc=f"[{self.log_title}] subscribe {topic}",
                max_retries=self.max_retries,
                retry_backoff=self.retry_backoff,
            )
            logging.info(f"[stream] subscribed to {topic}")

        async def generator() -> AsyncIterator[Message]:
            while True:
                message = await queue.get()
                yield message

        return generator()

    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None:
        attempts = 0
        while attempts < self.max_retries and self.timeout_secs < self.max_timeout_secs:
            try:
                async with asyncio.timeout(self.timeout_secs):
                    return await self._stream(topics)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{self.log_title}] connect operation timed out after {attempts + 1} attempts, retrying..."
                )
                attempts += 1
                self.timeout_secs *= self.multiplier
        return None


class PublicMetricStream:
    """Thin NATS transport: publishes/subscribes the subject verbatim.

    It used to prepend an `insert_prefix` to JetStream publishes and a separate
    `publish_prefix` to core publishes. That blanket `publish_prefix` mangled
    routing subjects (an alpha signal meant for `alpha_signal.<id>` went out on
    `portfolio_signal.alpha_signal.<id>`, which the PortfolioExecutor's
    `alpha_signal.*` subscription never matched). Subject construction now lives
    with the caller: MetricBuilder owns the dashboard-insert namespace, the
    executors own their routing roots."""

    def __init__(
        self,
        nats: NATSClient,
        max_retries: int = 5,
        retry_backoff: float = 0.5,
    ):
        self.nats = nats
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    async def init(self):
        await self.nats.jetstream()

    async def subscribe(
        self, subject: str, callback: Callable[[Msg], Awaitable[None]] | None
    ):
        async def default_cb(_: Msg) -> None:
            pass

        if callback is None:
            callback = default_cb

        await self.nats.subscribe(subject, cb=callback)

    async def publish(self, subject: str, payload: bytes, **kwargs) -> None:
        use_jetstream = kwargs.get("use_jetstream", False)

        async def _op() -> None:
            if use_jetstream:
                await self.nats.js_publish(subject=subject, payload=payload)
            else:
                await self.nats.publish(subject, payload)

        await _retry_grpc(
            _op,
            desc=f"publish {subject}",
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
        )

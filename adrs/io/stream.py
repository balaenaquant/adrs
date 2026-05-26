import json
import nats
import time
import asyncio
import logging

from nats.aio.msg import Msg
from nats.js.errors import NoStreamResponseError
from datetime import datetime, timezone
from typing import AsyncIterator, AsyncGenerator, cast, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed  # type: ignore[import-untyped]

from adrs.types import Topic, Message, CollectedData, Data, SubscriptionResponse

logger = logging.getLogger(__name__)

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


class PublicMetricStream:
    def __init__(
        self,
        nats: nats.NATS,  # type: ignore
        insert_prefix: str = "public_ts",
        publish_prefix: str = "portfolio_signal",
    ):
        self.nats = nats
        self.aegis_ts_prefix = insert_prefix
        self.publish_prefix = publish_prefix

    async def setup(self):
        """Create JetStream stream covering all metric subjects. Safe to call if stream already exists."""
        js = self.nats.jetstream()
        try:
            await js.find_stream(f"{self.aegis_ts_prefix}.>")
        except Exception:
            await js.add_stream(
                name=self.aegis_ts_prefix,
                subjects=[f"{self.aegis_ts_prefix}.>"],
            )

    async def subscribe(
        self, subject: str, callback: Callable[[Msg], Awaitable[None]] | None
    ):
        await self.nats.subscribe(subject, cb=callback)

    async def publish(self, subject: str, payload: bytes, **kwargs) -> None:
        use_jetstream = kwargs.get("use_jetstream", False)

        if use_jetstream:
            try:
                await self.nats.jetstream().publish_async(
                    subject=f"{self.aegis_ts_prefix}.{subject}",
                    payload=payload,
                )
            except NoStreamResponseError:
                await self.nats.publish(f"{self.aegis_ts_prefix}.{subject}", payload)
        else:
            await self.nats.publish(
                f"{self.publish_prefix}.{subject}",
                payload,
            )

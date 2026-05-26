import nats
import json
import time
import logging

from decimal import Decimal
from nats.aio.msg import Msg
from typing import Protocol, AsyncIterator, Callable, Awaitable

from adrs.types import Topic, Message


class MetricStream(Protocol):
    async def subscribe(
        self, subject: str, callback: Callable[[Msg], Awaitable[None]] | None
    ): ...

    async def publish(self, subject: str, payload: bytes, **kwargs) -> None: ...


class DatasourceStream(Protocol):
    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None: ...


async def connect_nats(url: str, user: str = "", password: str = "") -> nats.NATS:  # type: ignore
    async def error_handler(e):
        logging.error(f"[NATS] Error: {e}")

    async def disconnected_handler():
        logging.warning("[NATS] Disconnected — attempting reconnect...")

    async def reconnected_handler():
        logging.info(f"[NATS] Reconnected to {nc.connected_url.netloc}")  # type: ignore

    async def closed_handler():
        logging.error("[NATS] Connection permanently closed")

    nc = await nats.connect(
        url,
        user=user or None,
        password=password or None,
        ping_interval=10,
        max_outstanding_pings=3,
        allow_reconnect=True,
        max_reconnect_attempts=-1,
        reconnect_time_wait=2,
        connect_timeout=5,
        error_cb=error_handler,
        disconnected_cb=disconnected_handler,
        reconnected_cb=reconnected_handler,
        closed_cb=closed_handler,
    )

    return nc


class MetricBuilder:
    def __init__(self, metric_stream: MetricStream):
        self.metric_stream = metric_stream

    async def create_alpha_trigger(self, alpha_id: str, topic: str | None = None):
        return await self.metric_stream.publish(
            "alpha_trigger",
            json.dumps(
                {
                    "alpha_id": alpha_id,
                    "topic": topic,
                    "timestamp": time.time_ns(),
                }
            ).encode(),
            use_jetstream=True,
        )

    async def create_alpha_signal(self, alpha_id: str, signal: float | Decimal | str):
        return await self.metric_stream.publish(
            "alpha_signal",
            json.dumps(
                {
                    "alpha_id": alpha_id,
                    "signal": str(signal),
                    "timestamp": time.time_ns(),
                }
            ).encode(),
            use_jetstream=True,
        )

    async def create_portfolio_signal(
        self, portfolio_id: str, signals: dict[str, Decimal]
    ):
        for asset, signal in signals.items():
            await self.metric_stream.publish(
                "portfolio_signal",
                json.dumps(
                    {
                        "portfolio_id": portfolio_id,
                        "asset": asset,
                        "signal": str(signal),
                        "timestamp": time.time_ns(),
                    }
                ).encode(),
                use_jetstream=True,
            )

    async def create_alpha_alert(
        self,
        alpha_id: str,
        title: str,
        description: str = "",
        priority: int = 3,
    ):
        return await self.metric_stream.publish(
            "alpha_alert",
            json.dumps(
                {
                    "alpha_id": alpha_id,
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "timestamp": time.time_ns(),
                }
            ).encode(),
            use_jetstream=True,
        )

    async def create_portfolio_alert(
        self,
        portfolio_id: str,
        title: str,
        description: str = "",
        priority: int = 3,
    ):
        return await self.metric_stream.publish(
            "portfolio_alert",
            json.dumps(
                {
                    "portfolio_id": portfolio_id,
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "timestamp": time.time_ns(),
                }
            ).encode(),
            use_jetstream=True,
        )

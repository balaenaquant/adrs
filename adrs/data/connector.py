import json
import time

from decimal import Decimal
from typing import Protocol, AsyncIterator, Callable, Awaitable
from nats_client import Msg

from adrs.types import Topic, Message


class MetricStream(Protocol):
    async def subscribe(
        self, subject: str, callback: Callable[[Msg], Awaitable[None]] | None
    ): ...

    async def publish(self, subject: str, payload: bytes, **kwargs) -> None: ...


class DatasourceStream(Protocol):
    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None: ...


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

    async def create_trade(
        self,
        oms_id: str,
        package_id: str,
        client_order_id: str,
        asset: str,
        symbol: str,
        exchange: str,
        executed_quantity: str,
        executed_price: str,
        executed_time: int,
        start_quantity: str,
        start_price: str,
        start_time: int,
    ):
        return await self.metric_stream.publish(
            "trade",
            json.dumps(
                {
                    "oms_id": oms_id,
                    "package_id": package_id,
                    "client_order_id": client_order_id,
                    "asset": asset,
                    "symbol": symbol,
                    "exchange": exchange,
                    "executed_quantity": executed_quantity,
                    "executed_price": executed_price,
                    "executed_time": executed_time,
                    "start_quantity": start_quantity,
                    "start_price": start_price,
                    "start_time": start_time,
                }
            ).encode(),
            use_jetstream=True,
        )

    async def create_position(
        self,
        oms_id: str,
        asset: str,
        symbol: str,
        exchange: str,
        quantity: str,
        price: str,
        updated_time: int,
    ):
        return await self.metric_stream.publish(
            "position",
            json.dumps(
                {
                    "oms_id": oms_id,
                    "asset": asset,
                    "symbol": symbol,
                    "exchange": exchange,
                    "quantity": quantity,
                    "price": price,
                    "updated_time": updated_time,
                    "timestamp": time.time_ns(),
                }
            ).encode(),
            use_jetstream=True,
        )

    async def create_equity(
        self,
        oms_id: str,
        equity: str,
    ):
        return (
            await self.metric_stream.publish(
                "equity",
                json.dumps(
                    {
                        "oms_id": oms_id,
                        "equity": equity,
                        "timestamp": time.time_ns(),
                    }
                ).encode(),
                use_jetstream=True,
            ),
        )

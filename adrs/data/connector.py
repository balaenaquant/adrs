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


DEFAULT_METRIC_NAMESPACE = "public_ts"


class MetricBuilder:
    """Publishes dashboard metrics under a namespace token (the JetStream root
    the dashboard consumes, e.g. `public_ts`)."""

    def __init__(
        self,
        metric_stream: MetricStream,
        insert_prefix: str = DEFAULT_METRIC_NAMESPACE,
    ):
        self.metric_stream = metric_stream
        self.insert_prefix = insert_prefix

    def _metric_subject(self, name: str) -> str:
        return f"{self.insert_prefix}.{name}"

    @staticmethod
    def _dedup_headers(msg_id: str) -> dict[str, str]:
        """
        Nats-Msg-Id for JetStream's server-side dedup: if two OMS replicas
        (e.g. old + new pod overlapping during a rolling deploy) publish the
        same logical record, the server collapses the second one instead of
        it landing twice downstream.
        """
        return {"Nats-Msg-Id": msg_id}

    async def create_alpha_trigger(self, alpha_id: str, topic: str | None = None):
        return await self.metric_stream.publish(
            self._metric_subject("alpha_trigger"),
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
            self._metric_subject("alpha_signal"),
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
                self._metric_subject("portfolio_signal"),
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
            self._metric_subject("alpha_alert"),
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
            self._metric_subject("portfolio_alert"),
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
            self._metric_subject("trade"),
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
            # No time-bucketing: a trade for one client_order_id is a
            # one-time event, so the key never needs to expire.
            headers=self._dedup_headers(f"trade:{oms_id}:{client_order_id}"),
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
        # Wall-clock minute bucket, not `updated_time`: the exchange only
        # bumps updated_time when the position actually changes, so keying
        # on it would collapse every legitimate heartbeat tick for a
        # position that's simply flat - breaking staleness monitoring
        # downstream. Bucketing on the publish minute only collapses
        # genuine same-tick duplicates (e.g. two replicas overlapping during
        # a rolling deploy) and still lets each new minute's tick through.
        minute_bucket = int(time.time() // 60)
        return await self.metric_stream.publish(
            self._metric_subject("position"),
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
            headers=self._dedup_headers(f"position:{oms_id}:{symbol}:{minute_bucket}"),
        )

    async def create_equity(
        self,
        oms_id: str,
        equity: str,
    ):
        # See create_position for why this buckets on the publish minute
        # rather than any field in the payload.
        minute_bucket = int(time.time() // 60)
        return (
            await self.metric_stream.publish(
                self._metric_subject("equity"),
                json.dumps(
                    {
                        "oms_id": oms_id,
                        "equity": equity,
                        "timestamp": time.time_ns(),
                    }
                ).encode(),
                use_jetstream=True,
                headers=self._dedup_headers(f"equity:{oms_id}:{minute_bucket}"),
            ),
        )

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

from nats.aio.msg import Msg
from typing import Protocol, AsyncIterator, Callable, Awaitable

from adrs.types import Topic, Message


class Database(Protocol):
    async def subscribe(
        self, subject: str, callback: Callable[[Msg], Awaitable[None]] | None
    ): ...

    async def async_publish(self, subject: str, payload: bytes) -> None: ...

    def publish(self, subject: str, payload: bytes) -> None: ...


class Stream(Protocol):
    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None: ...

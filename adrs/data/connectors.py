from typing import Protocol, AsyncIterator

from adrs.types import Topic, Message


class Database(Protocol):
    async def insert(self, subject: str, payload: bytes) -> None: ...


class Stream(Protocol):
    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None: ...

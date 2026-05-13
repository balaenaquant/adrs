from typing import Protocol, AsyncIterator

from adrs.types import Topic, Message


class Stream(Protocol):
    async def connect(self, topics: list[Topic]) -> AsyncIterator[Message] | None: ...

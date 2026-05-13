from typing import Protocol


class Database(Protocol):
    async def insert(self, title: str, datas: list[dict]) -> None: ...

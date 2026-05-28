from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator

from rich.progress import (
    BarColumn,  # noqa: F401 — used in Task 3
    Progress,
    SpinnerColumn,  # noqa: F401 — used in Task 3
    TaskID,
    TaskProgressColumn,  # noqa: F401 — used in Task 3
    TextColumn,  # noqa: F401 — used in Task 3
    TimeRemainingColumn,  # noqa: F401 — used in Task 3
)


_progress_var: ContextVar["Progress | None"] = ContextVar("adrs_progress", default=None)


@dataclass
class _NoOpBar:
    completed: int = 0
    total: int | None = None

    def advance(self, n: int) -> None:
        return None


@dataclass
class _InnerBar:
    progress: Progress
    task_id: TaskID
    completed: int = 0
    total: int | None = None

    def advance(self, n: int) -> None:
        self.progress.advance(self.task_id, n)
        self.completed += n


@asynccontextmanager
async def inner_task(
    description: str, total: int | None
) -> AsyncIterator[_NoOpBar | _InnerBar]:
    progress = _progress_var.get()
    if progress is None:
        yield _NoOpBar(completed=0, total=None)
        return

    task_id = progress.add_task(description, total=total)
    bar = _InnerBar(progress=progress, task_id=task_id, completed=0, total=total)
    try:
        yield bar
    finally:
        progress.remove_task(task_id)

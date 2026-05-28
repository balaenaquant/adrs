from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from adrs.console import console


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


@dataclass
class _OuterBar:
    progress: Progress
    task_id: TaskID
    completed: int = 0
    total: int = 0

    def advance(self, n: int = 1) -> None:
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


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not sys.stdout.isatty(),
    )


@asynccontextmanager
async def progress_context(total_topics: int) -> AsyncIterator[_OuterBar]:
    progress = _make_progress()
    progress.start()
    token = _progress_var.set(progress)
    outer_id = progress.add_task("topics", total=total_topics)
    outer = _OuterBar(
        progress=progress, task_id=outer_id, completed=0, total=total_topics
    )
    try:
        yield outer
    finally:
        _progress_var.reset(token)
        progress.stop()

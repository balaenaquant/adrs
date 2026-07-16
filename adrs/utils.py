import math
import polars as pl
from typing import cast
from datetime import datetime, timedelta


def infer_interval(times: pl.Series) -> timedelta:
    """Infer the sampling interval of a timestamp series.

    Uses the most common spacing between consecutive timestamps (ties broken
    by the smallest), which is robust to occasional gaps — unlike taking the
    last diff, which returns the gap size if the series happens to end right
    after a hole in the data."""
    diffs = times.diff().drop_nulls()
    if diffs.is_empty():
        raise ValueError(
            "need at least two timestamps to infer an interval from the series"
        )
    return cast(timedelta, diffs.mode().min())


def backforward_split(
    start_time: datetime,
    end_time: datetime,
    size: tuple[float, float] | None = None,
    forward_days: int | None = None,
) -> tuple[datetime, datetime, datetime, datetime]:
    duration = end_time - start_time

    if size is not None:
        if size[0] + size[1] != 1.0:
            raise ValueError(f"Size must sum to 1.0, not {size[0] + size[1]}")

        back = timedelta(days=math.ceil(duration.days * size[0]))
        return (start_time, start_time + back, start_time + back, end_time)

    if forward_days is not None:
        if forward_days < 0:
            raise ValueError("forward_days must be non-negative")
        if forward_days > duration.days:
            raise ValueError(
                f"forward_days ({forward_days}) cannot be greater than the total duration ({duration.days} days)"
            )

        return (
            start_time,
            end_time - timedelta(days=forward_days),
            end_time - timedelta(days=forward_days),
            end_time,
        )

    raise ValueError(
        "Either size or forward_days must be provided to split the time range."
    )

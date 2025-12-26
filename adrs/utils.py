import math
from datetime import datetime, timedelta


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

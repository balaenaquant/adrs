import pytest
from datetime import datetime
from adrs.utils import backforward_split


def test_backforward_split():
    start_time, end_time = (
        datetime.fromisoformat("2020-05-11T00:00:00Z"),
        datetime.fromisoformat("2025-01-01T00:00:00Z"),
    )

    # does not provide size / forward_days
    with pytest.raises(ValueError):
        backforward_split(
            start_time=start_time,
            end_time=end_time,
        )

    # does not add up to 1.0
    with pytest.raises(ValueError):
        backforward_split(start_time=start_time, end_time=end_time, size=(0.1, 0.2))

    # forward_days is negative
    with pytest.raises(ValueError):
        backforward_split(start_time=start_time, end_time=end_time, forward_days=-1)

    # forward_days is greater than the total duration
    with pytest.raises(ValueError):
        backforward_split(start_time=start_time, end_time=end_time, forward_days=2000)

    # 75% backtest, 25% forward test
    B_start, B_end, F_start, F_end = backforward_split(
        start_time=start_time, end_time=end_time, size=(0.75, 0.25)
    )
    assert B_start == start_time
    assert B_end == datetime.fromisoformat("2023-11-04T00:00:00Z")
    assert F_start == datetime.fromisoformat("2023-11-04T00:00:00Z")
    assert F_end == end_time

    # use the last 365 days for forward test
    B_start, B_end, F_start, F_end = backforward_split(
        start_time=start_time, end_time=end_time, forward_days=365
    )
    assert B_start == start_time
    assert B_end == datetime.fromisoformat("2024-01-02T00:00:00Z")
    assert F_start == datetime.fromisoformat("2024-01-02T00:00:00Z")
    assert F_end == end_time

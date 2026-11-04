"""A cached day file counts as downloaded only once it is final: written at
least ``final_after`` after its UTC day ended. Earlier files were fetched while
the day was open and can be missing bars, so they are downloaded again."""

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from adrs.data.cache import Cache, is_final
from adrs.data.utils import make_filepath
from adrs.types import Topic

TOPIC = Topic.from_str("bybit-linear|candle?symbol=BTCUSDT&interval=1h")
GRACE = timedelta(hours=6)


def _write_day(root: Path, day: date, written_at: datetime) -> Path:
    path = root / make_filepath(TOPIC, "parquet", suffix=day.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"start_time": [datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)]}).write_parquet(path)
    ts = written_at.timestamp()
    os.utime(path, (ts, ts))
    return path


def _utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)


def test_is_final_needs_the_grace_period_after_day_end(tmp_path):
    day = date(2026, 10, 2)
    assert not is_final(_write_day(tmp_path, day, _utc(2026, 10, 2, 5)), day, GRACE)
    assert not is_final(_write_day(tmp_path, day, _utc(2026, 10, 3, 2)), day, GRACE)
    assert is_final(_write_day(tmp_path, day, _utc(2026, 10, 3, 6)), day, GRACE)


def test_complete_days_are_not_downloaded_again(tmp_path):
    for d in range(1, 4):
        _write_day(tmp_path, date(2026, 10, d), _utc(2026, 10, d + 1, 12))
    cache = Cache(tmp_path, "parquet", override_existing=False, final_after=GRACE)
    start, end = cache.init(TOPIC, _utc(2026, 10, 1), _utc(2026, 10, 4))
    assert start >= end  # nothing to download


def test_partial_latest_day_is_downloaded_again(tmp_path):
    _write_day(tmp_path, date(2026, 10, 1), _utc(2026, 10, 2, 12))
    _write_day(tmp_path, date(2026, 10, 2), _utc(2026, 10, 2, 5))  # fetched mid-day
    cache = Cache(tmp_path, "parquet", override_existing=False, final_after=GRACE)
    start, end = cache.init(TOPIC, _utc(2026, 10, 1), _utc(2026, 10, 3))
    assert (start, end) == (_utc(2026, 10, 2), _utc(2026, 10, 3))


def test_partial_day_stranded_behind_newer_days_is_downloaded_again(tmp_path):
    # The 2026-10-02 incident: a mid-day file for 10-02, then later runs cached
    # 10-03 and 10-04 without ever touching 10-02 again.
    _write_day(tmp_path, date(2026, 10, 2), _utc(2026, 10, 2, 5))
    _write_day(tmp_path, date(2026, 10, 3), _utc(2026, 10, 4, 12))
    _write_day(tmp_path, date(2026, 10, 4), _utc(2026, 10, 5, 12))
    cache = Cache(tmp_path, "parquet", override_existing=False, final_after=GRACE)
    start, end = cache.init(TOPIC, _utc(2026, 10, 2), _utc(2026, 10, 5))
    assert (start, end) == (_utc(2026, 10, 2), _utc(2026, 10, 3))

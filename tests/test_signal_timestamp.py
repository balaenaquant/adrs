"""The emitted signal is stamped with the time of ITS OWN bar, not the wall
clock at emit time.

The emit can lag the bar close (ws delivery, resync retries) and the periodic
resync can re-emit an older bar, so a `time.time_ns()` stamp drifts away from
the candle the signal was actually computed on.
"""

import asyncio
import json

import pytest
from datetime import datetime, timezone
from types import SimpleNamespace

import polars as pl

from adrs.execution.executor import AlphaExecutor, _latest_signal

BAR = datetime(2020, 1, 2, 12, 0, tzinfo=timezone.utc)
BAR_NS = int(BAR.timestamp()) * 1_000_000_000


class _RecordingAegis:
    def __init__(self):
        self.signals = []
        self.payloads = []
        self.metric_stream = SimpleNamespace(publish=self._publish)

    async def create_alpha_signal(self, alpha_id, signal, timestamp):
        self.signals.append((alpha_id, signal, timestamp))

    async def _publish(self, subject, payload, **kwargs):
        self.payloads.append(json.loads(payload.decode()))


def _alpha(signal_df):
    return SimpleNamespace(
        id="marcus_a",
        data_infos=[],
        data_processor=SimpleNamespace(
            process=lambda datamap, last_closed_time: pl.DataFrame({"x": [1.0]})
        ),
        next=lambda df: signal_df,
    )


def _emit(signal_df):
    ex = object.__new__(AlphaExecutor)
    ex.signal_namespace = "marcus"
    ex.datamap = object()
    ex.aegis = _RecordingAegis()
    emitted = asyncio.run(ex._emit_signal(_alpha(signal_df), BAR))
    assert emitted is True
    return ex.aegis


def test_latest_signal_reads_last_row():
    df = pl.DataFrame(
        {
            "start_time": [BAR.replace(hour=11), BAR],
            "signal": [0.1, 0.9],
        }
    )
    # value and bar time come from the same (last) row
    assert _latest_signal(df) == ("0.90", BAR_NS)


def test_latest_signal_treats_naive_as_utc():
    df = pl.DataFrame({"start_time": [BAR.replace(tzinfo=None)], "signal": [0.9]})
    assert _latest_signal(df).bar_time_ns == BAR_NS


def test_latest_signal_keeps_sub_ms_precision():
    df = pl.DataFrame(
        {
            "start_time": pl.Series(
                [BAR.replace(microsecond=1)], dtype=pl.Datetime("us", "UTC")
            ),
            "signal": [0.9],
        }
    )
    assert _latest_signal(df).bar_time_ns == BAR_NS + 1_000


def test_latest_signal_reports_every_missing_column():
    with pytest.raises(ValueError, match=r"start_time.*signal"):
        _latest_signal(pl.DataFrame({"close": [1.0]}))


def test_latest_signal_rejects_missing_start_time():
    # every signal belongs to a bar; no start_time means a broken `next()`
    with pytest.raises(ValueError, match="start_time"):
        _latest_signal(pl.DataFrame({"signal": [0.9]}))


def test_latest_signal_rejects_missing_signal():
    with pytest.raises(ValueError, match="signal"):
        _latest_signal(pl.DataFrame({"start_time": [BAR]}))


def test_latest_signal_rejects_non_numeric_signal():
    df = pl.DataFrame({"start_time": [BAR], "signal": ["0.9"]})
    with pytest.raises(ValueError, match="numeric"):
        _latest_signal(df)


def test_latest_signal_rejects_null_start_time():
    df = pl.DataFrame(
        {"start_time": pl.Series([None], dtype=pl.Datetime("ms")), "signal": [0.9]}
    )
    with pytest.raises(ValueError, match="start_time"):
        _latest_signal(df)


def test_latest_signal_rejects_null_signal():
    # the old numpy path turned this into a published "nan"
    df = pl.DataFrame(
        {"start_time": [BAR], "signal": pl.Series([None], dtype=pl.Float64)}
    )
    with pytest.raises(ValueError, match="signal"):
        _latest_signal(df)


def test_latest_signal_accepts_integer_signal():
    # alphas that branch with `.then(1)` yield an int column
    df = pl.DataFrame({"start_time": [BAR], "signal": pl.Series([1], dtype=pl.Int64)})
    assert _latest_signal(df) == ("1.00", BAR_NS)


def test_emit_stamps_metric_and_payload_with_bar_time():
    df = pl.DataFrame(
        {
            "start_time": [BAR.replace(hour=11), BAR],
            "signal": [0.1, 0.9],
        }
    )
    aegis = _emit(df)
    assert aegis.signals == [("marcus_a", "0.90", BAR_NS)]
    assert aegis.payloads == [{"signal": "0.90", "timestamp": BAR_NS}]


@pytest.mark.parametrize(
    "signal_df",
    [
        pl.DataFrame({"signal": [0.9]}),  # undated
        pl.DataFrame({"start_time": [BAR]}),  # no signal
        pl.DataFrame(
            {"start_time": [BAR], "signal": pl.Series([None], dtype=pl.Float64)}
        ),
    ],
    ids=["no_start_time", "no_signal", "null_signal"],
)
def test_emit_publishes_nothing_when_signal_df_is_malformed(signal_df):
    # better to fail loudly (the WS/resync callers alert on it) than to publish
    # a signal that is mis-dated or not a number
    ex = object.__new__(AlphaExecutor)
    ex.signal_namespace = "marcus"
    ex.datamap = object()
    ex.aegis = _RecordingAegis()
    with pytest.raises(ValueError):
        asyncio.run(ex._emit_signal(_alpha(signal_df), BAR))
    assert ex.aegis.signals == []
    assert ex.aegis.payloads == []

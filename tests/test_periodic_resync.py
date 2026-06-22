"""AlphaExecutor's REST resync fallback: re-trigger alphas on a timer so signals
keep flowing when the flow-NATS WS feed is unreliable, with per-alpha isolation.
"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import polars as pl

from adrs.execution.executor import AlphaExecutor

TOPIC = "bybit-linear|candle?symbol=BTCUSDT&interval=1h"


def _new_executor():
    """AlphaExecutor with __init__ skipped — set only what the resync path uses."""
    ex = object.__new__(AlphaExecutor)
    ex.signal_namespace = "marcus"
    return ex


class _FakeDatamap:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0

    async def resync(self, topic, dataloader):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("flow NATS / REST hiccup")


class _RecordingAegis:
    def __init__(self):
        self.signals = []
        self.published = []
        self.metric_stream = SimpleNamespace(publish=self._publish)

    async def create_alpha_signal(self, alpha_id, signal):
        self.signals.append((alpha_id, signal))

    async def _publish(self, subject, payload, **kwargs):
        self.published.append(subject)


def _alpha(alpha_id, *, raises=False):
    def process(datamap, last_closed_time):
        if raises:
            raise RuntimeError("processor blew up")
        return pl.DataFrame({"signal": [1.0]})

    return SimpleNamespace(
        id=alpha_id,
        data_infos=[SimpleNamespace(topic=TOPIC)],
        data_processor=SimpleNamespace(process=process),
        next=lambda df: pl.DataFrame({"signal": [1.0]}),
    )


def test_resync_retries_then_succeeds():
    ex = _new_executor()
    ex.datamap = _FakeDatamap(fail_times=2)
    ex.dataloader = object()
    ok = asyncio.run(ex._resync_alpha(_alpha("a"), retry_delay=timedelta(0)))
    assert ok is True
    assert ex.datamap.calls == 3  # 2 failures + 1 success


def test_resync_gives_up_after_max_retries():
    ex = _new_executor()
    ex.datamap = _FakeDatamap(fail_times=99)
    ex.dataloader = object()
    ok = asyncio.run(
        ex._resync_alpha(_alpha("a"), max_retries=3, retry_delay=timedelta(0))
    )
    assert ok is False
    assert ex.datamap.calls == 3


def test_periodic_resync_emits_for_all_alphas():
    ex = _new_executor()
    ex.datamap = _FakeDatamap()
    ex.dataloader = object()
    ex.aegis = _RecordingAegis()
    ex.alphas = [_alpha("marcus_a1"), _alpha("marcus_a2")]
    asyncio.run(ex.on_periodic_resync())
    # both published under the namespaced subject
    assert ex.aegis.published == [
        "alpha_signal.marcus.marcus_a1",
        "alpha_signal.marcus.marcus_a2",
    ]


def test_periodic_resync_isolates_one_failure():
    ex = _new_executor()
    ex.datamap = _FakeDatamap()
    ex.dataloader = object()
    ex.aegis = _RecordingAegis()
    # a1 blows up during processing; a2 must still emit
    ex.alphas = [_alpha("marcus_a1", raises=True), _alpha("marcus_a2")]
    asyncio.run(ex.on_periodic_resync())
    assert ex.aegis.published == ["alpha_signal.marcus.marcus_a2"]

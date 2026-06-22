"""MetricBuilder namespaces dashboard inserts; the transport stays verbatim.

Regression guard for the prefix refactor: PublicMetricStream no longer rewrites
subjects, so the namespace must come from MetricBuilder and routing publishes
must reach the wire unchanged.
"""

import asyncio

from adrs.data.connector import DEFAULT_METRIC_NAMESPACE, MetricBuilder


class _RecordingStream:
    def __init__(self):
        self.published: list[tuple[str, bool]] = []

    async def publish(self, subject, payload, **kwargs):
        self.published.append((subject, kwargs.get("use_jetstream", False)))

    async def subscribe(self, subject, callback):  # pragma: no cover - unused
        pass


def test_default_namespace():
    s = _RecordingStream()
    mb = MetricBuilder(s)
    assert mb.insert_prefix == DEFAULT_METRIC_NAMESPACE == "public_ts"
    asyncio.run(mb.create_alpha_signal("marcus_a", 1))
    assert s.published == [("public_ts.alpha_signal", True)]


def test_custom_namespace_prefixes_all_metrics():
    s = _RecordingStream()
    mb = MetricBuilder(s, insert_prefix="aegis_ts")
    asyncio.run(mb.create_alpha_signal("a", 1))
    asyncio.run(mb.create_portfolio_alert("p", "t"))
    asyncio.run(
        mb.create_trade(
            "o", "pk", "c", "BTC", "BTCUSDT", "bybit", "1", "2", 3, "4", "5", 6
        )
    )
    subjects = [sub for sub, _ in s.published]
    assert subjects == [
        "aegis_ts.alpha_signal",
        "aegis_ts.portfolio_alert",
        "aegis_ts.trade",
    ]
    # dashboard inserts always go through jetstream
    assert all(js for _, js in s.published)

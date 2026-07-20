"""MetricBuilder namespaces dashboard inserts; the transport stays verbatim.

Regression guard for the prefix refactor: PublicMetricStream no longer rewrites
subjects, so the namespace must come from MetricBuilder and routing publishes
must reach the wire unchanged.
"""

import asyncio
from unittest.mock import patch

from adrs.data.connector import DEFAULT_METRIC_NAMESPACE, MetricBuilder


class _RecordingStream:
    def __init__(self):
        self.published: list[tuple[str, bool]] = []
        self.headers: list[dict | None] = []

    async def publish(self, subject, payload, **kwargs):
        self.published.append((subject, kwargs.get("use_jetstream", False)))
        self.headers.append(kwargs.get("headers"))

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


# ---------------------------------------------------------------------------
# Nats-Msg-Id dedup headers
#
# Two OMS replicas overlapping during a rolling deploy can each publish the
# "same" position/equity/trade record for the same tick; these keys let
# JetStream's server-side dedup collapse the second one.
# ---------------------------------------------------------------------------


def test_create_position_dedup_key_stable_within_same_minute():
    s = _RecordingStream()
    mb = MetricBuilder(s)
    with patch("adrs.data.connector.time.time", return_value=1_000_000.0):
        asyncio.run(
            mb.create_position("o", "BTC", "BTCUSDT", "bybit", "1", "50000", 123)
        )
        asyncio.run(
            mb.create_position("o", "BTC", "BTCUSDT", "bybit", "1", "50000", 123)
        )
    assert s.headers[0] == s.headers[1] == {"Nats-Msg-Id": "position:o:BTCUSDT:16666"}


def test_create_position_dedup_key_differs_across_minutes():
    s = _RecordingStream()
    mb = MetricBuilder(s)
    with patch("adrs.data.connector.time.time", return_value=1_000_000.0):
        asyncio.run(
            mb.create_position("o", "BTC", "BTCUSDT", "bybit", "1", "50000", 123)
        )
    with patch("adrs.data.connector.time.time", return_value=1_000_100.0):
        asyncio.run(
            mb.create_position("o", "BTC", "BTCUSDT", "bybit", "1", "50000", 123)
        )
    assert s.headers[0] != s.headers[1]


def test_create_position_dedup_key_differs_across_symbols():
    s = _RecordingStream()
    mb = MetricBuilder(s)
    with patch("adrs.data.connector.time.time", return_value=1_000_000.0):
        asyncio.run(
            mb.create_position("o", "BTC", "BTCUSDT", "bybit", "1", "50000", 123)
        )
        asyncio.run(
            mb.create_position("o", "ETH", "ETHUSDT", "bybit", "1", "3000", 123)
        )
    assert s.headers[0] != s.headers[1]


def test_create_equity_dedup_key_stable_within_same_minute():
    s = _RecordingStream()
    mb = MetricBuilder(s)
    with patch("adrs.data.connector.time.time", return_value=1_000_000.0):
        asyncio.run(mb.create_equity("o", "1000"))
        asyncio.run(mb.create_equity("o", "1000"))
    assert s.headers[0] == s.headers[1] == {"Nats-Msg-Id": "equity:o:16666"}


def test_create_trade_dedup_key_is_permanent_per_client_order_id():
    s = _RecordingStream()
    mb = MetricBuilder(s)
    asyncio.run(
        mb.create_trade(
            "o", "pk", "c1", "BTC", "BTCUSDT", "bybit", "1", "2", 3, "4", "5", 6
        )
    )
    asyncio.run(
        mb.create_trade(
            "o", "pk", "c1", "BTC", "BTCUSDT", "bybit", "1", "2", 3, "4", "5", 6
        )
    )
    assert s.headers[0] == s.headers[1] == {"Nats-Msg-Id": "trade:o:c1"}

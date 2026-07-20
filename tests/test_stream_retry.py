"""Transport streams retry transient gRPC errors (JetStream rate-limiting →
gRPC UNAVAILABLE / HTTP 429) so a single hiccup doesn't crash the caller.

Covers PublicMetricStream.publish (NATS publish over gRPC) and
PublicNatsDatasourceStream.connect (NATS subscribe over gRPC). Non-retryable
codes (e.g. INVALID_ARGUMENT) must propagate immediately, not loop.
"""

import asyncio

import pytest
from grpc import StatusCode
from grpc.aio import AioRpcError, Metadata

from adrs.io.stream import PublicMetricStream, PublicNatsDatasourceStream


def _rpc_error(code: StatusCode) -> AioRpcError:
    return AioRpcError(code, Metadata(), Metadata(), details="rpc failed")


class _FakeNats:
    def __init__(self, fail_times=0, fail_code=StatusCode.UNAVAILABLE):
        self.fail_times = fail_times
        self.fail_code = fail_code
        self.js_calls = []
        self.core_calls = []
        self.sub_calls = []

    async def jetstream(self):
        pass

    async def js_publish(self, subject, payload, headers=None):
        self.js_calls.append((subject, payload, headers))
        if len(self.js_calls) <= self.fail_times:
            raise _rpc_error(self.fail_code)

    async def publish(self, subject, payload):
        self.core_calls.append((subject, payload))
        if len(self.core_calls) <= self.fail_times:
            raise _rpc_error(self.fail_code)

    async def subscribe(self, subject, cb):
        self.sub_calls.append(subject)
        if len(self.sub_calls) <= self.fail_times:
            raise _rpc_error(self.fail_code)


# --- PublicMetricStream.publish ---


def test_metric_jetstream_retries_then_succeeds():
    nats = _FakeNats(fail_times=2)  # UNAVAILABLE twice, then ok
    ms = PublicMetricStream(nats, retry_backoff=0)
    asyncio.run(ms.publish("public_ts.alpha_signal", b"x", use_jetstream=True))
    assert len(nats.js_calls) == 3  # 2 failures + 1 success


def test_metric_core_retries_then_succeeds():
    nats = _FakeNats(fail_times=1)
    ms = PublicMetricStream(nats, retry_backoff=0)
    asyncio.run(ms.publish("alpha_signal.id", b"y"))
    assert len(nats.core_calls) == 2


def test_metric_gives_up_after_max_retries():
    nats = _FakeNats(fail_times=99)
    ms = PublicMetricStream(nats, max_retries=2, retry_backoff=0)
    with pytest.raises(AioRpcError):
        asyncio.run(ms.publish("public_ts.alpha_signal", b"x", use_jetstream=True))
    assert len(nats.js_calls) == 3  # initial + 2 retries


def test_metric_non_retryable_reraised_immediately():
    nats = _FakeNats(fail_times=99, fail_code=StatusCode.INVALID_ARGUMENT)
    ms = PublicMetricStream(nats, max_retries=5, retry_backoff=0)
    with pytest.raises(AioRpcError):
        asyncio.run(ms.publish("public_ts.alpha_signal", b"x", use_jetstream=True))
    assert len(nats.js_calls) == 1  # not retried


def test_metric_jetstream_forwards_headers_for_dedup():
    nats = _FakeNats()
    ms = PublicMetricStream(nats, retry_backoff=0)
    asyncio.run(
        ms.publish(
            "public_ts.position",
            b"x",
            use_jetstream=True,
            headers={"Nats-Msg-Id": "position:o:BTCUSDT:123"},
        )
    )
    _, _, headers = nats.js_calls[0]
    assert headers == {"Nats-Msg-Id": "position:o:BTCUSDT:123"}


def test_metric_jetstream_omits_headers_when_caller_supplies_none():
    nats = _FakeNats()
    ms = PublicMetricStream(nats, retry_backoff=0)
    asyncio.run(ms.publish("public_ts.alpha_signal", b"x", use_jetstream=True))
    _, _, headers = nats.js_calls[0]
    assert headers is None


# --- PublicNatsDatasourceStream.connect (subscribe path) ---


class _Topic:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def test_datasource_subscribe_retries_then_succeeds():
    nats = _FakeNats(fail_times=2)  # UNAVAILABLE twice, then ok
    stream = PublicNatsDatasourceStream(nats, retry_backoff=0)
    result = asyncio.run(stream.connect([_Topic("kline.BTCUSDT")]))
    assert result is not None
    assert len(nats.sub_calls) == 3  # 2 failures + 1 success


def test_datasource_subscribe_gives_up_after_max_retries():
    nats = _FakeNats(fail_times=99)
    stream = PublicNatsDatasourceStream(nats, max_retries=2, retry_backoff=0)
    with pytest.raises(AioRpcError):
        asyncio.run(stream.connect([_Topic("kline.BTCUSDT")]))
    assert len(nats.sub_calls) == 3  # initial + 2 retries


def test_datasource_subscribe_non_retryable_reraised_immediately():
    nats = _FakeNats(fail_times=99, fail_code=StatusCode.INVALID_ARGUMENT)
    stream = PublicNatsDatasourceStream(nats, max_retries=5, retry_backoff=0)
    with pytest.raises(AioRpcError):
        asyncio.run(stream.connect([_Topic("kline.BTCUSDT")]))
    assert len(nats.sub_calls) == 1  # not retried

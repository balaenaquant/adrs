"""Tests for per-exchange error classification and the executor wiring.

`110001` ("order not exists / too late to cancel") must be treated as done so
the cancel leaves the backlog instead of being retried forever; unknown codes
must keep the legacy retry-via-backlog behaviour.
"""

import asyncio
from contextlib import asynccontextmanager

from cybotrade import Symbol
from cybotrade.bybit import BybitError
from cybotrade.binance import BinanceError

from adrs.oms.ops.order_executer import OrderExecutor
from adrs.oms.ops.order_pool import CancelBacklogs
from adrs.oms.rate_limit.error_policy import (
    ErrorAction,
    BybitErrorPolicy,
    BinanceErrorPolicy,
    DefaultErrorPolicy,
)
from adrs.oms.rate_limit.rate_limiter import LocalRateLimitError


# Native error getters can't be set directly, so subclass with property overrides
class FakeBybitError(BybitError):
    def __init__(self, retCode=None, http_status=None):
        self._rc = retCode
        self._hs = http_status

    @property
    def retCode(self):
        return self._rc

    @property
    def http_status(self):
        return self._hs


class FakeBinanceError(BinanceError):
    def __init__(self, code=None):
        self._code = code

    @property
    def code(self):
        return self._code


# --- pure classify ---------------------------------------------------------


def test_bybit_classify():
    p = BybitErrorPolicy()
    assert p.classify(FakeBybitError(retCode=110001)) is ErrorAction.TERMINAL_SUCCESS
    assert p.classify(FakeBybitError(retCode=10006)) is ErrorAction.RATE_LIMITED
    assert p.classify(FakeBybitError(http_status=403)) is ErrorAction.RATE_LIMITED
    assert p.classify(FakeBybitError(retCode=99999)) is ErrorAction.RETRY
    assert p.classify(LocalRateLimitError("x")) is ErrorAction.RETRY
    assert p.classify(RuntimeError("boom")) is ErrorAction.RETRY


def test_binance_classify():
    p = BinanceErrorPolicy()
    assert p.classify(FakeBinanceError(code=-2011)) is ErrorAction.TERMINAL_SUCCESS
    assert p.classify(FakeBinanceError(code=429)) is ErrorAction.RATE_LIMITED
    assert p.classify(FakeBinanceError(code=12345)) is ErrorAction.RETRY
    assert p.classify(RuntimeError("boom")) is ErrorAction.RETRY


def test_default_policy_always_retries():
    p = DefaultErrorPolicy()
    assert p.classify(FakeBybitError(retCode=110001)) is ErrorAction.RETRY


# --- cancel_single_order integration ---------------------------------------


class _StubRateLimiter:
    @asynccontextmanager
    async def guard(self, endpoint):
        yield


class _StubExchange:
    def __init__(self, exc):
        self._exc = exc

    async def cancel_order(self, symbol, client_order_id):
        raise self._exc


class _StubPools:
    def __init__(self, pool):
        self.pool = pool

    @asynccontextmanager
    async def get_order_pool(self):
        yield self.pool


def _executor(raised_exc, pool):
    ex = object.__new__(OrderExecutor)
    ex.error_policy = BybitErrorPolicy()
    ex.rate_limiter = _StubRateLimiter()
    ex.exchange = _StubExchange(raised_exc)
    ex.order_pools = _StubPools(pool)
    return ex


def test_cancel_terminal_success_pops_and_returns_none():
    pool = {"coid": object()}
    ex = _executor(FakeBybitError(retCode=110001), pool)
    result = asyncio.run(ex.cancel_single_order(Symbol("BTCUSDT"), "coid"))
    assert result is None  # _retry_one will treat as success -> remove backlog
    assert "coid" not in pool  # popped from the live pool


def test_cancel_unknown_code_backlogs_and_retains_pool():
    pool = {"coid": object()}
    ex = _executor(FakeBybitError(retCode=99999), pool)
    result = asyncio.run(ex.cancel_single_order(Symbol("BTCUSDT"), "coid"))
    assert isinstance(result, CancelBacklogs)  # unchanged: keep retrying
    assert "coid" in pool  # still tracked

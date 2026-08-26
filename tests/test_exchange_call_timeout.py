"""
Tests for the wall-clock ceiling guard()/reserve() puts on one exchange call.

These exist because of a production incident with no error in it. Over the
affected window the only OMS output was a single `[ON_AEGIS_UPDATE]`, the
per-tick "no significant change in signal weights" lines, and the occasional
"Portfolio signal retrieved" — while `[OMS STATUS CHECK]`, which
`on_order_placement` logs unconditionally before its first await, never appeared
again.

That combination is diagnostic. The event loop was alive (the 5s signal job kept
logging, and its no-change path touches no REST at all), but the two jobs that do
make REST calls stopped firing entirely. aion does not re-fire a job while its
previous run is still pending, so a call that never returns retires that job for
the life of the process: no exception, no log line, no further ticks.

Nothing below the limiter bounds those calls — cybotrade's HTTP client exposes no
timeout parameter and reqwest sets no read timeout by default — so a half-open
socket suspends the await indefinitely. The timeout lives in guard()/reserve()
because that is the one layer every exchange call already passes through.
"""

import asyncio
from collections import deque

import pytest

from adrs.oms.rate_limit.rate_limiter import (
    BinanceRateLimiter,
    BybitRateLimiter,
    EXCHANGE_CALL_TIMEOUT_SEC,
)
from adrs.oms.rate_limit.exchange_limit_profiles import (
    BinanceLimitProfile,
    BybitLimitProfile,
    BybitLimitState,
    BybitRateLimitPool,
    Endpoints,
)

NOW_MS = 1_700_000_000_000

# Short enough to keep the suite fast, long enough that a healthy call in these
# tests (which is instant) never trips it.
TEST_TIMEOUT_SEC = 0.05


def _binance() -> BinanceRateLimiter:
    lim = BinanceRateLimiter.__new__(BinanceRateLimiter)
    lim._reserve_locks = {}
    lim._waiters = {}
    lim.retry_after = 0
    lim.exchange_time_offset = 0
    lim.exchange = None
    lim.limit_profile = BinanceLimitProfile(
        request_weight_limit_per_minute=1920,
        order_limit_per_minute=960,
        order_limit_per_10_sec=240,
    )
    lim.current_limit_state = BinanceLimitProfile(
        request_weight_limit_per_minute=0,
        order_limit_per_minute=0,
        order_limit_per_10_sec=0,
    )
    lim.last_reset_10s_timestamp = NOW_MS // 10_000
    lim.last_reset_1m_timestamp = NOW_MS // 60_000
    lim._usage_headers = {}
    lim.call_timeout_sec = TEST_TIMEOUT_SEC
    return lim


def _bybit() -> BybitRateLimiter:
    lim = BybitRateLimiter.__new__(BybitRateLimiter)
    lim._reserve_locks = {}
    lim._waiters = {}
    lim.retry_after = 0
    lim.exchange_time_offset = 0
    lim.exchange = None
    lim.limit_profile = BybitLimitProfile(
        limits={pool: 10 for pool in BybitRateLimitPool}, interval=1
    )
    lim.current_limit_state = {
        pool: BybitLimitState(timestamps=deque()) for pool in BybitRateLimitPool
    }
    lim._now = NOW_MS
    lim.get_synced_time_ms = lambda: lim._now
    lim.call_timeout_sec = TEST_TIMEOUT_SEC
    return lim


async def _hangs_forever():
    """Stands in for a REST call on a half-open socket: no error, no return."""
    await asyncio.Event().wait()


def _seconds_until_it_gives_up(open_cm) -> float:
    """
    Run a hanging call inside `open_cm()` and return how long it took to raise.

    Returns elapsed rather than just asserting TimeoutError, because every
    caller here also needs an outer deadline to keep a failure from hanging the
    suite — and asserting only "TimeoutError was raised" would be satisfied by
    that outer deadline just as well as by the limiter's own. The elapsed time
    is what distinguishes them. Verified: deleting the timeout from
    BybitRateLimiter.guard makes these tests take the full outer 5s and fail.
    """

    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            async with open_cm():
                await _hangs_forever()
        except TimeoutError:
            return loop.time() - started
        raise AssertionError("the call did not time out")

    return asyncio.run(asyncio.wait_for(run(), timeout=5))


# --- the hang must become an exception, on the limiter's own deadline ------


def test_a_hung_call_under_binance_guard_gives_up_on_its_own_deadline():
    """
    The production shape. Without the timeout this coroutine never completes and
    the job that owns it never ticks again.
    """
    lim = _binance()
    elapsed = _seconds_until_it_gives_up(
        lambda: lim.guard(endpoint=Endpoints.GET_POSITION)
    )
    assert TEST_TIMEOUT_SEC <= elapsed < 1.0


def test_a_hung_call_under_bybit_guard_gives_up_on_its_own_deadline():
    lim = _bybit()
    elapsed = _seconds_until_it_gives_up(
        lambda: lim.guard(endpoint=Endpoints.GET_POSITION)
    )
    assert TEST_TIMEOUT_SEC <= elapsed < 1.0


def test_a_hung_call_under_reserve_gives_up_on_its_own_deadline():
    """
    reserve() is the path position.update_exchange and the open-orders snapshot
    take, so it needs the ceiling as much as guard() does. Its existing
    RESERVE_TIMEOUT_SEC bounds only the wait for a *slot*, never the call.
    """
    lim = _bybit()
    elapsed = _seconds_until_it_gives_up(
        lambda: lim.reserve(endpoint=Endpoints.GET_OPEN_ORDERS)
    )
    assert TEST_TIMEOUT_SEC <= elapsed < 1.0


# --- and must not disturb the healthy path --------------------------------


def test_a_normal_call_is_unaffected():
    lim = _binance()

    async def run():
        async with lim.guard(endpoint=Endpoints.GET_POSITION):
            await asyncio.sleep(0)
        return True

    assert asyncio.run(run()) is True


def test_a_body_exception_still_propagates_unchanged():
    """The timeout wraps the body; it must not swallow or reshape real errors."""
    lim = _binance()

    async def run():
        async with lim.guard(endpoint=Endpoints.GET_POSITION):
            raise ValueError("exchange said no")

    with pytest.raises(ValueError, match="exchange said no"):
        asyncio.run(run())


# --- a timeout must not wedge a pool --------------------------------------


def test_repeated_timeouts_do_not_permanently_wedge_a_uid_pool():
    """
    The d1d64d9 failure shape, reached through the new exit.

    guard() decrements the Bybit pool optimistically and only adopts the real
    quota from headers on success, so a failure that skips the refund walks the
    pool to zero with no path back — `_uid_pool_snapshot`'s reset guard cannot
    fire on an unset reset_ts. A timeout is not a rate-limit signal, so it must
    take the refund branch exactly as a dropped connection already does.

    Mutation check: deleting the timeout from BybitRateLimiter.guard fails this
    test. (Merely moving it outside the `try` does not — the CancelledError is
    still delivered at the `yield` inside, so `except BaseException` catches it
    and refunds either way.)
    """
    lim = _bybit()
    pool = BybitRateLimitPool.UID_POSITION
    ceiling = lim.limit_profile.limits[pool]

    async def _tick():
        lim._now += 1_100  # roll IP_GLOBAL's 1s window; isolate the UID pool
        try:
            async with lim.guard(endpoint=Endpoints.GET_POSITION):
                await _hangs_forever()
        except TimeoutError:
            pass

    for _ in range(ceiling * 2 + 5):
        asyncio.run(asyncio.wait_for(_tick(), timeout=5))

    # Nothing here was Bybit saying no, so the pool must still admit calls.
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


def test_the_default_ceiling_is_well_clear_of_a_healthy_call():
    """
    Guards the constant itself. It has to sit above any real REST round trip and
    below the cadence of the jobs making them (on_order_placement every 15s,
    on_aegis_update every 60s) so a timeout costs one tick, not the job.
    """
    assert 1.0 < EXCHANGE_CALL_TIMEOUT_SEC < 15.0

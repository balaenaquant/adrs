"""Tests for RateLimiter.reserve() — the waiting/priority acquire path.

Uses BybitRateLimiter with __init__ skipped and state pre-populated, so the
behaviour is driven directly without a live exchange. IP_GLOBAL still uses a
local rolling window (Bybit has no header for the account-wide IP limit);
UID_* pools are driven by remaining/limit/reset_ts, mirroring what a real
X-Bapi-Limit / X-Bapi-Limit-Status / X-Bapi-Limit-Reset-Timestamp response
would set via _reconcile_uid_pool().
"""

import asyncio
from collections import deque

import pytest

from adrs.oms.rate_limit import rate_limiter as rl
from adrs.oms.rate_limit.rate_limiter import (
    BybitRateLimiter,
    LocalRateLimitError,
)
from adrs.oms.rate_limit.exchange_limit_profiles import (
    BybitLimitProfile,
    BybitLimitState,
    BybitRateLimitPool,
    Endpoints,
)


def _bybit(*, limits=None) -> BybitRateLimiter:
    """BybitRateLimiter with __init__ skipped and a controllable clock."""
    lim = BybitRateLimiter.__new__(BybitRateLimiter)
    lim._reserve_locks = {}
    lim._waiters = {}
    lim.retry_after = 0
    lim.exchange_time_offset = 0
    lim.exchange = None  # no live client; _on_call_success no-ops without headers

    default = {pool: 2 for pool in BybitRateLimitPool}
    lim.limit_profile = BybitLimitProfile(limits=limits or default, interval=1)
    lim.current_limit_state = {
        pool: BybitLimitState(timestamps=deque()) for pool in BybitRateLimitPool
    }
    # Fixed clock so window maths are deterministic; tests set it as needed
    lim._now = 1_000_000
    lim.get_synced_time_ms = lambda: lim._now
    return lim


def test_reserve_succeeds_immediately_when_capacity_free():
    lim = _bybit()
    ep = Endpoints.GET_OPEN_ORDERS

    async def run():
        async with lim.reserve(endpoint=ep):
            pass

    asyncio.run(run())
    # IP_GLOBAL still a local window: one slot recorded
    assert len(lim.current_limit_state[BybitRateLimitPool.IP_GLOBAL].timestamps) == 1
    # UID pool has no header yet (bootstrap = full profile limit), then one
    # optimistic decrement from record_usage: limit 2 -> remaining 1
    assert lim.current_limit_state[BybitRateLimitPool.UID_OPEN_ORDERS].remaining == 1


def test_reserve_instant_fail_under_retry_after():
    lim = _bybit()
    lim.retry_after = lim._now + 10_000  # cooling down

    async def run():
        async with lim.reserve(endpoint=Endpoints.GET_OPEN_ORDERS):
            pass

    with pytest.raises(LocalRateLimitError):
        asyncio.run(run())


def test_reserve_times_out_when_slot_never_frees(monkeypatch):
    monkeypatch.setattr(rl, "RESERVE_TIMEOUT_SEC", 0.05)
    lim = _bybit()
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    # Pool exhausted per the exchange's own headers, resetting well after
    # the reserve timeout — so it never frees during this test.
    lim.current_limit_state[pool].remaining = 0
    lim.current_limit_state[pool].limit = 2
    lim.current_limit_state[pool].reset_ts = lim._now + 10_000

    async def run():
        async with lim.reserve(endpoint=Endpoints.GET_OPEN_ORDERS):
            pass

    with pytest.raises(LocalRateLimitError):
        asyncio.run(run())


def test_reserve_waits_then_acquires_when_slot_frees(monkeypatch):
    monkeypatch.setattr(rl, "RESERVE_TIMEOUT_SEC", 2.0)
    # Recheck quickly instead of sleeping a full window
    monkeypatch.setattr(BybitRateLimiter, "_next_free_delay", lambda self, ep: 0.01)
    lim = _bybit()
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    lim.current_limit_state[pool].remaining = 0
    lim.current_limit_state[pool].limit = 2
    lim.current_limit_state[pool].reset_ts = lim._now + 10_000

    async def run():
        async def free_slot_soon():
            await asyncio.sleep(0.05)
            # Simulate a fresh response reconciling capacity back to full
            lim.current_limit_state[pool].remaining = 1

        freer = asyncio.create_task(free_slot_soon())
        async with lim.reserve(endpoint=Endpoints.GET_OPEN_ORDERS):
            pass
        await freer

    asyncio.run(run())
    # Acquired: freed to 1, then reserve's record_usage decremented it back to 0
    assert lim.current_limit_state[pool].remaining == 0


def test_guard_yields_while_reserver_queued():
    lim = _bybit()
    ep = Endpoints.GET_OPEN_ORDERS
    key = lim._pool_key(ep)
    # Simulate a caller parked in reserve() on this pool
    lim._waiters[key] = 1
    # Capacity is free, but guard must still defer to the queued reserver
    assert lim._has_capacity(ep) is True
    assert lim.check_limits(endpoint=ep) is False
    # Once the reserver leaves, guard proceeds again
    lim._waiters[key] = 0
    assert lim.check_limits(endpoint=ep) is True


def test_reserve_fifo_order(monkeypatch):
    monkeypatch.setattr(rl, "RESERVE_TIMEOUT_SEC", 2.0)
    # Recheck quickly; FIFO comes from the lock, not the computed window delay
    monkeypatch.setattr(BybitRateLimiter, "_next_free_delay", lambda self, ep: 0.005)
    # One slot total on the UID pool, so reservers must queue one at a time
    limits = {pool: 10 for pool in BybitRateLimitPool}
    limits[BybitRateLimitPool.UID_OPEN_ORDERS] = 1
    lim = _bybit(limits=limits)
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    order: list[int] = []

    async def run():
        async def worker(i: int):
            async with lim.reserve(endpoint=Endpoints.GET_OPEN_ORDERS):
                order.append(i)
                # Hold the slot until freed below, then release it
                await asyncio.sleep(0.02)
                lim.current_limit_state[pool].remaining = 1

        # Launch in order; the lock should hand out slots FIFO
        tasks = [asyncio.create_task(worker(i)) for i in range(3)]
        await asyncio.gather(*tasks)

    asyncio.run(run())
    assert order == [0, 1, 2]


def test_waiters_repr_shows_queue_depth():
    lim = _bybit()
    lim.current_limit_state = {
        pool: BybitLimitState(timestamps=deque()) for pool in BybitRateLimitPool
    }
    assert lim._waiters_repr() == ""  # idle
    lim._waiters[BybitRateLimitPool.UID_OPEN_ORDERS] = 2
    rep = repr(lim)
    assert "Reserving[" in rep
    assert "UID_OPEN_ORDERS=2" in rep


def test_next_free_delay_matches_ip_global_window():
    lim = _bybit()
    pool = BybitRateLimitPool.IP_GLOBAL
    # Oldest timestamp 400ms ago, window is 1000ms → frees in ~600ms
    lim.current_limit_state[pool].timestamps = deque([lim._now - 400, lim._now])
    delay = lim._next_free_delay(Endpoints.GET_OPEN_ORDERS)
    assert delay == pytest.approx(0.6, abs=0.001)


def test_next_free_delay_uses_reset_ts_for_uid_pool():
    lim = _bybit()
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    # Exhausted per the exchange's headers, resetting in 600ms
    lim.current_limit_state[pool].remaining = 0
    lim.current_limit_state[pool].limit = 2
    lim.current_limit_state[pool].reset_ts = lim._now + 600
    delay = lim._next_free_delay(Endpoints.GET_OPEN_ORDERS)
    assert delay == pytest.approx(0.6, abs=0.001)


def test_reconcile_uid_pool_overwrites_from_headers():
    lim = _bybit()
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    lim._reconcile_uid_pool(
        pool,
        {
            "X-Bapi-Limit": "50",
            "X-Bapi-Limit-Status": "37",
            "X-Bapi-Limit-Reset-Timestamp": str(lim._now + 900),
        },
    )
    state = lim.current_limit_state[pool]
    assert (state.limit, state.remaining, state.reset_ts) == (50, 37, lim._now + 900)


def test_reconcile_uid_pool_ignores_headers_without_limit_status():
    lim = _bybit()
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    lim.current_limit_state[pool].remaining = 7
    lim._reconcile_uid_pool(pool, {"Retry-After": "5"})
    assert lim.current_limit_state[pool].remaining == 7


def test_on_call_success_reconciles_from_exchange_last_response_headers():
    lim = _bybit()

    class _FakeExchange:
        last_response_headers = {
            "X-Bapi-Limit": "10",
            "X-Bapi-Limit-Status": "4",
            "X-Bapi-Limit-Reset-Timestamp": str(1_000_000 + 500),
        }

    lim.exchange = _FakeExchange()
    lim._on_call_success(Endpoints.GET_OPEN_ORDERS)
    state = lim.current_limit_state[BybitRateLimitPool.UID_OPEN_ORDERS]
    assert (state.limit, state.remaining) == (10, 4)
    # IP_GLOBAL has no cost_info mapping, so nothing to reconcile against
    lim._on_call_success(Endpoints.GET_SERVER_TIME)

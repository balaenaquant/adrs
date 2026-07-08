"""Tests for RateLimiter.reserve() — the waiting/priority acquire path.

Uses BybitRateLimiter (rolling-window pools) with __init__ skipped and state
pre-populated, so the behaviour is driven directly without a live exchange.
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

    default = {pool: 2 for pool in BybitRateLimitPool}
    lim.limit_profile = BybitLimitProfile(
        limits=limits or default, interval=1
    )
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
    # One slot recorded on the endpoint's UID pool and the shared IP pool
    assert len(lim.current_limit_state[BybitRateLimitPool.UID_OPEN_ORDERS].timestamps) == 1
    assert len(lim.current_limit_state[BybitRateLimitPool.IP_GLOBAL].timestamps) == 1


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
    # Fill the UID pool to its limit with timestamps that won't age out
    # (synced clock is frozen, so the rolling window never advances)
    lim.current_limit_state[pool].timestamps = deque([lim._now, lim._now])

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
    lim.current_limit_state[pool].timestamps = deque([lim._now, lim._now])

    async def run():
        async def free_slot_soon():
            await asyncio.sleep(0.05)
            lim.current_limit_state[pool].timestamps.popleft()

        freer = asyncio.create_task(free_slot_soon())
        async with lim.reserve(endpoint=Endpoints.GET_OPEN_ORDERS):
            pass
        await freer

    asyncio.run(run())
    # Acquired: the freed deque had one popped, then reserve appended its own
    assert len(lim.current_limit_state[pool].timestamps) == 2


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
                lim.current_limit_state[pool].timestamps.popleft()

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


def test_next_free_delay_matches_window(monkeypatch):
    lim = _bybit()
    pool = BybitRateLimitPool.UID_OPEN_ORDERS
    # Oldest timestamp 400ms ago, window is 1000ms → frees in ~600ms
    lim.current_limit_state[pool].timestamps = deque([lim._now - 400, lim._now])
    delay = lim._next_free_delay(Endpoints.GET_OPEN_ORDERS)
    assert delay == pytest.approx(0.6, abs=0.001)

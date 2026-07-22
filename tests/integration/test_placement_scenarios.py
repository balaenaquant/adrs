"""Integration scenarios for the delta <-> backlog <-> pool interaction.

Each test runs the REAL placement stack over the FakeExchange and checks the
conservation invariant. These are the tests a mocked-executor unit test cannot
be: the bug they guard against lives in the seam the mock would replace.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from cybotrade import Symbol
from cybotrade.models import OrderSide

from adrs.oms.ops.order_pool import ExpiredBacklogs
from tests.integration.harness import (
    assert_conserved,
    backlog_signed,
    build_opm,
    drain,
    set_desired,
)

BTC = Symbol("BTCUSDT")


# ---------------------------------------------------------------------------
# Happy path: place, fill, converge
# ---------------------------------------------------------------------------


def test_place_then_fill_converges():
    opm, sim, _ = build_opm(["BTCUSDT"])
    set_desired(opm, "BTCUSDT", "0.2")

    asyncio.run(opm.on_order_placement())
    (coid,) = list(sim.open_orders)
    sim.fill(coid)  # exchange now holds 0.2, order leaves the book

    # next tick sees the filled position → delta 0 → nothing more to do
    asyncio.run(opm.on_order_placement())

    assert sim.positions[BTC] == Decimal("0.2")
    assert len(sim.open_orders) == 0
    assert_conserved(opm, sim)


# ---------------------------------------------------------------------------
# The regression: a failed place goes to the backlog, and the NEXT tick must
# not place it again — the backlog qty is discounted from the delta.
# This is the "OrderBacklog not tracked in delta calculation" bug.
# ---------------------------------------------------------------------------


def test_backlogged_place_is_not_double_placed_next_tick():
    opm, sim, _ = build_opm(["BTCUSDT"])
    set_desired(opm, "BTCUSDT", "0.2")

    sim.throttle_next_place(1)
    asyncio.run(opm.on_order_placement())  # place fails → OrderBacklogs(0.2)

    assert len(sim.open_orders) == 0
    assert backlog_signed(opm.order_pools.order_backlog, BTC) == Decimal("0.2")
    placed_after_tick1 = len([c for c in sim.calls if c[0] == "place"])

    # Second placement tick, throttle cleared. The 0.2 is already queued, so the
    # delta must be discounted to zero and NO new placement attempted.
    asyncio.run(opm.on_order_placement())

    placed_after_tick2 = len([c for c in sim.calls if c[0] == "place"])
    assert placed_after_tick2 == placed_after_tick1, "backlog qty was double-placed"


def test_backlog_retry_drains_and_converges():
    opm, sim, _ = build_opm(["BTCUSDT"])
    set_desired(opm, "BTCUSDT", "0.2")

    sim.throttle_next_place(1)
    asyncio.run(opm.on_order_placement())  # → OrderBacklogs
    assert len(opm.order_pools.order_backlog) == 1

    drain(opm)  # retry places the queued order, backlog empties

    assert len(opm.order_pools.order_backlog) == 0
    assert _resting(sim) == Decimal("0.2")
    assert_conserved(opm, sim)


# ---------------------------------------------------------------------------
# Reversal: an opposite-direction delta cancels the resting order
# ---------------------------------------------------------------------------


def test_reversal_cancels_resting_order():
    opm, sim, _ = build_opm(["BTCUSDT"])

    # Establish a resting BUY of 0.3 (pending +0.3).
    set_desired(opm, "BTCUSDT", "0.3")
    asyncio.run(opm.on_order_placement())
    assert _resting(sim) == Decimal("0.3")

    # Now we want flat → delta is negative → cancel branch.
    set_desired(opm, "BTCUSDT", "0")
    drain(opm)

    assert _resting(sim) == Decimal("0")
    assert_conserved(opm, sim)


# ---------------------------------------------------------------------------
# Known gap, pinned: ExpiredBacklogs carries a qty (OPM order_pool) but the
# delta-discount in on_order_placement only subtracts OrderBacklogs (OPM:322).
# So an expired-order replacement queued in the backlog is invisible to the
# delta and gets placed a second time → double position. This xfail documents
# the asymmetry; flip the discount to include ExpiredBacklogs and it passes.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="on_order_placement discounts only OrderBacklogs, not ExpiredBacklogs "
    "(order_placement_manager.py:322) — ExpiredBacklogs qty is double-placed",
    strict=True,
)
def test_expired_backlog_is_also_discounted_from_delta():
    opm, sim, _ = build_opm(["BTCUSDT"])
    set_desired(opm, "BTCUSDT", "0.2")
    _ts = datetime.now(timezone.utc)

    # An order expired and its cancel already confirmed (gone from the book);
    # a 0.2 replacement is queued as an ExpiredBacklogs. Exchange flat, no
    # resting order — the only claim on the 0.2 is this backlog entry.
    opm.order_pools.order_backlog.append(
        ExpiredBacklogs(
            symbol="BTCUSDT",
            total_retries=0,
            next_retry_at=None,
            client_order_id="expired-1",
            side=OrderSide.BUY,
            qty=Decimal("0.2"),
            replace_best_bid_ask_time=_ts,
            max_replace_limit_order_time=_ts,
            is_bbo=True,
            package_id="pkg-x",
            initial_price=Decimal("100"),
            initial_time=_ts,
        )
    )

    # Placement must treat the queued 0.2 as already accounted → place nothing.
    asyncio.run(opm.on_order_placement())

    placed = len([c for c in sim.calls if c[0] == "place"])
    assert placed == 0, "ExpiredBacklogs qty was not discounted → double placement"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resting(sim):
    total = Decimal("0")
    for o in sim.open_orders.values():
        total += o.remain_size if o.side.value == "buy" else -o.remain_size
    return total

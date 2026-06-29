"""Tests for OrderPlacementManager — the fill state machine.

The most critical untested code in the OMS. Uses object.__new__ throughout
to bypass __init__ and inject only what each test needs.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.models import OrderSide, OrderStatus, Position
from cybotrade.io import EventType

from adrs.oms.ops.order_placement_manager import OrderPlacementManager
from adrs.oms.ops.order_pool import (
    OrderBacklogs,
    OrderDetails,
    OrderPoolHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def _future(seconds=60):
    return _now() + timedelta(seconds=seconds)


def _async_cm(return_value):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_position(qty: str) -> Position:
    return Position(
        symbol=Symbol("BTCUSDT"),
        quantity=Decimal(qty),
        entry_price=Decimal("0"),
        updated_time=_now(),
    )


def _order_details(
    client_order_id="coid-1", remain_size="1.0", side=OrderSide.BUY
) -> OrderDetails:
    return OrderDetails(
        client_order_id=client_order_id,
        symbol="BTCUSDT",
        side=side,
        price=Decimal("50000"),
        remain_size=Decimal(remain_size),
        replace_best_bid_ask_time=_future(120),
        max_replace_limit_order_time=_future(60),
        package_id="pkg-1",
        initial_price=Decimal("50000"),
        initial_time=_now(),
    )


def _make_update(
    status: OrderStatus,
    *,
    client_order_id="coid-1",
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    size="1.0",
    filled_size="0.0",
    remain_size="1.0",
):
    return SimpleNamespace(
        status=status,
        client_order_id=client_order_id,
        symbol=Symbol(symbol),
        side=side,
        size=Decimal(size),
        filled_size=Decimal(filled_size),
        remain_size=Decimal(remain_size),
    )


def _opm(*, pool=None, backlog=None) -> OrderPlacementManager:
    """OPM with __init__ skipped."""
    opm = object.__new__(OrderPlacementManager)

    pool = pool if pool is not None else {}
    backlog = backlog if backlog is not None else []

    order_pools = MagicMock()
    order_pools.order_pool = pool
    order_pools.order_backlog = backlog
    order_pools.order_value_update = {}
    order_pools.get_order_pool = MagicMock(return_value=_async_cm(pool))
    order_pools.get_order_backlog = MagicMock(return_value=_async_cm(backlog))
    order_pools.dedup_append = OrderPoolHandler.dedup_append
    opm.order_pools = order_pools

    sym = Symbol("BTCUSDT")
    position = MagicMock()
    position.pending = {sym: _make_position("1.0")}
    position.exchange = {sym: _make_position("0.0")}
    position.desired = {sym: _make_position("1.0")}
    position.delta_lock = asyncio.Lock()
    opm.position = position

    opm.executor = MagicMock()
    opm.executor.package_id = "pkg-1"
    opm.executor.reprice_at_bbo = AsyncMock(return_value=None)

    opm.rate_limiter = MagicMock()

    opm.config_manager = SimpleNamespace(
        config=SimpleNamespace(
            replace_best_bid_ask_time=120,
            max_retries_allowed=5,
        ),
        symbol_infos={},
    )

    return opm


# ---------------------------------------------------------------------------
# update_positions  (synchronous — the cumulative fill accounting)
# ---------------------------------------------------------------------------


def test_update_positions_buy_increments_exchange_decrements_pending():
    opm = _opm()
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")
    opm.position.exchange[sym] = _make_position("0.0")

    update = _make_update(OrderStatus.PARTIALLY_FILLED, filled_size="0.3")
    opm.update_positions(update)

    assert opm.position.exchange[sym].quantity == Decimal("0.3")
    assert opm.position.pending[sym].quantity == Decimal("0.7")


def test_update_positions_sell_increments_exchange_negative():
    opm = _opm()
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("-1.0")
    opm.position.exchange[sym] = _make_position("0.0")

    update = _make_update(
        OrderStatus.PARTIALLY_FILLED, side=OrderSide.SELL, filled_size="0.3"
    )
    opm.update_positions(update)

    assert opm.position.exchange[sym].quantity == Decimal("-0.3")
    assert opm.position.pending[sym].quantity == Decimal("-0.7")


def test_update_positions_cumulative_fill_only_increments_new_slice():
    """Third fill update must not double-count already-recorded fills."""
    opm = _opm()
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")
    opm.position.exchange[sym] = _make_position("0.0")

    # First partial fill: 0.3
    opm.update_positions(_make_update(OrderStatus.PARTIALLY_FILLED, filled_size="0.3"))
    # Second partial fill: cumulative 0.6 → new slice is 0.3
    opm.update_positions(_make_update(OrderStatus.PARTIALLY_FILLED, filled_size="0.6"))

    assert opm.position.exchange[sym].quantity == Decimal("0.6")
    assert opm.position.pending[sym].quantity == Decimal("0.4")


# ---------------------------------------------------------------------------
# on_order_update — CREATED
# ---------------------------------------------------------------------------


def test_on_order_update_created_adds_to_pending_buy():
    opm = _opm()
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("0.0")

    update = _make_update(OrderStatus.CREATED, size="1.0")
    asyncio.run(opm.on_order_update(update))

    assert opm.position.pending[sym].quantity == Decimal("1.0")


def test_on_order_update_created_subtracts_from_pending_sell():
    opm = _opm()
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("0.0")

    update = _make_update(OrderStatus.CREATED, side=OrderSide.SELL, size="0.5")
    asyncio.run(opm.on_order_update(update))

    assert opm.position.pending[sym].quantity == Decimal("-0.5")


# ---------------------------------------------------------------------------
# on_order_update — CANCELLED
# ---------------------------------------------------------------------------


def test_on_order_update_cancelled_pops_from_pool_and_adjusts_pending():
    pool = {"coid-1": _order_details()}
    opm = _opm(pool=pool)
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")

    update = _make_update(OrderStatus.CANCELLED, remain_size="1.0")
    asyncio.run(opm.on_order_update(update))

    assert "coid-1" not in pool
    assert opm.position.pending[sym].quantity == Decimal("0.0")


def test_on_order_update_cancelled_idempotent_when_already_gone():
    # Order not in pool — must not raise
    opm = _opm(pool={})
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("0.5")

    update = _make_update(OrderStatus.CANCELLED, remain_size="0.5")
    asyncio.run(opm.on_order_update(update))
    # pending still adjusted
    assert opm.position.pending[sym].quantity == Decimal("0.0")


# ---------------------------------------------------------------------------
# on_order_update — REJECTED
# ---------------------------------------------------------------------------


def test_on_order_update_rejected_adjusts_pending_and_reprices():
    pool = {"coid-1": _order_details()}
    opm = _opm(pool=pool)
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")

    update = _make_update(OrderStatus.REJECTED, remain_size="1.0")
    asyncio.run(opm.on_order_update(update))

    assert "coid-1" not in pool
    assert opm.position.pending[sym].quantity == Decimal("0.0")
    opm.executor.reprice_at_bbo.assert_awaited_once()


def test_on_order_update_rejected_not_in_pool_still_reprices():
    # WS REJECTED can arrive before the REST insert fills the pool
    opm = _opm(pool={})
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")

    update = _make_update(OrderStatus.REJECTED, remain_size="0.5")
    asyncio.run(opm.on_order_update(update))

    # pending NOT adjusted (rejected_order is None)
    assert opm.position.pending[sym].quantity == Decimal("1.0")
    opm.executor.reprice_at_bbo.assert_awaited_once()


def test_on_order_update_rejected_backlog_queued_on_reprice_failure():
    pool = {"coid-1": _order_details()}
    backlog = []
    opm = _opm(pool=pool, backlog=backlog)
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")

    failed_backlog = OrderBacklogs(
        symbol="BTCUSDT",
        total_retries=1,
        client_order_id="coid-new",
        side=OrderSide.BUY,
        ori_entry_time=_now(),
        qty=Decimal("1.0"),
        offset=Decimal("0"),
        replace_best_bid_ask_time=_future(),
        max_replace_limit_order_time=_future(30),
        package_id="pkg-1",
    )
    opm.executor.reprice_at_bbo = AsyncMock(return_value=failed_backlog)

    update = _make_update(OrderStatus.REJECTED, remain_size="1.0")
    asyncio.run(opm.on_order_update(update))

    assert len(backlog) == 1
    assert backlog[0].client_order_id == "coid-new"


# ---------------------------------------------------------------------------
# on_order_update — PARTIALLY_FILLED
# ---------------------------------------------------------------------------


def test_on_order_update_partially_filled_updates_positions():
    pool = {"coid-1": _order_details(remain_size="1.0")}
    opm = _opm(pool=pool)
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")
    opm.position.exchange[sym] = _make_position("0.0")

    # validate_oms_state would fail if pool is empty — pool has the entry
    opm.position.update_exchange = AsyncMock()
    opm.position.update_pending = AsyncMock()
    opm.order_pools.resync_order_pool = AsyncMock()

    update = _make_update(
        OrderStatus.PARTIALLY_FILLED, filled_size="0.4", remain_size="0.6"
    )
    asyncio.run(opm.on_order_update(update))

    assert opm.position.exchange[sym].quantity == Decimal("0.4")
    assert opm.position.pending[sym].quantity == Decimal("0.6")
    # order stays in pool (not terminal)
    assert "coid-1" in pool


# ---------------------------------------------------------------------------
# on_order_update — FILLED
# ---------------------------------------------------------------------------


def test_on_order_update_filled_removes_from_pool_and_updates_positions():
    pool = {"coid-1": _order_details(remain_size="1.0")}
    opm = _opm(pool=pool)
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")
    opm.position.exchange[sym] = _make_position("0.0")

    opm.position.update_exchange = AsyncMock()
    opm.position.update_pending = AsyncMock()
    opm.order_pools.resync_order_pool = AsyncMock()

    update = _make_update(OrderStatus.FILLED, filled_size="1.0", remain_size="0.0")
    asyncio.run(opm.on_order_update(update))

    assert "coid-1" not in pool
    assert opm.position.exchange[sym].quantity == Decimal("1.0")
    # order_value_update entry cleaned up
    assert "coid-1" not in opm.order_pools.order_value_update


# ---------------------------------------------------------------------------
# on_order_update — PARTIALLY_FILLED_CANCELLED
# ---------------------------------------------------------------------------


def test_on_order_update_partially_filled_cancelled_adjusts_remaining_pending():
    """
    Order for 1.0 BTC: 0.4 filled, 0.6 remain, then cancelled.
    update_positions records the 0.4 fill (pending 1.0→0.6),
    then the PARTIALLY_FILLED_CANCELLED branch removes the 0.6 remain (pending 0.6→0.0).
    """
    pool = {"coid-1": _order_details(remain_size="0.6")}
    opm = _opm(pool=pool)
    sym = Symbol("BTCUSDT")
    opm.position.pending[sym] = _make_position("1.0")
    opm.position.exchange[sym] = _make_position("0.0")

    opm.position.update_exchange = AsyncMock()
    opm.position.update_pending = AsyncMock()
    opm.order_pools.resync_order_pool = AsyncMock()

    # No prior fills recorded — update_positions will apply the full 0.4
    update = _make_update(
        OrderStatus.PARTIALLY_FILLED_CANCELLED,
        filled_size="0.4",
        remain_size="0.6",
    )
    asyncio.run(opm.on_order_update(update))

    assert "coid-1" not in pool
    assert opm.position.pending[sym].quantity == Decimal("0.0")
    assert opm.position.exchange[sym].quantity == Decimal("0.4")


# ---------------------------------------------------------------------------
# on_exchange_event routing
# ---------------------------------------------------------------------------


def test_on_exchange_event_routes_order_update():
    opm = _opm()
    opm.on_order_update = AsyncMock()

    update_data = _make_update(OrderStatus.CREATED)
    event = SimpleNamespace(event_type=EventType.OrderUpdate, data=update_data)

    asyncio.run(opm.on_exchange_event(event))
    opm.on_order_update.assert_awaited_once_with(update_data)


def test_on_exchange_event_does_not_raise_on_error_event():
    opm = _opm()
    event = SimpleNamespace(event_type=EventType.Error, data="exchange error")
    asyncio.run(opm.on_exchange_event(event))  # must not raise

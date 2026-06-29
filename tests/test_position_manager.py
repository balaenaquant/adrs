"""Tests for PositionManager — delta calculation and position update logic."""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.models import Position, OrderSide

from adrs.oms.position import PositionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(symbol: str, quantity: str) -> Position:
    return Position(
        symbol=Symbol(symbol),
        quantity=Decimal(quantity),
        entry_price=Decimal("0"),
        updated_time=datetime.now(timezone.utc),
    )


def _async_cm(return_value):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _pm(*, balance="10000", leverage="1", symbols=None) -> PositionManager:
    """PositionManager with __init__ skipped; pre-populate via caller."""
    symbols = symbols or {"BTC": "BTCUSDT"}
    pm = object.__new__(PositionManager)
    pm.exchange = {}
    pm.pending = {}
    pm.desired = {}
    pm.delta_lock = asyncio.Lock()

    rate_limiter = MagicMock()
    rate_limiter.guard = MagicMock(return_value=_async_cm(None))
    pm.rate_limiter = rate_limiter

    pm.config = SimpleNamespace(
        config=SimpleNamespace(
            initial_balance=Decimal(balance),
            leverage=Decimal(leverage),
            base_asset_to_symbol_table=symbols,
        ),
        exchange=MagicMock(),
    )
    return pm


# ---------------------------------------------------------------------------
# compute_base_quantity
# ---------------------------------------------------------------------------


def test_compute_base_quantity_long():
    pm = _pm(balance="10000", leverage="1")
    qty = pm.compute_base_quantity(price=Decimal("50000"), weightage=Decimal("0.5"))
    # (10000 * 1 * 0.5) / 50000 = 0.1
    assert qty == Decimal("0.1")


def test_compute_base_quantity_short():
    pm = _pm(balance="10000", leverage="1")
    qty = pm.compute_base_quantity(price=Decimal("50000"), weightage=Decimal("-0.5"))
    assert qty == Decimal("-0.1")


def test_compute_base_quantity_with_leverage():
    pm = _pm(balance="10000", leverage="3")
    qty = pm.compute_base_quantity(price=Decimal("50000"), weightage=Decimal("0.5"))
    # (10000 * 3 * 0.5) / 50000 = 0.3
    assert qty == Decimal("0.3")


def test_compute_base_quantity_zero_weight():
    pm = _pm(balance="10000", leverage="1")
    qty = pm.compute_base_quantity(price=Decimal("50000"), weightage=Decimal("0"))
    assert qty == Decimal("0")


# ---------------------------------------------------------------------------
# delta_calculation (pure math path — pre-populate state, skip exchange calls)
# ---------------------------------------------------------------------------


def _pm_with_positions(desired, exchange, pending) -> PositionManager:
    """Build a PM with pre-loaded state and exchange calls that return no-ops."""
    pm = _pm()
    sym = Symbol("BTCUSDT")
    pm.desired = {sym: _make_position("BTCUSDT", desired)}
    pm.exchange = {sym: _make_position("BTCUSDT", exchange)}
    pm.pending = {sym: _make_position("BTCUSDT", pending)}

    # Stub update_* to do nothing — we test the math, not the exchange calls
    async def _noop():
        pass

    pm.update_pending = _noop
    pm.update_exchange = _noop
    return pm


def test_delta_calculation_case1():
    # desired +3, exchange -1, pending +2 → delta = 3 - (-1 + 2) = 2
    pm = _pm_with_positions(desired="3", exchange="-1", pending="2")
    deltas = asyncio.run(pm.delta_calculation())
    assert deltas[Symbol("BTCUSDT")] == Decimal("2")


def test_delta_calculation_case2():
    # desired -3, exchange +1, pending -2 → delta = -3 - (1 - 2) = -2
    pm = _pm_with_positions(desired="-3", exchange="1", pending="-2")
    deltas = asyncio.run(pm.delta_calculation())
    assert deltas[Symbol("BTCUSDT")] == Decimal("-2")


def test_delta_calculation_zero_delta():
    # desired 1, exchange 0.5, pending 0.5 → delta = 0
    pm = _pm_with_positions(desired="1", exchange="0.5", pending="0.5")
    deltas = asyncio.run(pm.delta_calculation())
    assert deltas[Symbol("BTCUSDT")] == Decimal("0")


def test_delta_calculation_multiple_symbols():
    pm = _pm(symbols={"BTC": "BTCUSDT", "ETH": "ETHUSDT"})
    btc = Symbol("BTCUSDT")
    eth = Symbol("ETHUSDT")
    pm.desired = {
        btc: _make_position("BTCUSDT", "1"),
        eth: _make_position("ETHUSDT", "-0.5"),
    }
    pm.exchange = {
        btc: _make_position("BTCUSDT", "0"),
        eth: _make_position("ETHUSDT", "0"),
    }
    pm.pending = {
        btc: _make_position("BTCUSDT", "0"),
        eth: _make_position("ETHUSDT", "0"),
    }

    async def _noop():
        pass

    pm.update_pending = _noop
    pm.update_exchange = _noop

    deltas = asyncio.run(pm.delta_calculation())
    assert deltas[btc] == Decimal("1")
    assert deltas[eth] == Decimal("-0.5")


# ---------------------------------------------------------------------------
# update_exchange
# ---------------------------------------------------------------------------


def test_update_exchange_populates_state():
    pm = _pm()
    sym = Symbol("BTCUSDT")
    pos = _make_position("BTCUSDT", "2.5")
    pm.config.exchange.get_positions = AsyncMock(return_value=[pos])

    asyncio.run(pm.update_exchange())
    assert pm.exchange[sym].quantity == Decimal("2.5")


def test_update_exchange_tolerates_error():
    pm = _pm()
    pm.config.exchange.get_positions = AsyncMock(side_effect=RuntimeError("timeout"))
    asyncio.run(pm.update_exchange())  # must not raise


# ---------------------------------------------------------------------------
# update_pending
# ---------------------------------------------------------------------------


def _make_open_order(side: OrderSide, remain_size: str, price: str = "50000"):
    return SimpleNamespace(
        side=side,
        remain_size=Decimal(remain_size),
        price=Decimal(price),
    )


def test_update_pending_sums_buy_orders():
    pm = _pm(symbols={"BTC": "BTCUSDT"})
    pm.config.exchange.get_open_orders = AsyncMock(
        return_value=[
            _make_open_order(OrderSide.BUY, "0.3"),
            _make_open_order(OrderSide.BUY, "0.2"),
        ]
    )
    asyncio.run(pm.update_pending())
    assert pm.pending[Symbol("BTCUSDT")].quantity == Decimal("0.5")


def test_update_pending_subtracts_sell_orders():
    pm = _pm(symbols={"BTC": "BTCUSDT"})
    pm.config.exchange.get_open_orders = AsyncMock(
        return_value=[
            _make_open_order(OrderSide.SELL, "0.3"),
            _make_open_order(OrderSide.SELL, "0.2"),
        ]
    )
    asyncio.run(pm.update_pending())
    assert pm.pending[Symbol("BTCUSDT")].quantity == Decimal("-0.5")


def test_update_pending_no_orders_is_zero():
    pm = _pm(symbols={"BTC": "BTCUSDT"})
    pm.config.exchange.get_open_orders = AsyncMock(return_value=[])
    asyncio.run(pm.update_pending())
    assert pm.pending[Symbol("BTCUSDT")].quantity == Decimal("0")


def test_update_pending_tolerates_error():
    pm = _pm(symbols={"BTC": "BTCUSDT"})
    pm.config.exchange.get_open_orders = AsyncMock(
        side_effect=RuntimeError("rate limit")
    )
    asyncio.run(pm.update_pending())  # must not raise

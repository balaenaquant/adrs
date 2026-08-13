"""
Positions sourced from the ACCOUNT_UPDATE stream.

The point of this path is that it is *absolute*. The incremental writer it
replaces drifts by construction -- its own comment records fills over-counting
from the third update onward -- so these tests pin the property that makes the
replacement worthwhile: applying the same frame twice must not double a position.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.models import Position

from adrs.oms.position import POSITION_ANCHOR_MAX_AGE_SEC, PositionManager

BTC = Symbol("BTCUSDT")
ETH = Symbol("ETHUSDT")


def _position(symbol: Symbol, qty: str) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        entry_price=Decimal("63500"),
        updated_time=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )


def _pm() -> PositionManager:
    pm = object.__new__(PositionManager)
    pm.exchange = {}
    pm.pending = {}
    pm.desired = {}
    pm.delta_lock = asyncio.Lock()
    pm._refresh_lock = asyncio.Lock()
    pm._exchange_refreshed_at = None
    pm.rate_limiter = MagicMock()
    pm.config = SimpleNamespace(
        config=SimpleNamespace(
            initial_balance=Decimal("10000"),
            leverage=Decimal("1"),
            base_asset_to_symbol_table={"BTC": "BTCUSDT"},
        ),
        exchange=MagicMock(),
    )
    return pm


def test_applying_a_frame_sets_the_position_absolutely():
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    assert pm.exchange[BTC].quantity == Decimal("0.15")


def test_applying_the_same_frame_twice_does_not_double_the_position():
    """The property the incremental writer could not offer."""
    pm = _pm()
    frame = [_position(BTC, "0.15")]
    pm.apply_stream_positions(frame)
    pm.apply_stream_positions(frame)
    assert pm.exchange[BTC].quantity == Decimal("0.15")


def test_a_later_frame_replaces_rather_than_accumulates():
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    pm.apply_stream_positions([_position(BTC, "0.40")])
    assert pm.exchange[BTC].quantity == Decimal("0.40")


def test_a_symbol_absent_from_the_frame_keeps_its_previous_value():
    """
    Frames are partial: they carry the positions that changed. Replacing the whole
    dict would zero every symbol Binance did not mention.
    """
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15"), _position(ETH, "-2")])
    pm.apply_stream_positions([_position(BTC, "0.20")])
    assert pm.exchange[BTC].quantity == Decimal("0.20")
    assert pm.exchange[ETH].quantity == Decimal("-2")


def test_a_flat_position_is_applied_not_ignored():
    """
    Closing to zero is the frame that matters most: ignore it and the OMS sizes
    orders against a position it no longer holds.
    """
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    pm.apply_stream_positions([_position(BTC, "0")])
    assert pm.exchange[BTC].quantity == Decimal("0")


def test_an_empty_frame_changes_nothing():
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    pm.apply_stream_positions([])
    assert pm.exchange[BTC].quantity == Decimal("0.15")


def test_apply_stream_positions_rebinds_rather_than_mutates_in_place():
    """
    update_exchange's race guard (`self.exchange.get(sym) is not
    before.get(sym)`, see the "Relies on every writer REBINDING the Position
    object rather than mutating it in place" comment in position.py) is sound
    only because every writer of `exchange` rebinds a whole Position object
    instead of mutating one in place -- identity comparison is meaningless
    against an object that never changes address.

    That's a real convention today, but order_placement_manager.py mutates
    `pending` in place (`self.position.pending[sym].quantity -= ...`), so a
    future `exchange` writer has a live pattern to copy that would silently
    disable the guard. This pins the contract directly: if
    apply_stream_positions ever mutated the existing Position instead of
    replacing it, this test would fail.
    """
    pm = _pm()
    original = _position(BTC, "0.15")
    pm.exchange[BTC] = original

    pm.apply_stream_positions([_position(BTC, "0.40")])

    assert pm.exchange[BTC] is not original, (
        "apply_stream_positions must rebind a new Position rather than "
        "mutate the existing one -- an in-place mutation would silently "
        "disable update_exchange's identity-based race guard"
    )
    assert original.quantity == Decimal("0.15"), (
        "the old Position object must be left untouched, proving the write "
        "was a rebind and not a mutation"
    )


def test_the_stream_does_not_stamp_the_rest_anchor():
    """
    The anchor is what proves liveness, and only a REST read may set it. If the
    stream stamped it, a dead REST path would look healthy forever and the sizing
    path would trust an unanchored stream indefinitely.
    """
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    assert pm._exchange_refreshed_at is None


def test_anchor_max_age_is_bounded():
    # Must exceed the 60s aegis cadence that sets the anchor, or the sizing path
    # would force a REST read on most ticks and the saving would vanish.
    assert 60.0 < POSITION_ANCHOR_MAX_AGE_SEC <= 120.0
    # Pinned to the project's exact required value, not just the bound above.
    assert POSITION_ANCHOR_MAX_AGE_SEC == 90.0


def test_delta_calculation_does_not_read_rest_while_the_anchor_is_fresh():
    import time

    pm = _pm()
    pm.config.exchange.get_positions = AsyncMock(return_value=[])
    pm._exchange_refreshed_at = time.monotonic()  # fresh anchor
    pm.desired = {BTC: _position(BTC, "1")}
    pm.exchange = {BTC: _position(BTC, "0")}
    pm.pending = {BTC: _position(BTC, "0")}
    pm.update_pending = lambda snapshot: None

    asyncio.run(pm.delta_calculation(SimpleNamespace(orders={})))
    pm.config.exchange.get_positions.assert_not_awaited()


def test_delta_calculation_forces_a_read_once_the_anchor_is_stale():
    import time

    pm = _pm()
    pm.config.exchange.get_positions = AsyncMock(return_value=[])
    pm._exchange_refreshed_at = time.monotonic() - POSITION_ANCHOR_MAX_AGE_SEC - 1
    pm.desired = {BTC: _position(BTC, "1")}
    pm.exchange = {BTC: _position(BTC, "0")}
    pm.pending = {BTC: _position(BTC, "0")}
    pm.update_pending = lambda snapshot: None

    asyncio.run(pm.delta_calculation(SimpleNamespace(orders={})))
    pm.config.exchange.get_positions.assert_awaited_once()


def test_a_position_update_event_reaches_the_position_manager():
    from cybotrade.io.event import Event, EventType

    from adrs.oms.ops.order_placement_manager import OrderPlacementManager

    opm = object.__new__(OrderPlacementManager)
    opm.position = _pm()
    asyncio.run(
        opm.on_exchange_event(
            Event(
                event_type=EventType.PositionUpdate,
                orig="{}",
                data=[_position(BTC, "0.25")],
            )
        )
    )
    assert opm.position.exchange[BTC].quantity == Decimal("0.25")


def test_an_order_fill_no_longer_moves_the_exchange_position():
    """
    The retired writer. update_positions used to do `exchange += asset_filled`,
    which drifted; ACCOUNT_UPDATE now owns that value absolutely. `pending` keeps
    its incremental accounting, corrected each tick by the open-orders snapshot.
    """
    from cybotrade.models import OrderSide

    from adrs.oms.ops.order_placement_manager import OrderPlacementManager

    opm = object.__new__(OrderPlacementManager)
    opm.position = _pm()
    opm.position.exchange = {BTC: _position(BTC, "1.0")}
    opm.position.pending = {BTC: _position(BTC, "1.0")}
    opm.order_pools = SimpleNamespace(order_value_update={})

    update = SimpleNamespace(
        client_order_id="coid",
        symbol=BTC,
        side=OrderSide.BUY,
        filled_size=Decimal("0.3"),
    )
    opm.update_positions(update)

    assert opm.position.exchange[BTC].quantity == Decimal("1.0"), (
        "exchange must be owned by the stream, not moved by fills"
    )
    assert opm.position.pending[BTC].quantity == Decimal("0.7"), (
        "pending keeps its incremental accounting"
    )

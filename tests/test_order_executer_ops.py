"""Tests for OrderExecutor operational methods.

Covers cancel_single_order, cancel_multi_limit_order, _get_current_price_safe,
place_single_limit_order, reprice_at_bbo, reprice_at_mid, and
place_multiple_limit_order.

split_order_quantity and _make_context are in test_order_executer.py.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cybotrade import Symbol
from cybotrade.models import OrderSide, SymbolInfo

from adrs.oms.ops.order_executer import OrderExecutor
from adrs.oms.ops.order_pool import CancelBacklogs, OrderBacklogs, OrderPoolHandler
from adrs.oms.rate_limit.error_policy import DefaultErrorPolicy, ErrorAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def _future(s=60):
    return _now() + timedelta(seconds=s)


def _async_cm(return_value):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _symbol_info() -> SymbolInfo:
    return SymbolInfo(
        symbol="BTCUSDT",
        quantity_precision=3,
        price_precision=2,
        exchange="bybit_linear",
        tick_size=Decimal("0.5"),
        max_post_only_qty=Decimal("1000"),
        max_limit_qty=Decimal("1000"),
        min_limit_qty=Decimal("0.001"),
        max_market_qty=Decimal("1000"),
        min_market_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_notional=Decimal("1000000"),
        quanto_multiplier=Decimal("1"),
    )


def _open_order(side: OrderSide, remain_size: str, client_order_id: str = "oid-1"):
    return SimpleNamespace(
        symbol=Symbol("BTCUSDT"),
        side=side,
        remain_size=Decimal(remain_size),
        client_order_id=client_order_id,
        price=Decimal("50000"),
    )


def _executor(*, pool=None, backlog=None, error_policy=None) -> OrderExecutor:
    """OrderExecutor with __init__ skipped."""
    ex = object.__new__(OrderExecutor)

    pool = pool if pool is not None else {}
    backlog = backlog if backlog is not None else []

    ex.exchange = MagicMock()
    ex.config = SimpleNamespace(
        portfolio_id="p1",
        min_limit_replace_interval=5,
        max_limit_replace_interval=15,
        replace_best_bid_ask_time=120,
    )
    ex.symbol_infos = {Symbol("BTCUSDT"): _symbol_info()}

    rate_limiter = MagicMock()
    rate_limiter.guard = MagicMock(return_value=_async_cm(None))
    ex.rate_limiter = rate_limiter

    ex.error_policy = error_policy or DefaultErrorPolicy()

    order_pools = MagicMock()
    order_pools.order_pool = pool
    order_pools.order_backlog = backlog
    order_pools.order_records = {}
    order_pools.order_value_update = {}
    order_pools.get_order_pool = MagicMock(return_value=_async_cm(pool))
    order_pools.get_order_backlog = MagicMock(return_value=_async_cm(backlog))
    order_pools.dedup_append = OrderPoolHandler.dedup_append
    ex.order_pools = order_pools

    ex.package_id = "pkg-1"
    return ex


# ---------------------------------------------------------------------------
# cancel_single_order
# ---------------------------------------------------------------------------


def test_cancel_single_order_success_pops_pool():
    pool = {"coid-1": object()}
    ex = _executor(pool=pool)
    ex.exchange.cancel_order = AsyncMock()

    result = asyncio.run(ex.cancel_single_order(Symbol("BTCUSDT"), "coid-1"))

    assert result is None
    assert "coid-1" not in pool
    ex.exchange.cancel_order.assert_awaited_once()


def test_cancel_single_order_fatal_error_returns_none_keeps_pool():
    pool = {"coid-1": object()}

    policy = MagicMock()
    policy.classify = MagicMock(return_value=ErrorAction.FATAL)
    ex = _executor(pool=pool, error_policy=policy)
    ex.exchange.cancel_order = AsyncMock(side_effect=RuntimeError("fatal"))

    result = asyncio.run(ex.cancel_single_order(Symbol("BTCUSDT"), "coid-1"))

    assert result is None
    assert "coid-1" in pool  # kept in pool for expiry/delta logic


def test_cancel_single_order_retryable_returns_cancel_backlog():
    pool = {"coid-1": object()}

    policy = MagicMock()
    policy.classify = MagicMock(return_value=ErrorAction.RETRY)
    ex = _executor(pool=pool, error_policy=policy)
    ex.exchange.cancel_order = AsyncMock(side_effect=RuntimeError("network"))

    result = asyncio.run(ex.cancel_single_order(Symbol("BTCUSDT"), "coid-1"))

    assert isinstance(result, CancelBacklogs)
    assert result.client_order_id == "coid-1"
    assert "coid-1" in pool  # NOT popped on retry


def test_cancel_single_order_already_gone_from_pool_returns_none():
    # Order was already popped by WS before cancel error fires → treat as done
    pool = {}  # empty — already_left = True

    policy = MagicMock()
    policy.classify = MagicMock(return_value=ErrorAction.RETRY)
    ex = _executor(pool=pool, error_policy=policy)
    ex.exchange.cancel_order = AsyncMock(side_effect=RuntimeError("lost"))

    result = asyncio.run(ex.cancel_single_order(Symbol("BTCUSDT"), "coid-1"))

    assert result is None  # already_left short-circuits retry


# ---------------------------------------------------------------------------
# cancel_multi_limit_order
# ---------------------------------------------------------------------------


def test_cancel_multi_limit_order_zero_target_raises():
    ex = _executor()
    with __import__("pytest").raises(ValueError):
        asyncio.run(
            ex.cancel_multi_limit_order(Symbol("BTCUSDT"), Decimal("0"), open_orders=[])
        )


def test_cancel_multi_limit_order_buy_target_cancels_sell_orders():
    # target +1.0 (BUY) → cancel SELL pending orders
    sell_order = _open_order(OrderSide.SELL, "1.5", client_order_id="sell-1")
    buy_order = _open_order(OrderSide.BUY, "0.5", client_order_id="buy-1")  # kept

    ex = _executor()
    ex.exchange.cancel_order = AsyncMock()

    remainder = asyncio.run(
        ex.cancel_multi_limit_order(
            Symbol("BTCUSDT"), Decimal("1.0"), open_orders=[sell_order, buy_order]
        )
    )

    # Cancelled 1.5 SELL to cover target 1.0 → remainder = 1.0 - 1.5 = -0.5
    assert remainder == Decimal("-0.5")
    ex.exchange.cancel_order.assert_awaited_once()


def test_cancel_multi_limit_order_no_opposite_orders_returns_target():
    # No SELL orders to cancel when BUY target
    ex = _executor()
    ex.exchange.cancel_order = AsyncMock()

    remainder = asyncio.run(
        ex.cancel_multi_limit_order(Symbol("BTCUSDT"), Decimal("1.0"), open_orders=[])
    )

    assert remainder == Decimal("1.0")  # target unchanged
    ex.exchange.cancel_order.assert_not_awaited()


def test_cancel_multi_limit_order_failed_cancel_queued_in_backlog():
    sell_order = _open_order(OrderSide.SELL, "1.0", client_order_id="sell-1")
    backlog = []
    # Order must be in the local pool so already_left=False doesn't short-circuit
    pool = {"sell-1": object()}

    policy = MagicMock()
    policy.classify = MagicMock(return_value=ErrorAction.RETRY)
    ex = _executor(pool=pool, backlog=backlog, error_policy=policy)
    ex.exchange.cancel_order = AsyncMock(side_effect=RuntimeError("timeout"))

    asyncio.run(
        ex.cancel_multi_limit_order(
            Symbol("BTCUSDT"), Decimal("1.0"), open_orders=[sell_order]
        )
    )

    assert len(backlog) == 1
    assert backlog[0].client_order_id == "sell-1"


# ---------------------------------------------------------------------------
# _get_current_price_safe
# ---------------------------------------------------------------------------


def test_get_current_price_safe_returns_price():
    ex = _executor()
    ex.exchange.get_current_price = AsyncMock(return_value=Decimal("50000"))

    result = asyncio.run(ex._get_current_price_safe(Symbol("BTCUSDT")))
    assert result == Decimal("50000")


def test_get_current_price_safe_returns_none_on_error():
    ex = _executor()
    ex.exchange.get_current_price = AsyncMock(side_effect=RuntimeError("timeout"))

    result = asyncio.run(ex._get_current_price_safe(Symbol("BTCUSDT")))
    assert result is None


# ---------------------------------------------------------------------------
# place_single_limit_order
# ---------------------------------------------------------------------------


def test_place_single_limit_order_success_adds_to_pool():
    pool = {}
    ex = _executor(pool=pool)
    ex.exchange.place_order = AsyncMock()

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        result = asyncio.run(
            ex.place_single_limit_order(
                symbol=Symbol("BTCUSDT"),
                offset=Decimal("0"),
                qty=Decimal("0.1"),
                side=OrderSide.BUY,
                symbol_info=_symbol_info(),
            )
        )

    assert result is None
    assert len(pool) == 1
    ex.exchange.place_order.assert_awaited_once()


def test_place_single_limit_order_exchange_error_returns_backlog():
    ex = _executor()
    ex.exchange.place_order = AsyncMock(side_effect=RuntimeError("post failed"))

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        result = asyncio.run(
            ex.place_single_limit_order(
                symbol=Symbol("BTCUSDT"),
                offset=Decimal("0"),
                qty=Decimal("0.1"),
                side=OrderSide.BUY,
                symbol_info=_symbol_info(),
            )
        )

    assert isinstance(result, OrderBacklogs)
    assert result.qty == Decimal("0.1")
    assert result.side == OrderSide.BUY


def test_place_single_limit_order_fatal_error_returns_none():
    policy = MagicMock()
    policy.classify = MagicMock(return_value=ErrorAction.FATAL)
    ex = _executor(error_policy=policy)
    ex.exchange.place_order = AsyncMock(side_effect=RuntimeError("fatal"))

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        result = asyncio.run(
            ex.place_single_limit_order(
                symbol=Symbol("BTCUSDT"),
                offset=Decimal("0"),
                qty=Decimal("0.1"),
                side=OrderSide.BUY,
                symbol_info=_symbol_info(),
            )
        )

    assert result is None


def test_place_single_limit_order_reuses_client_order_id_for_dedup():
    """Backlog retries pass the original client_order_id so the exchange can dedup."""
    pool = {}
    ex = _executor(pool=pool)
    ex.exchange.place_order = AsyncMock()

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        asyncio.run(
            ex.place_single_limit_order(
                symbol=Symbol("BTCUSDT"),
                offset=Decimal("0"),
                qty=Decimal("0.1"),
                side=OrderSide.BUY,
                symbol_info=_symbol_info(),
                client_order_id="reuse-me",
            )
        )

    assert "reuse-me" in pool


def test_place_single_limit_order_sell_uses_ask_price():
    pool = {}
    ex = _executor(pool=pool)
    ex.exchange.place_order = AsyncMock()

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        asyncio.run(
            ex.place_single_limit_order(
                symbol=Symbol("BTCUSDT"),
                offset=Decimal("0"),
                qty=Decimal("0.1"),
                side=OrderSide.SELL,
                symbol_info=_symbol_info(),
            )
        )

    call_kwargs = ex.exchange.place_order.call_args.kwargs
    # SELL should use ask (50001) and ceil to tick → >= 50001
    assert call_kwargs["limit"] >= Decimal("50001")


# ---------------------------------------------------------------------------
# reprice_at_bbo
# ---------------------------------------------------------------------------


def test_reprice_at_bbo_uses_zero_offset():
    """reprice_at_bbo must pass offset=0 so the order posts at BBO."""
    ex = _executor()
    ex.place_single_limit_order = AsyncMock(return_value=None)

    asyncio.run(
        ex.reprice_at_bbo(
            symbol=Symbol("BTCUSDT"),
            qty=Decimal("0.1"),
            side=OrderSide.BUY,
            replace_best_bid_ask_time=_future(),
            package_id="pkg-1",
        )
    )

    call_kwargs = ex.place_single_limit_order.call_args.kwargs
    assert call_kwargs["offset"] == Decimal("0")


# ---------------------------------------------------------------------------
# reprice_at_mid
# ---------------------------------------------------------------------------


def test_reprice_at_mid_uses_computed_offset():
    ex = _executor()
    ex.place_single_limit_order = AsyncMock(return_value=None)
    ex.exchange.get_current_price = AsyncMock(return_value=Decimal("50000"))

    # Override compute_limit_offsets to return a known offset
    async def _fixed_offset(level, ctx):
        return [Decimal("0.001")]

    ex.compute_limit_offsets = _fixed_offset

    asyncio.run(
        ex.reprice_at_mid(
            symbol=Symbol("BTCUSDT"),
            qty=Decimal("0.1"),
            side=OrderSide.BUY,
            replace_best_bid_ask_time=_future(),
            package_id="pkg-1",
            initial_price=Decimal("50000"),
        )
    )

    call_kwargs = ex.place_single_limit_order.call_args.kwargs
    assert call_kwargs["offset"] == Decimal("0.001")


def test_reprice_at_mid_falls_back_to_bbo_when_price_unavailable():
    ex = _executor()
    ex.place_single_limit_order = AsyncMock(return_value=None)
    ex.exchange.get_current_price = AsyncMock(side_effect=RuntimeError("timeout"))

    asyncio.run(
        ex.reprice_at_mid(
            symbol=Symbol("BTCUSDT"),
            qty=Decimal("0.1"),
            side=OrderSide.BUY,
            replace_best_bid_ask_time=_future(),
            package_id="pkg-1",
            initial_price=Decimal("50000"),
        )
    )

    call_kwargs = ex.place_single_limit_order.call_args.kwargs
    # falls back to BBO offset = 0
    assert call_kwargs["offset"] == Decimal("0")


# ---------------------------------------------------------------------------
# place_multiple_limit_order
# ---------------------------------------------------------------------------


def test_place_multiple_limit_order_places_all_splits():
    ex = _executor()
    ex.exchange.place_order = AsyncMock()
    ex.exchange.get_current_price = AsyncMock(return_value=Decimal("50000"))

    # Override split to return two fixed sizes
    async def _two_splits(qty, ctx):
        return [Decimal("0.05"), Decimal("0.05")]

    ex.split_order_quantity = _two_splits

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        asyncio.run(ex.place_multiple_limit_order(Symbol("BTCUSDT"), Decimal("0.1")))

    assert ex.exchange.place_order.await_count == 2


def test_place_multiple_limit_order_no_price_skips():
    ex = _executor()
    ex.exchange.get_current_price = AsyncMock(side_effect=RuntimeError("timeout"))
    ex.exchange.place_order = AsyncMock()

    asyncio.run(ex.place_multiple_limit_order(Symbol("BTCUSDT"), Decimal("0.1")))

    ex.exchange.place_order.assert_not_awaited()


def test_place_multiple_limit_order_broken_split_falls_back_to_naive():
    """If split_order_quantity override raises, base impl is used as fallback."""
    ex = _executor()
    ex.exchange.place_order = AsyncMock()
    ex.exchange.get_current_price = AsyncMock(return_value=Decimal("50000"))

    async def _broken(qty, ctx):
        raise RuntimeError("override bug")

    ex.split_order_quantity = _broken

    with patch(
        "adrs.oms.ops.order_executer.OrderUtils.get_order_book",
        new=AsyncMock(return_value=[Decimal("49999"), Decimal("50001")]),
    ):
        asyncio.run(ex.place_multiple_limit_order(Symbol("BTCUSDT"), Decimal("0.1")))

    # Naive fallback places 1 order for the full qty
    assert ex.exchange.place_order.await_count == 1

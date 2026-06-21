"""Tests for OrderExecutor.split_order_quantity quantization.

The naive default splitter must round (truncate) every order size to the
instrument's `quantity_precision` — exchanges reject a qty that isn't a
multiple of the lot step (bybit "10001 Qty invalid").
"""

import asyncio
from decimal import Decimal

from cybotrade import Symbol
from cybotrade.models import OrderSide, SymbolInfo

from adrs.oms.ops.order_executer import OrderExecutor, PlacementContext


def _symbol_info(quantity_precision: int) -> SymbolInfo:
    return SymbolInfo(
        symbol="BTCUSDT",
        quantity_precision=quantity_precision,
        price_precision=2,
        exchange="bybit_linear",
        tick_size=Decimal("0.1"),
        max_post_only_qty=Decimal("1000"),
        max_limit_qty=Decimal("1000"),
        min_limit_qty=Decimal("0.001"),
        max_market_qty=Decimal("1000"),
        min_market_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_notional=Decimal("1000000"),
        quanto_multiplier=Decimal("1"),
    )


def _ctx(quantity_precision: int, min_qty="0.001", max_qty="1000") -> PlacementContext:
    return PlacementContext(
        symbol=Symbol("BTCUSDT"),
        side=OrderSide.BUY,
        market_price=Decimal("60000"),
        symbol_info=_symbol_info(quantity_precision),
        min_qty=Decimal(min_qty),
        max_qty=Decimal(max_qty),
    )


def _split(qty, ctx):
    return asyncio.run(
        OrderExecutor.split_order_quantity(object.__new__(OrderExecutor), qty, ctx)
    )


def test_single_order_truncated_to_precision():
    # raw 28-decimal qty (the bug) -> truncated to 3 dp
    sizes = _split(Decimal("0.03122562937000123"), _ctx(3))
    assert sizes == [Decimal("0.031")]


def test_precision_zero_truncates_to_integer():
    sizes = _split(Decimal("7.9"), _ctx(0))
    assert sizes == [Decimal("7")]


def test_qty_below_min_after_truncation_is_dropped():
    # 0.0009 truncates to 0.000 at precision 3 -> below min_qty -> no order
    sizes = _split(Decimal("0.0009"), _ctx(3, min_qty="0.001"))
    assert sizes == []


def test_chunking_truncates_each_chunk_and_remainder():
    # qty 2500.7777 with max 1000 -> [1000, 1000, 500.777]
    sizes = _split(Decimal("2500.7777"), _ctx(3, max_qty="1000"))
    assert sizes == [Decimal("1000"), Decimal("1000"), Decimal("500.777")]


def test_remainder_below_min_dropped():
    # remainder 0.0005 < min 0.001 -> only the full chunks
    sizes = _split(Decimal("1000.0005"), _ctx(3, max_qty="1000"))
    assert sizes == [Decimal("1000")]

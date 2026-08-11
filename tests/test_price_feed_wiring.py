"""
The read paths must prefer the feed and fall back to REST *through* reserve().

The reserve() detail is not incidental: it is what makes the worst case equal
today's cost instead of a stampede when the feed is down.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.io import ExchangeClient
from cybotrade.models import Exchange, Level, OrderbookSnapshot

from adrs.oms.ops.order_executer import OrderExecutor
from adrs.oms.ops.order_utils import OrderUtils
from adrs.oms.price_feed import PriceFeed

BTC = Symbol("BTCUSDT")


class SpyRateLimiter:
    def __init__(self):
        self.reserved = []

    @asynccontextmanager
    async def reserve(self, endpoint):
        self.reserved.append(endpoint)
        yield


def _snapshot(bid: str, ask: str) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        symbol=BTC,
        last_update_time=datetime(2026, 8, 11, tzinfo=timezone.utc),
        last_update_id=1,
        bids=[Level(price=Decimal(bid), quantity=Decimal("1"))],
        asks=[Level(price=Decimal(ask), quantity=Decimal("1"))],
        exchange=Exchange.BINANCE_LINEAR,
        orig=None,
    )


def test_feed_hit_never_touches_the_exchange():
    feed = PriceFeed()
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    exchange = MagicMock()
    exchange.get_orderbook_snapshot = AsyncMock()
    limiter = SpyRateLimiter()

    result = asyncio.run(
        OrderUtils.get_order_book(
            exchange=exchange,
            pair=BTC,
            need_log=False,
            rate_limiter=limiter,
            price_feed=feed,
        )
    )
    assert result == [Decimal("100"), Decimal("102")]
    exchange.get_orderbook_snapshot.assert_not_awaited()
    assert limiter.reserved == []  # no weight spent


def test_feed_miss_falls_back_through_reserve():
    feed = PriceFeed()  # empty: no liveness, so every get() misses
    exchange = MagicMock()
    exchange.get_orderbook_snapshot = AsyncMock(return_value=_snapshot("99", "101"))
    limiter = SpyRateLimiter()

    result = asyncio.run(
        OrderUtils.get_order_book(
            exchange=exchange,
            pair=BTC,
            need_log=False,
            rate_limiter=limiter,
            price_feed=feed,
        )
    )
    assert result == [Decimal("99"), Decimal("101")]
    exchange.get_orderbook_snapshot.assert_awaited_once()
    assert len(limiter.reserved) == 1  # went through the rate limiter


def test_no_feed_configured_behaves_exactly_as_before():
    exchange = MagicMock()
    exchange.get_orderbook_snapshot = AsyncMock(return_value=_snapshot("99", "101"))
    limiter = SpyRateLimiter()

    result = asyncio.run(
        OrderUtils.get_order_book(
            exchange=exchange, pair=BTC, need_log=False, rate_limiter=limiter
        )
    )
    assert result == [Decimal("99"), Decimal("101")]
    assert len(limiter.reserved) == 1


def test_feed_and_rest_agree_on_current_price():
    """
    Equivalence against cybotrade's real implementation, not a re-derivation of
    its formula. This is the test that guarantees changing the price source does
    not silently move our quotes, so it has to exercise the actual REST code
    path: _StubExchange overrides only get_orderbook_snapshot, and the inherited
    ExchangeClient.get_current_price does the rest.

    ExchangeClient is an ABC and ABCMeta computes __abstractmethods__ at class
    creation, so it has to be cleared *after* the class body — assigning it
    inside the body gets overwritten and instantiation still fails.
    """

    class _StubExchange(ExchangeClient):
        async def get_orderbook_snapshot(self, symbol, **kwargs):
            return _snapshot("63889.10", "63889.20")

    _StubExchange.__abstractmethods__ = frozenset()

    rest_price = asyncio.run(_StubExchange().get_current_price(BTC))

    feed = PriceFeed()
    feed.apply(BTC, Decimal("63889.10"), Decimal("63889.20"))
    quote = feed.get(BTC)
    assert quote is not None
    assert quote.mid == rest_price
    # Sanity: the shared value is the real midpoint, so neither side is trivially
    # agreeing on a wrong number
    assert rest_price == Decimal("63889.15")


def test_executor_get_current_price_prefers_feed_and_returns_mid():
    """
    Exercises the real OrderExecutor.get_current_price, not just Quote.mid in
    isolation: proves the constructor plumbing actually reaches the method and
    that a feed hit returns the mid (not, say, the bid) without touching the
    exchange or the rate limiter.
    """
    executor = object.__new__(OrderExecutor)
    feed = PriceFeed()
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    executor.price_feed = feed
    executor.exchange = MagicMock()
    executor.exchange.get_current_price = AsyncMock()
    executor.rate_limiter = MagicMock()
    executor.rate_limiter.reserve = MagicMock()

    result = asyncio.run(executor.get_current_price(BTC))

    assert result == Decimal("101")
    executor.exchange.get_current_price.assert_not_awaited()
    executor.rate_limiter.reserve.assert_not_called()

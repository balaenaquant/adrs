"""
Event translation and cache invalidation for the OMS-owned price feed.

The reconnect rule is the one that makes the staleness guard sound: persist_conn
re-runs on_connected on every reconnect, and Binance forces a disconnect every 24
hours, so this path runs daily in normal operation rather than only under failure.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cybotrade import Symbol
from cybotrade.binance import BinancePublicWS
from cybotrade.io.event import Event, EventType
from cybotrade.models import BookTicker, Exchange
from cybotrade.websocket import Message

from adrs.oms.oms import OMS
from adrs.oms.price_feed import PriceFeed

BTC = Symbol("BTCUSDT")


def _oms_with_feed() -> OMS:
    """OMS with __init__ skipped; only the price-feed collaborators are needed."""
    oms = object.__new__(OMS)
    oms.price_feed = PriceFeed()
    return oms


def _oms_for_lifecycle(exchange: Exchange, testnet: bool = False) -> OMS:
    """As above, plus the config the start/stop path reads."""
    oms = _oms_with_feed()
    oms.price_feed_task = None
    oms.config = SimpleNamespace(
        config=SimpleNamespace(
            credentials=SimpleNamespace(exchange=exchange, testnet=testnet),
            base_asset_to_symbol_table={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
        )
    )
    return oms


def _book_ticker(bid: str, ask: str) -> BookTicker:
    return BookTicker(
        symbol=BTC,
        bid=Decimal(bid),
        ask=Decimal(ask),
        bid_qty=Decimal("1"),
        ask_qty=Decimal("1"),
        update_id=1,
        event_time=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )


def test_book_ticker_event_becomes_a_quote():
    oms = _oms_with_feed()
    oms.on_price_feed_event(
        Event(
            event_type=EventType.BookTicker, orig="{}", data=_book_ticker("100", "102")
        )
    )
    quote = oms.price_feed.get(BTC)
    assert quote is not None and quote.bid == Decimal("100")


def test_subscribed_event_clears_the_cache():
    """A Subscribed event means the socket (re)connected: nothing cached is trustworthy."""
    oms = _oms_with_feed()
    oms.on_price_feed_event(
        Event(
            event_type=EventType.BookTicker, orig="{}", data=_book_ticker("100", "102")
        )
    )
    assert oms.price_feed.get(BTC) is not None
    oms.on_price_feed_event(
        Event(event_type=EventType.Subscribed, orig="{}", data=["btcusdt@bookTicker"])
    )
    assert oms.price_feed.get(BTC) is None


def test_unknown_event_refreshes_liveness_without_creating_a_quote():
    oms = _oms_with_feed()
    oms.on_price_feed_event(Event(event_type=EventType.Unknown, orig="junk", data=None))
    assert oms.price_feed.get(BTC) is None
    assert oms.price_feed.stats()["liveness_age_sec"] is not None


def test_large_event_time_divergence_is_logged(caplog):
    """
    A frame whose exchange timestamp is far older than 'now' means the loop
    stalled or our clock has drifted; either way the quote's age understates it.
    """
    oms = _oms_with_feed()
    oms.rate_limiter = MagicMock()
    # Exchange stamped the frame 5s before our synced clock reads
    oms.rate_limiter.get_synced_time_ms = MagicMock(
        return_value=int(datetime(2026, 8, 11, tzinfo=timezone.utc).timestamp() * 1000)
        + 5_000
    )
    with caplog.at_level(logging.WARNING):
        oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
    assert "divergence" in caplog.text.lower()
    # The quote is still stored: this is an observability signal, not a guard
    assert oms.price_feed.get(BTC) is not None


def test_no_feed_is_started_on_a_non_binance_exchange():
    """
    Only Binance has a public adapter. On anything else the OMS must run exactly
    as before, pricing from REST, so no socket and no task may appear.
    """
    oms = _oms_for_lifecycle(Exchange.BYBIT_LINEAR)
    oms._start_price_feed()
    assert oms.price_feed_task is None
    assert not hasattr(oms, "price_feed_ws")


def test_binance_feed_subscribes_the_configured_symbols():
    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR, testnet=True)
        with patch("adrs.oms.oms.BinancePublicWS") as ws_cls:
            ws = ws_cls.return_value
            ws.start = AsyncMock()
            ws.streams = ["btcusdt@bookTicker", "ethusdt@bookTicker"]
            oms._start_price_feed()

            assert ws_cls.call_args.kwargs["symbols"] == ["BTCUSDT", "ETHUSDT"]
            # The feed and the REST fallback must price against the same book
            assert ws_cls.call_args.kwargs["testnet"] is True
            assert ws.on_event == oms._await_price_feed_event
            assert oms.price_feed_task is not None
        oms._stop_price_feed()

    asyncio.run(scenario())


def test_the_real_adapter_can_deliver_into_the_feed():
    """
    Regression: BinancePublicWS does `await self.on_event(event)`, so the callback
    must be a coroutine function. Assigning the plain synchronous handler raises
    "'NoneType' object can't be awaited" on every frame, and start() swallows it
    per-message -- the feed would silently deliver nothing and every price read
    would fall back to REST forever. Drives the real adapter, no mock, so the
    contract is checked rather than assumed.
    """

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)
        oms.rate_limiter = MagicMock()
        oms.rate_limiter.get_synced_time_ms = MagicMock(return_value=0)
        with patch.object(BinancePublicWS, "start", new=AsyncMock()):
            oms._start_price_feed()
        ws = oms.price_feed_ws

        # A reconnect announces itself first, exactly as persist_conn drives it
        await ws.on_connected(None)
        assert oms.price_feed.stats()["liveness_age_sec"] is not None

        await ws.on_message(
            Message.Text(
                json.dumps(
                    {
                        "stream": "btcusdt@bookTicker",
                        "data": {
                            "e": "bookTicker",
                            "s": "BTCUSDT",
                            "b": "100.1",
                            "B": "3",
                            "a": "100.2",
                            "A": "4",
                            "u": 7,
                            "E": 1_760_000_000_000,
                        },
                    }
                )
            )
        )

        quote = oms.price_feed.get(BTC)
        assert quote is not None
        assert quote.bid == Decimal("100.1") and quote.ask == Decimal("100.2")
        oms._stop_price_feed()

    asyncio.run(scenario())


def test_stop_cancels_the_task_and_drops_every_quote():
    """
    Shutdown and a symbol-table change both stop the feed. Nothing cached may
    outlive the connection that produced it.
    """

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)
        oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
        assert oms.price_feed.get(BTC) is not None

        running = asyncio.Event()

        async def never_ends():
            running.set()
            await asyncio.sleep(3600)

        oms.price_feed_task = asyncio.create_task(never_ends())
        await running.wait()
        task = oms.price_feed_task

        oms._stop_price_feed()

        assert oms.price_feed_task is None
        with pytest.raises(asyncio.CancelledError):
            await task
        assert oms.price_feed.get(BTC) is None

    asyncio.run(scenario())

"""
Event translation and cache invalidation for the OMS-owned price feed.

The reconnect rule is the one that makes the staleness guard sound: persist_conn
re-runs on_connected on every reconnect, and Binance forces a disconnect every 24
hours, so this path runs daily in normal operation rather than only under failure.
"""

import asyncio
import json
import logging
import time
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

from adrs.oms.oms import OMS, PRICE_FEED_DIVERGENCE_LOG_INTERVAL_SEC
from adrs.oms.price_feed import PriceFeed

BTC = Symbol("BTCUSDT")


def _oms_with_feed() -> OMS:
    """OMS with __init__ skipped; only the price-feed collaborators are needed."""
    oms = object.__new__(OMS)
    oms.price_feed = PriceFeed()
    # _note_event_time_divergence aggregates over a window, and its whole body is
    # try/except-guarded -- so a missing attribute here would be swallowed and
    # silently log nothing rather than fail loudly.
    oms._divergence_count = 0
    oms._divergence_worst_ms = 0
    oms._divergence_worst_symbol = None
    oms._divergence_window_started_at = None
    oms._divergence_last_emit_at = None
    oms._divergence_emitted_previous_window = False
    return oms


def _diverging_oms(divergence_ms: int) -> OMS:
    """OMS whose synced clock reads `divergence_ms` ahead of every frame."""
    oms = _oms_with_feed()
    oms.rate_limiter = MagicMock()
    oms.rate_limiter.get_synced_time_ms = MagicMock(
        return_value=int(datetime(2026, 8, 11, tzinfo=timezone.utc).timestamp() * 1000)
        + divergence_ms
    )
    return oms


def _feed_frames(oms: OMS, n: int) -> None:
    async def run():
        for _ in range(n):
            await oms.on_price_feed_event(
                Event(
                    event_type=EventType.BookTicker,
                    orig="{}",
                    data=_book_ticker("100", "102"),
                )
            )

    asyncio.run(run())


def _close_divergence_window(oms: OMS) -> None:
    """Backdate the window so the next frame closes it and reports."""
    oms._divergence_window_started_at = (
        time.monotonic() - PRICE_FEED_DIVERGENCE_LOG_INTERVAL_SEC - 1
    )


def _divergence_records(caplog) -> list:
    return [r for r in caplog.records if "divergence" in r.getMessage().lower()]


def _credentials(exchange: Exchange, testnet: bool = False) -> SimpleNamespace:
    """
    Credentials stand-in, compared by value like the real pydantic model, so
    on_refresh_config's "did credentials change" check behaves realistically.
    """
    return SimpleNamespace(
        exchange=exchange,
        testnet=testnet,
        to_exchange_event=lambda: SimpleNamespace(on_event=None, start=AsyncMock()),
    )


def _oms_for_lifecycle(exchange: Exchange, testnet: bool = False) -> OMS:
    """As above, plus the config the start/stop path reads."""
    oms = _oms_with_feed()
    oms.price_feed_task = None
    oms.price_feed_ws = None
    oms.config = SimpleNamespace(
        config=SimpleNamespace(
            credentials=_credentials(exchange, testnet),
            base_asset_to_symbol_table={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
        )
    )
    return oms


def _oms_for_refresh(exchange: Exchange, testnet: bool = False) -> OMS:
    """
    As above, plus what on_refresh_config touches. `config` becomes a ConfigManager
    stand-in whose `.config` is the object _oms_for_lifecycle built, so `refresh()`
    can mutate it in place the way the real one does.
    """
    oms = _oms_for_lifecycle(exchange, testnet=testnet)
    oms.rate_limiter = MagicMock()
    # Real rate limiters return epoch ms; a bare MagicMock would make the
    # divergence comparison a MagicMock-vs-int TypeError.
    oms.rate_limiter.get_synced_time_ms = MagicMock(return_value=0)
    oms.exchange_events_task = None
    oms.opm = MagicMock()
    manager = MagicMock()
    manager.config = oms.config.config
    manager.refresh = AsyncMock()
    manager.update_symbol_info = AsyncMock()
    oms.config = manager
    return oms


def _refresh_changes_credentials_to(oms: OMS, credentials: SimpleNamespace) -> None:
    """Make the next refresh() swap in different credentials, as a real one would."""

    def _swap():
        oms.config.config.credentials = credentials

    oms.config.refresh = AsyncMock(side_effect=_swap)


def _async_cm(return_value):
    """Minimal async context manager that yields return_value."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def _never_ends(started: asyncio.Event | None = None) -> None:
    """Stand-in for a live feed task: only cancellation ends it."""
    if started is not None:
        started.set()
    await asyncio.sleep(3600)


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
    asyncio.run(
        oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
    )
    quote = oms.price_feed.get(BTC)
    assert quote is not None and quote.bid == Decimal("100")


def test_subscribed_event_clears_the_cache():
    """A Subscribed event means the socket (re)connected: nothing cached is trustworthy."""
    oms = _oms_with_feed()
    asyncio.run(
        oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
    )
    assert oms.price_feed.get(BTC) is not None
    asyncio.run(
        oms.on_price_feed_event(
            Event(
                event_type=EventType.Subscribed,
                orig="{}",
                data=["btcusdt@bookTicker"],
            )
        )
    )
    assert oms.price_feed.get(BTC) is None


def test_unknown_event_refreshes_liveness_without_creating_a_quote():
    oms = _oms_with_feed()
    asyncio.run(
        oms.on_price_feed_event(
            Event(event_type=EventType.Unknown, orig="junk", data=None)
        )
    )
    assert oms.price_feed.get(BTC) is None
    assert oms.price_feed.stats()["liveness_age_sec"] is not None


def test_a_burst_of_diverging_frames_logs_nothing_until_the_window_closes(caplog):
    """
    The regression this exists for. bookTicker delivers hundreds of frames a
    second, and a stall delays every frame queued behind it -- so per-frame
    logging turned one 9-second stall into 156 near-identical WARNINGs inside a
    minute. 200 diverging frames inside one window must produce no output.
    """
    oms = _diverging_oms(divergence_ms=5_000)
    with caplog.at_level(logging.INFO):
        _feed_frames(oms, 200)
    assert _divergence_records(caplog) == []
    # ...but every one was counted, ready for the summary
    assert oms._divergence_count == 200
    # The quotes are still stored: this is observability, not a guard
    assert oms.price_feed.get(BTC) is not None


def test_the_window_summary_reports_the_count_and_the_worst_value(caplog):
    oms = _diverging_oms(divergence_ms=5_000)
    _feed_frames(oms, 40)
    # A worse frame lands in the same window
    oms.rate_limiter.get_synced_time_ms = MagicMock(
        return_value=int(datetime(2026, 8, 11, tzinfo=timezone.utc).timestamp() * 1000)
        + 9_784
    )
    _feed_frames(oms, 1)
    _close_divergence_window(oms)

    with caplog.at_level(logging.INFO):
        _feed_frames(oms, 1)

    records = _divergence_records(caplog)
    assert len(records) == 1, "one stall, one line"
    msg = records[0].getMessage()
    assert "42 frame(s)" in msg, msg
    assert "9784ms" in msg, msg
    assert "BTCUSDT" in msg, msg
    # Counters reset, so the next window starts clean
    assert oms._divergence_count == 0
    assert oms._divergence_worst_ms == 0


def test_an_isolated_stall_is_info_and_a_persistent_one_escalates(caplog):
    """
    A single startup or GC stall is not actionable; the same thing two windows
    running means the loop is persistently behind and quote age is understating
    staleness for real.
    """
    oms = _diverging_oms(divergence_ms=5_000)

    _feed_frames(oms, 3)
    _close_divergence_window(oms)
    with caplog.at_level(logging.INFO):
        _feed_frames(oms, 1)
    first = _divergence_records(caplog)
    assert len(first) == 1 and first[0].levelno == logging.INFO

    caplog.clear()
    _feed_frames(oms, 3)
    _close_divergence_window(oms)
    with caplog.at_level(logging.INFO):
        _feed_frames(oms, 1)
    second = _divergence_records(caplog)
    assert len(second) == 1 and second[0].levelno == logging.WARNING


def test_frames_within_the_threshold_never_log_or_open_a_window(caplog):
    oms = _diverging_oms(divergence_ms=100)  # well under the 2s threshold
    with caplog.at_level(logging.INFO):
        _feed_frames(oms, 50)
    assert _divergence_records(caplog) == []
    assert oms._divergence_count == 0
    assert oms._divergence_window_started_at is None


def test_a_missing_rate_limiter_cannot_break_the_feed(caplog):
    """
    The metric is guarded end to end: a frame must still become a quote even if
    the divergence check itself blows up.
    """
    oms = _oms_with_feed()
    oms.rate_limiter = SimpleNamespace()  # no get_synced_time_ms at all
    with caplog.at_level(logging.INFO):
        _feed_frames(oms, 1)
    assert oms.price_feed.get(BTC) is not None


def test_no_feed_is_started_on_a_non_binance_exchange():
    """
    Only Binance has a public adapter. On anything else the OMS must run exactly
    as before, pricing from REST, so no socket and no task may appear.
    """
    oms = _oms_for_lifecycle(Exchange.BYBIT_LINEAR)
    oms._start_price_feed()
    assert oms.price_feed_task is None
    assert oms.price_feed_ws is None


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
            assert ws.on_event == oms.on_price_feed_event
            assert oms.price_feed_task is not None
        oms._stop_price_feed()

    asyncio.run(scenario())


def test_the_real_adapter_can_deliver_into_the_feed():
    """
    Regression: BinancePublicWS does `await self.on_event(event)`, so
    on_price_feed_event must stay a coroutine function. Make it a plain `def` and
    every frame raises ("a coroutine or an awaitable is required"), which start()
    swallows per-message -- the feed would log "Started", cache nothing, and every
    price read would fall back to REST forever, with the task looking healthy.

    Drives the real adapter rather than calling the handler directly, so the
    contract with cybotrade is checked instead of assumed.
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
        await oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
        assert oms.price_feed.get(BTC) is not None

        running = asyncio.Event()
        oms.price_feed_task = asyncio.create_task(_never_ends(running))
        await running.wait()
        task = oms.price_feed_task

        oms._stop_price_feed()

        assert oms.price_feed_task is None
        assert oms.price_feed_ws is None
        # wait_for, not a bare await: if the cancel is ever dropped this must fail
        # on a timeout rather than hang for the hour the stand-in sleeps.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert oms.price_feed.get(BTC) is None

    asyncio.run(scenario())


def test_starting_twice_does_not_orphan_the_running_socket():
    """
    Start without a stop must be a no-op. Overwriting price_feed_task would leave
    a live socket nobody can cancel, still writing into the shared cache.
    """

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)
        running = asyncio.Event()
        oms.price_feed_task = asyncio.create_task(_never_ends(running))
        await running.wait()
        task = oms.price_feed_task

        with patch.object(BinancePublicWS, "start", new=AsyncMock()):
            oms._start_price_feed()

        assert oms.price_feed_task is task
        oms._stop_price_feed()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())


def test_handle_shutdown_stops_the_feed():
    """Shutdown must take the socket down with it, not leave it writing quotes."""

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)
        oms.opm = MagicMock()
        oms.opm.order_pools.get_order_pool = MagicMock(return_value=_async_cm({}))

        running = asyncio.Event()
        oms.price_feed_task = asyncio.create_task(_never_ends(running))
        await running.wait()
        task = oms.price_feed_task

        # Empty order pool: _handle_shutdown exits via SystemExit(0)
        with pytest.raises(SystemExit):
            await oms._handle_shutdown()

        assert oms.price_feed_task is None
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())


def test_a_task_that_ends_on_its_own_is_logged_as_an_error(caplog):
    """
    cybotrade's _stream catches stream errors, prints and breaks, so start()
    returns *normally* when a live socket dies. Nothing awaits the task, so this
    log is the only OMS-side signal that the feed went dead.
    """

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)

        async def ends_immediately():
            return None

        task = asyncio.create_task(ends_immediately())
        oms._supervise_task(task, "PRICE_FEED")
        await task
        await asyncio.sleep(0)  # let the done-callback run

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert "stopped delivering" in caplog.text


def test_a_task_that_raises_is_logged_with_its_exception(caplog):
    """
    persist_conn_with retries only reconnects, so a failure to connect at all
    escapes start() into the task. Logging it also retrieves it.
    """

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)

        async def boom():
            raise RuntimeError("connect refused")

        task = asyncio.create_task(boom())
        oms._supervise_task(task, "PRICE_FEED")
        with pytest.raises(RuntimeError):
            await task
        await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert "connect refused" in caplog.text


def test_a_feed_that_fails_to_connect_is_logged(caplog):
    """
    End-to-end for the start site, not just the helper: persist_conn_with retries
    only reconnects, so a failure to connect at all comes straight back out of
    start() and would otherwise vanish into a task nobody retrieves.
    """

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)
        with patch.object(
            BinancePublicWS,
            "start",
            new=AsyncMock(side_effect=RuntimeError("connect refused")),
        ):
            oms._start_price_feed()
            task = oms.price_feed_task
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(task, timeout=5)
            await asyncio.sleep(0)
        oms._stop_price_feed()

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert "connect refused" in caplog.text


def test_a_cancelled_task_is_not_logged_as_an_error(caplog):
    """Cancellation is how we stop the feed on purpose; it must not read as a fault."""

    async def scenario():
        oms = _oms_for_lifecycle(Exchange.BINANCE_LINEAR)
        running = asyncio.Event()
        task = asyncio.create_task(_never_ends(running))
        oms._supervise_task(task, "PRICE_FEED")
        await running.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert caplog.text == ""


def test_credentials_change_restarts_the_feed():
    """
    Credentials carry the exchange and the testnet flag, so the connection they
    produced is invalid once they change: new socket, and nothing cached carried over.
    """

    async def scenario():
        oms = _oms_for_refresh(Exchange.BINANCE_LINEAR)
        with patch.object(BinancePublicWS, "start", new=AsyncMock()):
            oms._start_price_feed()
            first_task, first_ws = oms.price_feed_task, oms.price_feed_ws
            await oms.on_price_feed_event(
                Event(
                    event_type=EventType.BookTicker,
                    orig="{}",
                    data=_book_ticker("100", "102"),
                )
            )
            assert oms.price_feed.get(BTC) is not None

            # Same exchange, different environment: still a different book
            _refresh_changes_credentials_to(
                oms, _credentials(Exchange.BINANCE_LINEAR, testnet=True)
            )
            await oms.on_refresh_config()

            assert oms.price_feed_task is not None
            assert oms.price_feed_task is not first_task
            assert oms.price_feed_ws is not first_ws
            assert oms.price_feed_ws.testnet is True
            # No quote may survive the connection that produced it
            assert oms.price_feed.get(BTC) is None

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first_task, timeout=5)
            oms._stop_price_feed()

    asyncio.run(scenario())


def test_switching_away_from_binance_stops_the_feed():
    """
    The hazard this guards: a Binance feed left running after a switch to Bybit
    would keep serving Binance quotes to a Bybit executor through the shared cache.
    """

    async def scenario():
        oms = _oms_for_refresh(Exchange.BINANCE_LINEAR)
        with patch.object(BinancePublicWS, "start", new=AsyncMock()):
            oms._start_price_feed()
        first_task = oms.price_feed_task
        await oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
        assert oms.price_feed.get(BTC) is not None

        _refresh_changes_credentials_to(oms, _credentials(Exchange.BYBIT_LINEAR))
        await oms.on_refresh_config()

        assert oms.price_feed_task is None
        assert oms.price_feed_ws is None
        assert oms.price_feed.get(BTC) is None
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first_task, timeout=5)

    asyncio.run(scenario())

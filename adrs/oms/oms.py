import os
import json
import copy
import time
import asyncio
import logging
import signal

from typing import Dict, Protocol
from decimal import Decimal
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta

from aion import Scheduler, Trigger
from nats_client import Msg
from adrs.data import MetricStream, MetricBuilder
from adrs.data.connector import DEFAULT_METRIC_NAMESPACE
from adrs.subjects import portfolio_signal_subject, oms_command_subject
from cybotrade import Symbol
from cybotrade.binance import BinancePublicWS
from cybotrade.io.event import Event, EventType
from cybotrade.models import Position, OrderSide, OrderStatus, Exchange

from adrs.oms.config import ConfigManager
from adrs.oms.price_feed import PriceFeed
from adrs.oms.ops.order_executer import OrderExecutor, MAX_CONCURRENT_ORDER_OPS
from adrs.oms.ops.order_pool import CancelBacklogs
from adrs.oms.position import PositionManager
from adrs.oms.risk import RiskEngine
from adrs.oms.ops.order_placement_manager import OrderPlacementManager
from adrs.oms.rate_limit.rate_limiter import RateLimiter
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.error_policy import ExchangeErrorPolicy

logger = logging.getLogger(__name__)

# A record whose order the exchange history still can't find after this long
# is unrecoverable; keeping it would poll the shared budget forever
AEGIS_NOT_FOUND_GIVE_UP = timedelta(hours=1)

# A frame this much older than our synced clock means the event loop stalled or
# our clock drifted. Reported rather than enforced: the staleness guard stays on
# the monotonic clock, which is immune to both.
PRICE_FEED_EVENT_TIME_DIVERGENCE_WARN_MS = 2_000

# How often the divergence summary may be emitted. Detection stays per frame;
# only the reporting is throttled.
#
# A stall delays every frame queued behind it, and bookTicker delivers hundreds
# of frames a second, so logging per frame described a single event hundreds of
# times: one 9-second startup stall produced 156 near-identical WARNINGs inside
# one minute, burying the rest of the log. The aggregated line carries the count
# and the worst value, which says more than any single occurrence did.
PRICE_FEED_DIVERGENCE_LOG_INTERVAL_SEC = 60.0


class PortfolioSignal(BaseModel):
    assets: Dict[str, Decimal]
    timestamp: int


class OMSEventHandler(Protocol):
    """
    Observer for key OMS state transitions. Pass an implementation to OMS.__init__
    to receive structured callbacks instead of parsing log output.

    All methods are synchronous — keep them fast (record to a queue, increment a
    counter). The default (None) skips every call.
    """

    def on_signal_received(self, signal: "PortfolioSignal") -> None: ...
    def on_desired_updated(self, symbol: "Symbol", quantity: "Decimal") -> None: ...
    def on_signal_skipped(self, symbol: "Symbol", reason: str) -> None: ...


def generate_cron(total_seconds: int):
    if total_seconds < 1:
        return "* * * * * *"

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{seconds} {minutes} */{hours} * * *"
    elif minutes > 0:
        return f"{seconds} */{minutes} * * * *"
    else:
        return f"*/{seconds} * * * * *"


def getenv(name: str) -> str:
    env = os.getenv(name)
    if env is None:
        raise ValueError(f"{name} is not present in environment")
    return env


class OMS:
    # Subclasses override these to swap in custom implementations without
    # touching __init__. executor_cls is forwarded to OrderPlacementManager.
    executor_cls: type[OrderExecutor] = OrderExecutor
    position_cls: type[PositionManager] = PositionManager
    opm_cls: type[OrderPlacementManager] = OrderPlacementManager
    risk_cls: type[RiskEngine] = RiskEngine

    def __init__(
        self,
        config: ConfigManager,
        metric_stream: MetricStream,
        rate_limiter: RateLimiter,
        error_policy: ExchangeErrorPolicy | None = None,
        insert_prefix: str = DEFAULT_METRIC_NAMESPACE,
        signal_namespace: str | None = None,
        observer: OMSEventHandler | None = None,
    ):
        super().__init__()
        self.config = config
        self.metric_stream = metric_stream
        self.metric_builder = MetricBuilder(self.metric_stream, insert_prefix)
        # Must match the PortfolioExecutor's namespace so the OMS subscribes to
        # the same `portfolio_signal.<ns>.<portfolio_id>` the executor publishes.
        self.signal_namespace = signal_namespace
        self.observer = observer
        # Per-exchange error classification; derive from config when not injected
        error_policy = error_policy or config.config.credentials.to_error_policy()
        self.position = self.position_cls(
            config=config,
            rate_limiter=rate_limiter,
        )
        # Owned here, not by the OPM: the feed serves both the executor's reads
        # and the signal recompute, and a later Bybit adapter will share it.
        self.price_feed = PriceFeed()
        # Aggregation state for the event-time divergence report; see
        # _note_event_time_divergence. Monotonic, so an NTP step cannot make a
        # window look finished or eternal.
        self._divergence_count = 0
        self._divergence_worst_ms = 0
        self._divergence_worst_symbol: str | None = None
        self._divergence_window_started_at: float | None = None
        self._divergence_last_emit_at: float | None = None
        self._divergence_emitted_previous_window = False
        self.opm = self.opm_cls(
            position=self.position,
            config=self.config,
            rate_limiter=rate_limiter,
            error_policy=error_policy,
            executor_cls=self.executor_cls,
            price_feed=self.price_feed,
        )
        self.risk = self.risk_cls(
            config=self.config,
            position=self.position,
            rate_limiter=rate_limiter,
            on_breach=self._trigger_shutdown,
        )
        self.scheduler = Scheduler()
        self.previous_signal: PortfolioSignal | None = None
        self.latest_signal: PortfolioSignal | None = None
        self.rate_limiter = rate_limiter
        # Initialised in run(); set here so on_refresh_config is safe to call
        # before run() starts (e.g. in tests).
        self.exchange_events_task: asyncio.Task | None = None
        # Initialised in run(); set here so on_refresh_config and shutdown are
        # safe to call before run() starts (e.g. in tests).
        self.price_feed_task: asyncio.Task | None = None
        self.price_feed_ws: BinancePublicWS | None = None

    async def init(self):
        """Initialise the OMS state when first started."""
        logger.info("[INIT] Initilalizing position values")

        # LATEST SIGNAL
        self.latest_signal = await self.get_latest_signal(
            self.config.config.portfolio_id,
        )
        logger.info(f"Latest signal at startup {self.latest_signal}")

        # ORDER POOL + POSITION share one open-orders snapshot
        snapshot = await self.opm.order_pools.fetch_open_orders_snapshot()
        await self.opm.order_pools.resync_order_pool(snapshot)
        await self.position.update_exchange()
        self.position.update_pending(snapshot)
        for s in self.config.config.base_asset_to_symbol_table.values():
            symbol = Symbol(s)
            if symbol not in self.position.pending.keys():
                self.position.pending[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal("0"),
                    entry_price=Decimal("0"),
                    updated_time=datetime.now(tz=timezone.utc),
                )
            if symbol not in self.position.exchange.keys():
                self.position.exchange[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal("0"),
                    entry_price=Decimal("0"),
                    updated_time=datetime.now(tz=timezone.utc),
                )

            # Init using desired and exchange
            quantity = (
                self.position.exchange[symbol].quantity
                + self.position.pending[symbol].quantity
            )
            self.position.desired[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=Decimal("0"),
                updated_time=datetime.now(tz=timezone.utc),
            )

    async def get_latest_signal(self, portfolio_id: str) -> PortfolioSignal | None:
        """
        Cold-start hook: fetch the last known signal so the OMS does not start
        blind. The base has no signal store, so it starts without one and waits
        for the first live signal; subclasses override to read their own store.
        """
        return None

    def _setup_signals(self):
        """
        SIGTERM stops the scheduler, not _handle_shutdown() directly: run()'s
        finally already calls _handle_shutdown() exactly once whenever
        scheduler.start() returns, for any reason. Calling it again here
        would race that (double order-cancel pass, and a SystemExit raised
        inside this handler's own task wouldn't propagate to end the
        process anyway - it would just be a swallowed background-task
        exception). Stopping the scheduler reuses the existing, single
        shutdown path instead of adding a second one.
        """
        loop = asyncio.get_event_loop()

        loop.add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(self.scheduler.shutdown())
        )

    async def _trigger_shutdown(self):
        """
        Single shutdown entry point for risk breaches (and anything else that
        must stop the OMS from inside a running handler): stop the scheduler,
        which makes scheduler.start() return, and run()'s finally performs the
        one order-cancel pass and process exit. Calling _handle_shutdown()
        directly from a handler would cancel orders but leave the scheduler
        ticking - the next placement tick would put every position right back.
        (Same reasoning as the SIGTERM handler, see _setup_signals.)
        """
        logger.error("[SHUTDOWN] Triggered - stopping scheduler")
        await self.scheduler.shutdown()

    async def _handle_shutdown(self):
        """To close all pending orders on shutdown"""
        logger.info("Shutdown signal received. Cancelling orders...")

        self._stop_price_feed()

        # Snapshot under the pool lock, then release BEFORE the gather: each
        # cancel_single_order re-acquires the same lock, so holding it across
        # the gather would self-deadlock.
        async with self.opm.order_pools.get_order_pool() as order_pool:
            orders_to_cancel = list(order_pool.values())

        sem = asyncio.Semaphore(MAX_CONCURRENT_ORDER_OPS)

        async def _bounded_cancel(symbol: str, client_order_id: str):
            async with sem:
                return await self.opm.executor.cancel_single_order(
                    Symbol(symbol), client_order_id
                )

        cancel_tasks = [
            _bounded_cancel(order.symbol, order.client_order_id)
            for order in orders_to_cancel
        ]

        if not cancel_tasks:
            raise SystemExit(0)

        cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

        cancel_retries = []
        for result in cancel_results:
            if isinstance(result, Exception):
                logger.error(f"Failed to cancel order: {result}")
            if isinstance(result, CancelBacklogs):
                cancel_retries.append(result)

        if not cancel_retries:
            logger.info("All orders cancelled successfully.")
            return

        logger.warning(f"Retrying {len(cancel_retries)} cancellations in 60s...")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.error("Shutdown forced during retry wait.")
            return

        retry_tasks = [
            _bounded_cancel(retry.symbol, retry.client_order_id)
            for retry in cancel_retries
        ]

        if retry_tasks:
            await asyncio.gather(*retry_tasks, return_exceptions=True)
            logger.info("Retry attempts finished.")

        raise SystemExit(0)

    def _supervise_task(self, task: asyncio.Task, label: str) -> None:
        """
        Log when a long-lived stream task ends. Logging only, never restarts.

        Nothing ever awaits these tasks, so without this "started and healthy" and
        "started and dead" are indistinguishable from the OMS. Both endings are
        silent today:

        - cybotrade's `_stream` catches stream errors, prints and breaks, so
          `start()` returns *normally* once a live socket dies.
        - `persist_conn_with` retries only *reconnects*; it uses `?` on the initial
          connect, and in `_stream` that await sits outside the try, so failing to
          connect at all escapes into the task, where nobody retrieves it.

        Deliberately does not restart: a restart loop against a hard-failing
        connect would hammer the exchange, and the price feed already fails safe
        (PriceFeed withholds quotes once liveness lapses, so reads go to REST).
        Retrieving the exception here also stops asyncio complaining about it.
        """

        def _on_done(finished: asyncio.Task) -> None:
            if finished.cancelled():
                logger.info(f"[{label}] Task cancelled")
                return
            exc = finished.exception()
            if exc is not None:
                logger.error(f"[{label}] Task failed: {exc!r}", exc_info=exc)
                return
            logger.error(
                f"[{label}] Task ended on its own, so the stream stopped "
                f"delivering and this connection will not come back"
            )

        task.add_done_callback(_on_done)

    async def on_price_feed_event(self, event: Event) -> None:
        """
        Translate public market-data events into the feed.

        Every event refreshes liveness, including ones that carry no quote: an
        unparseable frame still proves the socket is delivering. Subscribed means
        the connection just (re)established, so nothing cached survives it.

        Async to satisfy the adapter's callback contract — BinancePublicWS does
        `await self.on_event(event)`, and a plain def there raises on every frame
        — not because the translation itself needs to await anything.
        """
        match event.event_type:
            case EventType.BookTicker:
                book_ticker = event.data
                self._note_event_time_divergence(book_ticker)
                self.price_feed.apply(
                    book_ticker.symbol, book_ticker.bid, book_ticker.ask
                )
            case EventType.Subscribed:
                logger.info(f"[PRICE_FEED] (Re)subscribed to {event.data}, clearing")
                self.price_feed.clear()
                self.price_feed.note_message()
            case _:
                self.price_feed.note_message()

    def _note_event_time_divergence(self, book_ticker) -> None:
        """
        Record a frame whose exchange timestamp trails our synced clock, and
        report at most one summary per PRICE_FEED_DIVERGENCE_LOG_INTERVAL_SEC.

        Detection is per frame; reporting is not. One stall delays every frame
        behind it, so a per-frame line describes a single event once per frame --
        156 of them in a minute, observed in production. Count plus worst value
        is both quieter and more informative.

        Severity escalates rather than starting loud: an isolated stall (startup,
        a GC pause) is not actionable, whereas the same thing two windows running
        means the loop is persistently behind and quote age is understating
        staleness for real.

        The whole body is guarded. "Never let a metric break the feed" only holds
        if every step is covered, and a raise here would propagate through
        on_price_feed_event into the adapter, where start() swallows it per frame
        -- losing the quote and leaving no OMS-level trace.
        """
        try:
            now = time.monotonic()
            exchange_ms = int(book_ticker.event_time.timestamp() * 1000)
            divergence_ms = self.rate_limiter.get_synced_time_ms() - exchange_ms

            if divergence_ms > PRICE_FEED_EVENT_TIME_DIVERGENCE_WARN_MS:
                if self._divergence_window_started_at is None:
                    self._divergence_window_started_at = now
                    # A stall long after the last report is a fresh incident, not
                    # a continuation, so it starts quiet again.
                    last = self._divergence_last_emit_at
                    if (
                        last is not None
                        and now - last > 2 * PRICE_FEED_DIVERGENCE_LOG_INTERVAL_SEC
                    ):
                        self._divergence_emitted_previous_window = False
                self._divergence_count += 1
                if divergence_ms > self._divergence_worst_ms:
                    self._divergence_worst_ms = divergence_ms
                    self._divergence_worst_symbol = str(book_ticker.symbol)

            started = self._divergence_window_started_at
            if (
                started is None
                or now - started < PRICE_FEED_DIVERGENCE_LOG_INTERVAL_SEC
            ):
                return

            count = self._divergence_count
            worst = self._divergence_worst_ms
            symbol = self._divergence_worst_symbol
            elapsed = now - started
            # Reset before emitting, so a failure inside logging cannot wedge the
            # window open and suppress every later report.
            self._divergence_count = 0
            self._divergence_worst_ms = 0
            self._divergence_worst_symbol = None
            self._divergence_window_started_at = None
            self._divergence_last_emit_at = now

            message = (
                f"[PRICE_FEED] Event-time divergence: {count} frame(s) in the "
                f"last {elapsed:.0f}s, worst {worst}ms ({symbol}). The loop "
                f"stalled or the clock has drifted, so quote age understates "
                f"real staleness"
            )
            if self._divergence_emitted_previous_window:
                logger.warning(message)
            else:
                logger.info(message)
            self._divergence_emitted_previous_window = True
        except Exception:
            return

    def _start_price_feed(self) -> None:
        """
        Start the public feed for the configured symbols.

        Only Binance has an adapter today; on any other exchange the OMS runs
        exactly as before, reading prices from REST.
        """
        if self.price_feed_task is not None and not self.price_feed_task.done():
            # Starting without stopping would orphan the running socket, which
            # would keep writing into the shared cache with nobody able to cancel it.
            logger.warning(
                "[PRICE_FEED] Already running, not starting a second connection"
            )
            return
        if self.config.config.credentials.exchange is not Exchange.BINANCE_LINEAR:
            logger.info(
                "[PRICE_FEED] No public feed for "
                f"{self.config.config.credentials.exchange}, using REST prices"
            )
            return
        symbols = list(self.config.config.base_asset_to_symbol_table.values())
        # testnet must match the credentials: the feed and the REST fallback have
        # to price against the same book, and the testnet book is not the live one.
        self.price_feed_ws = BinancePublicWS(
            symbols=symbols,
            testnet=self.config.config.credentials.testnet,
        )
        self.price_feed_ws.on_event = self.on_price_feed_event
        self.price_feed_task = asyncio.create_task(self.price_feed_ws.start())
        self._supervise_task(self.price_feed_task, "PRICE_FEED")
        logger.info(f"[PRICE_FEED] Started for {self.price_feed_ws.streams}")

    def _stop_price_feed(self) -> None:
        if self.price_feed_task is not None:
            self.price_feed_task.cancel()
            self.price_feed_task = None
        # Drop the adapter with the task: after a switch away from Binance this
        # would otherwise still point at the dead connection's adapter.
        self.price_feed_ws = None
        self.price_feed.clear()

    async def on_refresh_config(self):
        """To refresh and update config if there are any changes made during runtime"""
        old_config = copy.deepcopy(self.config.config)
        await self.config.refresh()
        self.opm.executor.config = self.config.config

        # Both branches below can invalidate the feed's connection. Restart it
        # once, after they have run, so a refresh that changes both does not open
        # a socket only to cancel it.
        needs_price_feed_restart = False

        # Restart with a new exchange events when credentials have changed
        if old_config.credentials != self.config.config.credentials:
            logger.info(
                "Detected credentials update, refreshing exchange event handler..."
            )
            if self.exchange_events_task is not None:
                self.exchange_events_task.cancel()
            self.exchange_event = self.config.config.credentials.to_exchange_event()
            self.exchange_event.on_event = self.opm.on_exchange_event
            self.exchange_events_task = asyncio.create_task(self.exchange_event.start())
            self._supervise_task(self.exchange_events_task, "EXCHANGE_EVENTS")
            self.opm.executor.exchange = self.config.exchange
            self.opm.order_pools.exchange = self.config.exchange
            # The exchange and the testnet flag both live in credentials, and both
            # decide whether there is a feed at all and which book it reads. A
            # feed left running after this would quote the previous exchange.
            needs_price_feed_restart = True

        if (
            old_config.base_asset_to_symbol_table
            != self.config.config.base_asset_to_symbol_table
        ):
            try:
                await self.config.update_symbol_info(self.rate_limiter, force=True)
            except Exception as e:
                logger.warning(f"update symbol info failed due to {e}")

            await self.init()

            # The stream list is encoded in the feed's URL, so a changed symbol
            # table needs a new connection.
            needs_price_feed_restart = True

        if needs_price_feed_restart:
            # clear() happens in _stop_price_feed: quotes for symbols we may no
            # longer trade, or from an exchange we no longer talk to, must not linger.
            self._stop_price_feed()
            self._start_price_feed()

    async def on_portfolio_signal(self, msg: Msg):
        """To store the latest signal from portfolio server"""
        try:
            payload = json.loads(msg.data.decode())
            portfolio_signal = PortfolioSignal(
                assets=payload["assets"],
                timestamp=int(payload["timestamp"]),
            )
            logger.info(f"Portfolio signal retrieved {portfolio_signal}")

            if sum(abs(p) for p in portfolio_signal.assets.values()) > Decimal("1"):
                logger.warning(
                    "[ON_PROCESS_LATEST_SIGNAL] Current sum of positions in latest signal is more than 1"
                )
                return

            unknown_assets = [
                asset
                for asset in portfolio_signal.assets
                if asset not in self.config.config.base_asset_to_symbol_table
            ]
            if unknown_assets:
                logger.error(
                    f"[ON_PORTFOLIO_SIGNAL] Rejecting signal with unknown asset(s) "
                    f"{unknown_assets}, not in base_asset_to_symbol_table"
                )
                return

            self.latest_signal = portfolio_signal
            if self.observer is not None:
                self.observer.on_signal_received(portfolio_signal)
        except Exception as e:
            logger.error(f"Failed to process portfolio signal due to {e}")

    async def on_command(self, msg: Msg):
        """Control-plane callback: an operator sent a command on this OMS's
        command subject. Run the matching operation and, when the caller used
        request-reply (msg.reply set), publish the outcome back.

        One reply point with a result dict keeps every path — parse error,
        unknown command, failure, success — replying exactly once.
        """
        try:
            payload = json.loads(msg.data.decode())
            command = payload["command"]
        except Exception as e:
            logger.error(f"[ON_COMMAND] Failed to parse command message: {e}")
            await self._reply(msg, {"status": "error", "error": f"bad message: {e}"})
            return

        try:
            if command == "rebalance":
                await self.rebalance()
                result = {"status": "ok", "command": command}
            else:
                logger.warning(f"[ON_COMMAND] Unknown command: {command}")
                result = {"status": "error", "error": f"unknown command: {command}"}
        except Exception as e:
            logger.error(f"[ON_COMMAND] Command '{command}' failed: {e}")
            result = {"status": "error", "command": command, "error": str(e)}

        await self._reply(msg, result)

    async def _reply(self, msg: Msg, result: dict):
        """Publish a result to the requester's reply inbox. No-op for
        fire-and-forget callers, whose msg.reply is empty."""
        reply_to = getattr(msg, "reply", "")
        if not reply_to:
            return
        try:
            await self.metric_stream.publish(reply_to, json.dumps(result).encode())
        except Exception as e:
            logger.error(f"[ON_COMMAND] Failed to reply on {reply_to}: {e}")

    async def on_process_latest_signal(self):
        """
        Periodic: recompute desired from the latest signal, skipping assets
        whose weight is unchanged. Use rebalance() to force a recompute at the
        current price even when the signal is static.
        """
        await self._recompute_desired(force=False)

    async def _recompute_desired(self, force: bool = False):
        """
        Turn the latest signal into desired positions at the current price.

        Desired quantity is price-dependent (balance * leverage * weight /
        price), so with force=True the weight-unchanged skip is bypassed and
        desired is repriced against a moved market even if the signal has not
        changed.
        """
        if not self.latest_signal:
            logger.warning("[RECOMPUTE_DESIRED] There is no signal to act on")
            return

        market_quotes: dict[Symbol, Decimal] = {}
        PRECISION_4 = Decimal("0.0001")
        symbol_table = self.config.config.base_asset_to_symbol_table

        # First pass: decide which assets need a recompute, before touching
        # any shared state
        to_recompute: dict[str, Decimal] = {}
        for asset, weightage in self.latest_signal.assets.items():
            symbol = Symbol(symbol_table[asset])
            # .get: a newly added asset has no entry in the previous signal
            previous_weightage = (
                self.previous_signal.assets.get(asset) if self.previous_signal else None
            )
            if (
                not force
                and previous_weightage is not None
                and previous_weightage.quantize(exp=PRECISION_4)
                == weightage.quantize(exp=PRECISION_4)
            ):
                logger.info(
                    f"There is no significant change in signal weights for {symbol}, it is ignored"
                )
                if self.observer is not None:
                    self.observer.on_signal_skipped(symbol, "no_change")
                continue
            to_recompute[asset] = weightage

        if to_recompute:
            recompute_symbols = {
                str(Symbol(symbol_table[asset])) for asset in to_recompute
            }
            async with self.opm.order_pools.get_order_backlog() as order_backlog:
                order_backlog[:] = [
                    b for b in order_backlog if b.symbol not in recompute_symbols
                ]

        for asset, weightage in to_recompute.items():
            symbol = Symbol(symbol_table[asset])
            if symbol not in market_quotes.keys():
                price = await self.opm.executor.get_current_price(symbol=symbol)
                if price is None:
                    logger.warning(
                        f"[RECOMPUTE_DESIRED] No price for {symbol}, skipping this cycle"
                    )
                    return
                market_quotes[symbol] = price

            quantity = self.position.compute_base_quantity(
                price=market_quotes[symbol], weightage=weightage
            )
            self.position.desired[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=market_quotes[symbol],
                updated_time=datetime.now(tz=timezone.utc),
            )
            desired_position = self.position.desired[symbol]
            logger.debug(
                f"[RECOMPUTE_DESIRED] {self.config.config.portfolio_id} wants {desired_position.quantity} of {desired_position.symbol} at the price of {desired_position.entry_price}"
            )
            if self.observer is not None:
                self.observer.on_desired_updated(symbol, desired_position.quantity)

        self.position.desired = self.risk.cap_desired(self.position.desired)
        self.previous_signal = self.latest_signal

    async def rebalance(self):
        """
        Reprice desired against the current market even when the signal is
        static (bypassing the weight-unchanged skip). Placement is left to the
        scheduled on_order_placement tick, so this never races the cron and
        needs no placement lock.
        """
        logger.info("[REBALANCE] repricing desired at current market")
        await self._recompute_desired(force=True)

    async def on_aegis_update(self):
        """
        To upsert latest equity value from exchange to aegis``
        """
        logger.info("[ON_AEGIS_UPDATE]")

        # The 60s cadence this handler runs on is the liveness anchor for the
        # streamed exchange position (see POSITION_ANCHOR_MAX_AGE_SEC) -- it
        # must not be contingent on the aegis/metrics path below, which reads
        # from a separate DB and exchange call that can fail independently.
        # update_exchange() already swallows and logs its own failures, so a
        # bad read here degrades to the anchor aging out rather than raising.
        try:
            await self.position.update_exchange()
        except Exception as e:
            logger.warning(f"Failed to refresh exchange positions due to {e}")

        async def _check_and_sync_trade(
            sem,
            oms_id: str,
            package_id: str,
            record: tuple[Symbol, str, Decimal, datetime],
        ):
            """
            Worker function.
            Returns: (package_id, record) IF it is fully filled and synced or 0-filled.
            Returns: None IF it is pending or failed.
            """
            async with sem:
                symbol, client_order_id, start_price, start_time = (
                    record[0],
                    record[1],
                    record[2],
                    record[3],
                )
                asset = parts[0] if symbol and (parts := symbol.split()) else ""

                try:
                    async with self.rate_limiter.guard(
                        endpoint=Endpoints.GET_ORDER_DETAILS
                    ):
                        result = (
                            await self.config.exchange.get_order_details_from_history(
                                symbol=symbol,
                                client_order_id=client_order_id,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch {client_order_id}: {e}")
                    return None  # Failed fetch, keep in pool

                if not result:
                    age = datetime.now(timezone.utc) - start_time.astimezone(
                        timezone.utc
                    )
                    if age > AEGIS_NOT_FOUND_GIVE_UP:
                        logger.error(
                            f"[ON_AEGIS_UPDATE] {client_order_id} still not in exchange history after {age}, dropping record"
                        )
                        return (package_id, record)
                    logger.warning(
                        f"[ON_AEGIS_UPDATE] {client_order_id} NOT FOUND IN EXCHANGE HISTORY"
                    )
                    return None  # Not found, keep in pool

                if result.status in (
                    OrderStatus.CREATED,
                    OrderStatus.PARTIALLY_FILLED,
                ):
                    # Still live; syncing now would freeze a partial fill as
                    # the final trade and later fills would never reach aegis
                    return None

                if result.filled_size == Decimal("0"):
                    # Terminal with nothing filled, no trade will ever come
                    return (package_id, record)

                match result.side:
                    case OrderSide.BUY:
                        order_side = 1
                    case OrderSide.SELL:
                        order_side = -1
                    case OrderSide.NONE:
                        logger.error("[ON_AEGIS_UPDATE] Orderside shouldn't be NONE")
                        order_side = 0

                try:
                    await self.metric_builder.create_trade(
                        oms_id=oms_id,
                        package_id=package_id,
                        client_order_id=client_order_id,
                        asset=asset,
                        symbol=str(symbol),
                        exchange=self.config.exchange.exchange(),
                        start_quantity=str(order_side * result.size),
                        executed_quantity=str(order_side * result.filled_size),
                        executed_price=str(result.price),
                        executed_time=int(
                            result.updated_time.astimezone(tz=timezone.utc).timestamp()
                            * 1_000_000
                        )
                        * 1_000,
                        start_time=int(
                            start_time.astimezone(tz=timezone.utc).timestamp()
                            * 1_000_000
                        )
                        * 1_000,
                        start_price=str(start_price),
                    )

                    return (package_id, record)

                except Exception as e:
                    logger.error(f"Failed to save {client_order_id} to Aegis: {e}")
                    return None  # DB Write failed, keep in pool to retry next time

        balance_endpoint = Endpoints.GET_WALLET_BALANCE
        try:
            oms_id = self.config.config.oms_id
            logger.debug("Inserting trades to aegis")

            sem = asyncio.Semaphore(2)
            tasks = []

            # Orders still in the local pool are live: nothing terminal to
            # sync yet, polling their history only burns the shared budget
            async with self.opm.order_pools.get_order_pool() as order_pool:
                live_ids = set(order_pool.keys())

            for package_id, records in list(self.opm.order_pools.order_records.items()):
                for record in records:
                    if record[1] in live_ids:
                        continue
                    tasks.append(_check_and_sync_trade(sem, oms_id, package_id, record))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, BaseException):
                        logger.error(f"Trade sync error: {res}")
                        continue
                    if res:
                        filled_pkg_id, filled_record = res
                        try:
                            self.opm.order_pools.order_records[filled_pkg_id].remove(
                                filled_record
                            )
                            if not self.opm.order_pools.order_records[filled_pkg_id]:
                                del self.opm.order_pools.order_records[filled_pkg_id]
                        except (ValueError, KeyError):
                            pass

            # EQUITY UPDATE
            async with self.rate_limiter.guard(endpoint=balance_endpoint):
                logger.debug("Inserting equity to aegis")
                balance = await self.config.exchange.get_wallet_balance()
                await self.metric_builder.create_equity(
                    oms_id=oms_id, equity=str(balance.margin_balance)
                )

            # POSITION UPDATE
            logger.debug("Inserting position to aegis")
            # Anchor already refreshed at the top of this handler, independently
            # of the try below -- no second read needed here.
            for position in self.position.exchange.values():
                asset = parts[0] if (parts := position.symbol.split()) else ""
                await self.metric_builder.create_position(
                    oms_id=oms_id,
                    asset=asset,
                    symbol=str(position.symbol),
                    exchange=self.config.exchange.exchange(),
                    quantity=str(position.quantity),
                    price=str(position.entry_price),
                    updated_time=int(
                        position.updated_time.astimezone(tz=timezone.utc).timestamp()
                        * 1_000_000
                    )
                    * 1_000,
                )
        except Exception as e:
            logger.warning(f"Failed to upsert to aegis due to {e}")

    async def run(self):
        try:
            await self.init()
            self._setup_signals()
            await self.metric_stream.subscribe(
                portfolio_signal_subject(
                    self.config.config.portfolio_id, self.signal_namespace
                ),
                callback=self.on_portfolio_signal,
            )

            # Control plane: operator commands (e.g. rebalance) for this OMS,
            # over the same broker on a dedicated command subject.
            await self.metric_stream.subscribe(
                oms_command_subject(self.config.config.oms_id, self.signal_namespace),
                callback=self.on_command,
            )

            await self.scheduler.schedule(
                id="on_refresh_config",
                handler=self.on_refresh_config,
                trigger=Trigger.Cron("*/2 * * * * *"),  # every 2 seconds
            )

            await self.scheduler.schedule(
                id="on_aegis_update",
                handler=self.on_aegis_update,
                trigger=Trigger.Cron("*/1 * * * *"),  # every 1 minute
            )

            await self.scheduler.schedule(
                id="on_process_latest_signal",
                handler=self.on_process_latest_signal,
                trigger=Trigger.Cron("*/5 * * * * *"),  # every 5 seconds
            )

            await self.scheduler.schedule(
                id="on_order_placement",
                handler=self.opm.on_order_placement,
                trigger=Trigger.Cron(
                    generate_cron(self.config.config.order_placement_interval)
                ),  # every 15 seconds
            )

            await self.scheduler.schedule(
                id="on_retry_backlog",
                handler=self.opm.on_retry_backlog,
                trigger=Trigger.Cron("*/2 * * * * *"),  # 2 seconds
            )

            await self.scheduler.schedule(
                id="on_order_expiry_check",
                handler=self.opm.on_order_expiry_check,
                trigger=Trigger.Cron(generate_cron(self.config.config.expiry_check)),
            )

            await self.scheduler.schedule(
                id="on_resync_time",
                handler=self.rate_limiter.on_resync_time,
                # Hourly, not daily. Every cooldown deadline comparison and
                # Binance's weight-window reset are computed against
                # get_synced_time_ms(), so drift accumulated over a day either
                # releases a cooldown early or resets the weight counter on the
                # wrong side of the exchange's minute boundary. The call itself
                # costs weight 1.
                trigger=Trigger.Cron("0 * * * *"),  # every hour
            )

            await self.scheduler.schedule(
                id="on_risk_check",
                handler=self.risk.run_risk_checks,
                trigger=Trigger.Cron("*/10 * * * * *"),  # every 10 seconds
            )

            self.exchange_event = self.config.config.credentials.to_exchange_event()
            self.exchange_event.on_event = self.opm.on_exchange_event
            self.exchange_events_task = asyncio.create_task(self.exchange_event.start())
            self._supervise_task(self.exchange_events_task, "EXCHANGE_EVENTS")
            self._start_price_feed()
            await self.scheduler.start()
        except Exception:
            logger.exception("OMS run() terminated unexpectedly")
            raise
        finally:
            await self._handle_shutdown()

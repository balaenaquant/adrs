import asyncio
import logging

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Literal, Dict, List


from adrs.oms.config import ConfigManager
from adrs.oms.ops.order_executer import (
    CancelFatal,
    CancelMultiResult,
    OrderExecutor,
    MAX_CONCURRENT_ORDER_OPS,
)
from adrs.oms.ops.order_utils import OrderUtils
from adrs.oms.position import PositionManager
from adrs.oms.ops.order_pool import (
    BacklogDetails,
    CancelBacklogs,
    ExpiredBacklogs,
    OrderBacklogs,
    OrderDetails,
    OrderPoolHandler,
)

from cybotrade import Symbol
from cybotrade.io import Event, EventType
from cybotrade.models import OrderSide, OrderUpdate, OrderStatus

from adrs.oms.rate_limit.rate_limiter import RateLimiter
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.error_policy import ExchangeErrorPolicy

logger = logging.getLogger(__name__)

# Caps the per-tick burst below the ~8/s order/cancel pool limit
MAX_CONCURRENT_BACKLOG_RETRIES = 3

# Consecutive on_order_placement ticks a symbol can fail to cancel its
# opposite-side orders before the circuit breaker logs loudly. Every such
# tick already skips that symbol's replacement (see on_order_placement); this
# only escalates a persistent problem (e.g. sustained rate limiting) from a
# per-tick warning to a standing alert, instead of it silently piling into
# the retry backlog tick after tick.
CANCEL_FAILURE_CIRCUIT_BREAKER_THRESHOLD = 3


class OrderPlacementManager:
    def __init__(
        self,
        position: PositionManager,
        config: ConfigManager,
        rate_limiter: RateLimiter,
        error_policy: ExchangeErrorPolicy,
        executor_cls: type[OrderExecutor] = OrderExecutor,
    ) -> None:
        self.position = position
        self.config_manager = config
        self.order_pools = OrderPoolHandler(
            exchange=config.exchange,
            config=self.config_manager,
            rate_limiter=rate_limiter,
        )
        self.executor = executor_cls(
            config_manager=self.config_manager,
            order_pools=self.order_pools,
            rate_limiter=rate_limiter,
            error_policy=error_policy,
        )
        self.rate_limiter = rate_limiter
        # Consecutive on_order_placement ticks in a row where this symbol's
        # cancel_multi_limit_order had at least one failure. Reset on any
        # tick where every chosen cancel succeeds.
        self._consecutive_cancel_failures: Dict[Symbol, int] = {}

    def _note_cancel_outcome(self, symbol: Symbol, cancel_result: CancelMultiResult):
        if cancel_result.failed_count == 0:
            self._consecutive_cancel_failures.pop(symbol, None)
            return
        streak = self._consecutive_cancel_failures.get(symbol, 0) + 1
        self._consecutive_cancel_failures[symbol] = streak
        if streak >= CANCEL_FAILURE_CIRCUIT_BREAKER_THRESHOLD:
            logger.error(
                f"[CIRCUIT_BREAKER] {symbol} has failed to cancel opposite-side "
                f"orders for {streak} consecutive ticks ({cancel_result.failed_count} "
                f"failure(s) this tick) - placement has been skipped every one of "
                f"those ticks, but this many in a row means the problem is not "
                f"self-resolving. Check the rate limiter / exchange connectivity."
            )

    async def on_exchange_event(self, event: Event):
        """
        Handles various events sent by exchange
        """
        match event.event_type:
            case EventType.Authenticated | EventType.Subscribed:
                logger.info(f"[ON_EVENT] '{event.event_type}': {event.data}")
            case EventType.OrderUpdate:
                logger.debug(f"[ON_EVENT] '{event.event_type}': {event.data}")
                await self.on_order_update(event.data)
            case EventType.Error:
                logger.error(f"[ON_EVENT] '{event.event_type}': {event.data}")
            case _:
                logger.warning(f"[ON_EVENT] '{event.event_type}': {event.data}")

    async def validate_oms_state(self, update: OrderUpdate) -> bool:
        """
        Return True when the update can be applied to local state.

        An unknown id only warrants a full open-orders resync when the order
        is still live on the exchange; in-flight placements and terminal
        events are handled with at most a position refresh.
        """
        async with self.order_pools.get_order_pool() as order_pool:
            if update.client_order_id in order_pool:
                return True

        is_terminal = update.status in (
            OrderStatus.FILLED,
            OrderStatus.PARTIALLY_FILLED_CANCELLED,
        )

        if self.order_pools.is_pending_insert(update.client_order_id):
            if is_terminal:
                await self.position.update_exchange()
            logger.info(
                f"[VALIDATE] {update.client_order_id} in flight, WS beat the REST response"
            )
            return False

        if is_terminal:
            await self.position.update_exchange()
            logger.info(
                f"[VALIDATE] Terminal update for unknown order {update.client_order_id}, refreshed positions"
            )
            return False

        logger.warning(
            f"[VALIDATE] Live order {update.client_order_id} missing from local pool, resyncing"
        )
        snapshot = await self.order_pools.fetch_open_orders_snapshot()
        await self.position.update_exchange()
        self.position.update_pending(snapshot)
        await self.order_pools.resync_order_pool(snapshot)
        return False

    def update_positions(self, update: OrderUpdate):
        # filled_size is cumulative. Apply only the new slice and store the
        # cumulative total (not the slice) as the next baseline, otherwise the
        # baseline drifts and fills over-count from the third update onward.
        last_filled = self.order_pools.order_value_update.get(
            update.client_order_id, Decimal("0")
        )
        increment = update.filled_size - last_filled
        self.order_pools.order_value_update[update.client_order_id] = update.filled_size
        asset_filled = increment if update.side == OrderSide.BUY else -increment
        self.position.pending[update.symbol].quantity -= asset_filled
        self.position.exchange[update.symbol].quantity += asset_filled

    async def on_order_update(self, update: OrderUpdate):
        """
        Handles Event OrderUpdate where position update and calculation would be done
        """
        symbol = str(update.symbol)

        if update.status == OrderStatus.CREATED:
            logger.info(
                f"[ORDER_UPDATE|CREATED] Limit {symbol} order: {update.client_order_id} is {update.status}"
            )
            # Update local store in case rest api hits rate limits
            self.position.pending[update.symbol].quantity += (
                update.size if update.side == OrderSide.BUY else -update.size
            )
            return

        if update.status == OrderStatus.CANCELLED:
            # Pop covers externally cancelled orders (manual UI, exchange purge)
            # which would otherwise stay in the pool and fail cancel forever;
            # already removed by our own cancel path is fine, pop is idempotent
            async with self.order_pools.get_order_pool() as order_pool:
                order_pool.pop(update.client_order_id, None)
            # Remove unfulffilled amount from pending for to remake order in next delta calculation
            self.position.pending[update.symbol].quantity -= (
                update.remain_size
                if update.side == OrderSide.BUY
                else -update.remain_size
            )
            logger.info(
                f"[ORDER_UPDATE|CANCELLED] Removed {update.client_order_id} from order_pool due to order {update.status}"
            )
            return

        # Majority of cases if when price moved during order placement invalidating post-only orders
        # Hence retry with bbo
        if update.status == OrderStatus.REJECTED:
            logger.info(
                f"[ORDER_UPDATE|REJECTED] {update.client_order_id} has been rejected due to order {update.status}"
            )
            # TEST WILL NEED TO WATCH OUT THE FLOW WHEN THIS ACTUALLY TRIGGERS
            async with self.order_pools.get_order_pool() as order_pool:
                # pop: WS REJECTED can beat the REST response that inserts the
                # pool entry, del would raise and kill the event handler
                rejected_order = order_pool.pop(update.client_order_id, None)
            if rejected_order is not None:
                # Mirror CANCELLED accounting: CREATED added this qty to pending
                # and the reprice below adds its own on its CREATED, without
                # this the rejected order's qty leaks in pending until resync
                self.position.pending[update.symbol].quantity -= (
                    update.remain_size
                    if update.side == OrderSide.BUY
                    else -update.remain_size
                )
            result = await self.executor.reprice_at_bbo(
                symbol=update.symbol,
                qty=update.remain_size,
                side=update.side,
                replace_best_bid_ask_time=datetime.now(timezone.utc)
                + timedelta(
                    seconds=self.config_manager.config.replace_best_bid_ask_time
                ),
                package_id=self.executor.package_id,
            )
            if result:
                async with self.order_pools.get_order_backlog() as order_backlog:
                    self.order_pools.dedup_append(order_backlog, result)
            return

        # Create and cancel don't need as order might not be in order pool yet
        if not await self.validate_oms_state(update=update):
            # validate logged the specific cause and refreshed what it needed
            return

        is_terminal = (
            update.status == OrderStatus.FILLED
            or update.status == OrderStatus.PARTIALLY_FILLED_CANCELLED
        )

        # Update Order Details. validate_oms_state released the pool lock, so the
        # entry may have been popped concurrently (e.g. a FILLED beat us) — bail
        # if it's gone. All ops below are synchronous, so the lock stays a leaf.
        async with self.order_pools.get_order_pool() as order_pool:
            entry = order_pool.get(update.client_order_id)
            if entry is None:
                return
            entry.remain_size = update.remain_size
            if is_terminal:
                order_pool.pop(update.client_order_id, None)
            pool_size = len(order_pool)

        if update.status == OrderStatus.PARTIALLY_FILLED:
            logger.info(
                f"[ORDER_UPDATE|PARTIALLY_FILLED] {update.client_order_id} has filled {update.filled_size}, remain {update.remain_size}"
            )
            self.update_positions(update=update)
            return

        if is_terminal:
            self.update_positions(update=update)
            if update.status == OrderStatus.PARTIALLY_FILLED_CANCELLED:
                # Unfilled remainder was added to pending at CREATED but the
                # cancel leaves it working forever; mirror the CANCELLED branch
                self.position.pending[update.symbol].quantity -= (
                    update.remain_size
                    if update.side == OrderSide.BUY
                    else -update.remain_size
                )
            self.order_pools.order_value_update.pop(update.client_order_id, None)
            logger.info(
                f"[ORDER_UPDATE|FILLED] Removed {update.client_order_id} from order_pool due to order {update.status}, current order_pool size: {pool_size}"
            )
            return

    def on_position_check(self):
        desired = self.position.desired
        pending = self.position.pending
        exchange = self.position.exchange
        logger.info("[OMS STATUS CHECK]")
        for position in (
            (desired, "DESIRED"),
            (pending, "PENDING"),
            (exchange, "EXCHANGE"),
        ):
            logger.info(
                f"[{position[1]} POSITION] {' '.join([(parts[0] if (parts := symbol.split()) else 'Unknown') + ': ' + str(position.quantity) for symbol, position in position[0].items()])}"
            )
        logger.info(f"[RATE LIMITER STATS] {self.rate_limiter}")

    async def on_order_placement(self):
        """
        Periodic rebalance execution logic.

        Rebalance logic:
        - delta > 0 => buy, delta < 0 => sell
        - pending tracks already-submitted orders (same sign = same direction)

        Rules:
        - If delta and pending have opposite signs → cancel pending
        - Else (same direction) → place extra orders

        TLDR: Same direction → add orders, Opposite → cancel
        """
        self.executor.update_package_id()
        self.on_position_check()
        snapshot = await self.order_pools.fetch_open_orders_snapshot()
        deltas = await self.position.delta_calculation(snapshot)
        queued: dict[Symbol, Decimal] = {}
        async with self.order_pools.get_order_backlog() as order_backlog:
            for backlog in order_backlog:
                if not isinstance(backlog, OrderBacklogs):
                    continue
                symbol = Symbol(backlog.symbol)
                signed = backlog.qty if backlog.side == OrderSide.BUY else -backlog.qty
                queued[symbol] = queued.get(symbol, Decimal("0")) + signed

        for symbol, queued_qty in queued.items():
            if symbol not in deltas:
                continue
            adjusted = deltas[symbol] - queued_qty
            if deltas[symbol] >= 0:
                deltas[symbol] = max(Decimal("0"), adjusted)
            else:
                deltas[symbol] = min(Decimal("0"), adjusted)

        if all(delta == Decimal("0") for delta in deltas.values()):
            logger.info("Delta between desired and pending/exchange is currently zero")
            return

        logger.info(f"[ON_ORDER_PLACEMENT] Deltas {deltas}")

        try:
            await self.config_manager.update_symbol_info(self.rate_limiter)
        except Exception as e:
            logger.warning(f"update symbol info failed due to {e}")

        for symbol, delta in deltas.items():
            if delta == 0:
                continue

            # To determine whether the delta is big enough to proceed
            residual_threshold = self.config_manager.symbol_infos[symbol].min_limit_qty
            if abs(delta) < residual_threshold:
                logger.info(
                    f"{symbol}'s delta {delta} is lower than provided threshold {residual_threshold}, it is ignored"
                )
                continue

            # To double check just in case exchange didn't properly update min limit
            notional_threshold = self.config_manager.symbol_infos[symbol].min_notional
            current_price = await self.executor.get_current_price(symbol=symbol)
            if current_price is None:
                logger.warning(
                    f"{symbol}'s current price couldn't be fetched, skipping this tick"
                )
                continue
            notional_value = abs(delta) * current_price
            if notional_value < notional_threshold:
                logger.info(
                    f"{symbol}'s delta {delta} is lower than provided notional threshold {notional_threshold}, it is ignored"
                )
                continue

            pending = self.position.pending[symbol].quantity
            direction = pending * delta if pending != 0 else delta

            # if there is orders on the opposite side cancel first
            if direction < 0 and pending != 0:
                open_orders = snapshot.orders.get(symbol)
                if open_orders is None:
                    # Fetch failed for this symbol; delta retries next tick
                    logger.warning(
                        f"{symbol} missing from open-orders snapshot, skipping reversal this tick"
                    )
                    continue
                cancel_result = await self.executor.cancel_multi_limit_order(
                    symbol=symbol, target=delta, open_orders=open_orders
                )
                self._note_cancel_outcome(symbol, cancel_result)
                if cancel_result.failed_count:
                    # Some opposite-side orders are still live/unconfirmed;
                    # placing a same-tick replacement now would size it as if
                    # they were gone. Skip - the failed cancels are already
                    # queued for retry, and the next tick's fresh delta
                    # (informed by whatever actually happens to them) decides.
                    logger.warning(
                        f"[ON_ORDER_PLACEMENT] {cancel_result.failed_count} cancel(s) "
                        f"failed for {symbol}; skipping this tick's replacement order"
                    )
                    continue
                delta = cancel_result.remainder

            # place order based on delta or remander
            await self.executor.place_multiple_limit_order(
                symbol=symbol,
                quantity=delta,
            )
        self.on_position_check()

    async def on_retry_backlog(self):
        """
        To reattempt all failed orders/cancels
        """
        if self.rate_limiter.retry_after >= self.rate_limiter.get_synced_time_ms():
            logger.info("[ON_RETRY_BACKLOG] Skipping tick, rate limiter cooling down")
            return

        now = datetime.now(timezone.utc)

        # Snapshot due items under the lock; run the I/O with it released
        async with self.order_pools.get_order_backlog() as order_backlog:
            if len(order_backlog) == 0:
                return
            logger.info(f"[ON_RETRY_BACKLOG] Current backlog size {len(order_backlog)}")
            due: list[BacklogDetails] = []
            for backlog in list(order_backlog):
                if backlog.next_retry_at is not None and backlog.next_retry_at > now:
                    continue
                if (
                    backlog.total_retries
                    >= self.config_manager.config.max_retries_allowed
                ):
                    order_backlog.remove(backlog)
                    logger.warning(
                        f"[ON_RETRY_BACKLOG] Removed backlog {backlog} exceeded max retries, Current backlog size {len(self.order_pools.order_backlog)}"
                    )
                    continue
                due.append(backlog)

        if not due:
            return

        sem = asyncio.Semaphore(MAX_CONCURRENT_BACKLOG_RETRIES)

        async def _bounded(backlog: BacklogDetails):
            async with sem:
                return await self._retry_one(backlog, now)

        outcomes = await asyncio.gather(
            *[_bounded(backlog) for backlog in due],
            return_exceptions=True,
        )

        # Re-acquire only to apply list mutations
        async with self.order_pools.get_order_backlog() as order_backlog:
            for backlog, outcome in zip(due, outcomes):
                if isinstance(outcome, BaseException):
                    logger.error(
                        f"[ON_RETRY_BACKLOG] Retry crashed for {backlog} due to {outcome}"
                    )
                    backlog.total_retries += 1
                    backlog.next_retry_at = now + timedelta(
                        seconds=min(2 * backlog.total_retries, 30)
                    )
                    continue

                should_remove, replacement = outcome
                if should_remove and backlog in order_backlog:
                    order_backlog.remove(backlog)
                if replacement is not None:
                    self.order_pools.dedup_append(order_backlog, replacement)

    async def _retry_one(self, backlog: BacklogDetails, now: datetime):
        """Retry one item. Returns (should_remove, replacement_order)."""
        if isinstance(backlog, OrderBacklogs):
            symbol = Symbol(backlog.symbol)
            result = await self.executor.place_single_limit_order(
                symbol=symbol,
                offset=backlog.offset,
                qty=backlog.qty,
                side=backlog.side,
                symbol_info=self.config_manager.symbol_infos[symbol],
                package_id=backlog.package_id,
                initial_price=backlog.initial_price,
                initial_time=backlog.initial_time,
                client_order_id=backlog.client_order_id,
            )
        elif isinstance(backlog, CancelBacklogs | ExpiredBacklogs):
            async with self.order_pools.get_order_pool() as order_pool:
                still_ours = backlog.client_order_id in order_pool
            if not still_ours:
                logger.info(
                    f"[ON_RETRY_BACKLOG] {backlog.client_order_id} already left the pool, dropping backlog without replacement"
                )
                return (True, None)
            result = await self.executor.cancel_single_order(
                symbol=Symbol(backlog.symbol),
                client_order_id=backlog.client_order_id,
            )
        else:
            raise Exception("Unknown Backlog Type")

        if isinstance(result, CancelFatal):
            return (True, None)

        # Failed: strike + backoff, keep
        if result is not None:
            backlog.total_retries += 1
            delay = min(2 * backlog.total_retries, 30)
            backlog.next_retry_at = now + timedelta(seconds=delay)
            return (False, None)

        # Success: remove; expired cancel also re-places
        if isinstance(backlog, ExpiredBacklogs):
            return (True, await self._reprice_expired(backlog))
        return (True, None)

    async def _reprice_expired(self, backlog: ExpiredBacklogs):
        """Reconfirm remaining qty after cancel, then re-place. Returns replacement or None."""
        qty = backlog.qty
        details = None
        try:
            async with self.rate_limiter.guard(endpoint=Endpoints.GET_ORDER_DETAILS):
                details = (
                    await self.config_manager.exchange.get_order_details_from_history(
                        symbol=Symbol(backlog.symbol),
                        client_order_id=backlog.client_order_id,
                    )
                )
        except Exception as e:
            logger.warning(
                f"[ON_RETRY_BACKLOG] Could not reconfirm remaining qty for {backlog.client_order_id} due to {e}, leaving requantification to the delta path"
            )
            return None
        if details is not None:
            if details.status in (
                OrderStatus.CREATED,
                OrderStatus.PARTIALLY_FILLED,
            ):
                logger.warning(
                    f"[ON_RETRY_BACKLOG] {backlog.client_order_id} still shows open in history, skipping replacement"
                )
                return None
            qty = details.remain_size
        if qty <= Decimal("0"):
            logger.info(
                f"[ON_RETRY_BACKLOG] {backlog.client_order_id} fully filled during backlog wait, nothing to replace"
            )
            return None
        # Now try placing order will turn expire into order backlog if fails
        if backlog.is_bbo:
            return await self.executor.reprice_at_bbo(
                symbol=Symbol(backlog.symbol),
                qty=qty,
                side=backlog.side,
                replace_best_bid_ask_time=backlog.replace_best_bid_ask_time,
                package_id=backlog.package_id,
                initial_price=backlog.initial_price,
                initial_time=backlog.initial_time,
            )
        return await self.executor.reprice_at_mid(
            symbol=Symbol(backlog.symbol),
            qty=qty,
            side=backlog.side,
            replace_best_bid_ask_time=backlog.replace_best_bid_ask_time,
            package_id=backlog.package_id,
            initial_price=backlog.initial_price,
            initial_time=backlog.initial_time,
        )

    async def on_order_expiry_check(self):
        """
        Remove orders that have exceeded the allowed lifetime.

        Replaces them at Update Price (MID) or BBO depending on the trigger.
        """

        @dataclass
        class PendingExpiry:
            order: OrderDetails
            strategy: Literal["BBO", "MID"]

        # Decide which orders expired (no delta_lock — read-only price discovery)
        async with self.order_pools.get_order_pool() as order_pool:
            orders_to_check = list(order_pool.values())

        if not orders_to_check:
            return

        async with self.order_pools.get_order_backlog() as order_backlog:
            queued_ids = {b.client_order_id for b in order_backlog}
        if queued_ids:
            orders_to_check = [
                o for o in orders_to_check if o.client_order_id not in queued_ids
            ]
            if not orders_to_check:
                return

        pending_actions: dict[str, PendingExpiry] = {}
        current_time = datetime.now(timezone.utc)

        get_depth_endpoint = Endpoints.GET_ORDERBOOK_SNAPSHOT
        current_prices: Dict[Symbol, List[Decimal]] = {}
        for order in orders_to_check:
            symbol = Symbol(order.symbol)

            reason = None
            if order.replace_best_bid_ask_time <= current_time:
                reason = "BBO"
            elif order.max_replace_limit_order_time <= current_time:
                reason = "MID"

            if reason:
                if symbol not in current_prices:
                    try:
                        order_book = await OrderUtils.get_order_book(
                            exchange=self.executor.exchange,
                            pair=symbol,
                            need_log=False,
                            rate_limiter=self.rate_limiter,
                            endpoint=get_depth_endpoint,
                        )
                        current_prices[symbol] = order_book
                    except Exception as e:
                        logger.warning(
                            f"Current price for {order.symbol} couldn't be fetch due to {e}"
                        )
                order_book = current_prices.get(symbol)
                current_price = (
                    (order_book[0] if order.side == OrderSide.BUY else order_book[1])
                    if order_book
                    else None
                )
                if not current_price or current_price != order.price:
                    # Entry stays in the pool until cancel succeeds:
                    # cancel_single_order pops on success, and treats a failed
                    # cancel for an already-vanished entry as done — popping here
                    # would make every transient cancel failure look like success
                    # and reprice a still-live order
                    pending_actions[order.client_order_id] = PendingExpiry(
                        order, reason
                    )

        if not pending_actions:
            return

        logger.info(
            f"[ON_ORDER_EXPIRY_CHECK] Processing {len(pending_actions)} expired orders"
        )

        # Cancel + replace under delta_lock (the in-flight window); collect
        # backlog locally so the backlog lock isn't held across the gathers
        backlog_to_add: list[BacklogDetails] = []
        sem = asyncio.Semaphore(MAX_CONCURRENT_ORDER_OPS)

        async def _bounded(coro):
            async with sem:
                return await coro

        async with self.position.delta_lock:
            cancel_tasks = [
                _bounded(
                    self.executor.cancel_single_order(
                        symbol=Symbol(action.order.symbol),
                        client_order_id=action.order.client_order_id,
                    )
                )
                for action in pending_actions.values()
            ]

            cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

            replacement_tasks = []

            for action, result in zip(pending_actions.values(), cancel_results):
                if isinstance(result, BaseException):
                    logger.error(
                        f"[ON_ORDER_EXPIRY_CHECK] Critical Cancel Error: {result}"
                    )
                    continue

                if isinstance(result, CancelFatal):
                    logger.error(
                        f"[ON_ORDER_EXPIRY_CHECK] FATAL cancel for "
                        f"{action.order.client_order_id}, skipping replacement"
                    )
                    continue

                if result is not None:
                    expire_backlog = ExpiredBacklogs(
                        side=action.order.side,
                        qty=action.order.remain_size,
                        is_bbo=(action.strategy == "BBO"),
                        max_replace_limit_order_time=action.order.max_replace_limit_order_time,
                        replace_best_bid_ask_time=action.order.replace_best_bid_ask_time,
                        package_id=action.order.package_id,
                        initial_price=action.order.initial_price,
                        initial_time=action.order.initial_time,
                        **result.model_dump(),
                    )
                    backlog_to_add.append(expire_backlog)
                    continue

                if action.strategy == "BBO":
                    logger.info(
                        f"[ON_ORDER_EXPIRY_CHECK] Repricing {action.order.symbol} order at BBO with {action.order.remain_size}"
                    )
                    new_task = self.executor.reprice_at_bbo(
                        symbol=Symbol(action.order.symbol),
                        qty=action.order.remain_size,
                        side=action.order.side,
                        replace_best_bid_ask_time=action.order.replace_best_bid_ask_time,
                        package_id=action.order.package_id,
                        initial_price=action.order.initial_price,
                        initial_time=action.order.initial_time,
                    )
                else:
                    logger.info(
                        f"[ON_ORDER_EXPIRY_CHECK] Repricing {action.order.symbol} order at MID with {action.order.remain_size}"
                    )
                    new_task = self.executor.reprice_at_mid(
                        symbol=Symbol(action.order.symbol),
                        qty=action.order.remain_size,
                        side=action.order.side,
                        replace_best_bid_ask_time=action.order.replace_best_bid_ask_time,
                        package_id=action.order.package_id,
                        initial_price=action.order.initial_price,
                        initial_time=action.order.initial_time,
                    )
                replacement_tasks.append(_bounded(new_task))

            if replacement_tasks:
                logger.info(
                    f"[ON_ORDER_EXPIRY_CHECK] Sending {len(replacement_tasks)} replacement orders. "
                    f"({len(pending_actions) - len(replacement_tasks)} failed/skipped)"
                )

                replace_results = await asyncio.gather(
                    *replacement_tasks, return_exceptions=True
                )

                for res in replace_results:
                    if res is None:
                        continue
                    if isinstance(res, BaseException):
                        logger.error(
                            f"[ON_ORDER_EXPIRY_CHECK] Critical Replace Error: {res}"
                        )
                    else:
                        backlog_to_add.append(res)

        if backlog_to_add:
            async with self.order_pools.get_order_backlog() as order_backlog:
                for item in backlog_to_add:
                    self.order_pools.dedup_append(order_backlog, item)

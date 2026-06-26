import asyncio
import logging

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Literal, Dict, List


from adrs.oms.config import ConfigManager
from adrs.oms.ops.order_executer import OrderExecutor
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

logger = logging.getLogger(__name__)


class OrderPlacementManager:
    def __init__(
        self,
        position: PositionManager,
        config: ConfigManager,
        rate_limiter: RateLimiter,
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
        )
        self.rate_limiter = rate_limiter

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
        # validation check
        if (
            len(self.order_pools.order_pool) == 0
            or update.client_order_id not in self.order_pools.order_pool.keys()
        ):
            # Reset everything that relies on web hook
            await self.position.update_exchange()
            await self.position.update_pending()
            await self.order_pools.resync_order_pool()
            return False
        return True

    def update_positions(self, update: OrderUpdate):
        last_update = self.order_pools.order_value_update.get(
            update.client_order_id, Decimal("0")
        )
        self.order_pools.order_value_update[update.client_order_id] = (
            update.filled_size - last_update
        )
        asset_filled = (
            self.order_pools.order_value_update[update.client_order_id]
            if update.side == OrderSide.BUY
            else -self.order_pools.order_value_update[update.client_order_id]
        )
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
            self.order_pools.order_pool.pop(update.client_order_id, None)
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
                    order_backlog.append(result)
            return

        # Create and cancel don't need as order might not be in order pool yet
        if not await self.validate_oms_state(update=update):
            logger.warning(
                "Local Order pool has been desynced with the exchange reseting local values"
            )
            # Already update values just return
            return

        # Update Order Details
        self.order_pools.order_pool[
            update.client_order_id
        ].remain_size = update.remain_size

        if update.status == OrderStatus.PARTIALLY_FILLED:
            logger.info(
                f"[ORDER_UPDATE|PARTIALLY_FILLED] {update.client_order_id} has filled {update.filled_size}, remain {update.remain_size}"
            )
            self.update_positions(update=update)
            return

        if (
            update.status == OrderStatus.FILLED
            or update.status == OrderStatus.PARTIALLY_FILLED_CANCELLED
        ):
            fulfilled_order = self.order_pools.order_pool.pop(update.client_order_id)
            self.update_positions(update=update)
            logger.info(
                f"[ORDER_UPDATE|FILLED] Removed {fulfilled_order.client_order_id} from order_pool due to order {update.status}, current order_pool size: {len(self.order_pools.order_pool)}"
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
        deltas = await self.position.delta_calculation()
        if sum(delta for delta in deltas.values()) == Decimal("0"):
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
            current_price = await self.executor._get_current_price_safe(symbol=symbol)
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
                remainder = await self.executor.cancel_multi_limit_order(
                    symbol=symbol, target=delta
                )
                delta = remainder

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

        # Snapshot the due items under the lock: drop the dead ones, collect the
        # rest. The actual exchange I/O runs with the lock released so a slow
        # retry cannot block executor appends from the other schedulers.
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

        outcomes = await asyncio.gather(
            *[self._retry_one(backlog, now) for backlog in due],
            return_exceptions=True,
        )

        # Re-acquire the lock only to apply the list mutations.
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
                    order_backlog.append(replacement)

    async def _retry_one(self, backlog: BacklogDetails, now: datetime):
        """
        Attempt one backlog item. On failure, mutates the item's own retry and
        backoff fields and keeps it. Returns (should_remove, replacement_order)
        for the caller to apply under the backlog lock.
        """
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
            result = await self.executor.cancel_single_order(
                symbol=Symbol(backlog.symbol),
                client_order_id=backlog.client_order_id,
            )
        else:
            raise Exception("Unknown Backlog Type")

        # Exchange API call failed: strike + gentle backoff, keep in backlog
        if result is not None:
            backlog.total_retries += 1
            delay = min(2 * backlog.total_retries, 30)
            backlog.next_retry_at = now + timedelta(seconds=delay)
            return (False, None)

        # Success: remove. An expired cancel additionally re-places the order.
        if isinstance(backlog, ExpiredBacklogs):
            return (True, await self._reprice_expired(backlog))
        return (True, None)

    async def _reprice_expired(self, backlog: ExpiredBacklogs):
        """
        After an expired order's stale cancel succeeds, reconfirm the remaining
        qty and re-place it. Returns the replacement OrderBacklogs to re-queue
        if placement failed, else None.
        """
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
                f"[ON_RETRY_BACKLOG] Could not reconfirm remaining qty for {backlog.client_order_id} due to {e}, using frozen qty {qty}"
            )
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

        # Refer to Delta Calculation for reasoning
        async with self.position.delta_lock:
            orders_to_check = list(self.order_pools.order_pool.values())

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
                        (
                            order_book[0]
                            if order.side == OrderSide.BUY
                            else order_book[1]
                        )
                        if order_book
                        else None
                    )
                    if not current_price or current_price != order.price:
                        # Entry stays in the pool until cancel succeeds:
                        # cancel_single_order pops on success, and treats a
                        # failed cancel for an already-vanished entry as done —
                        # popping here would make every transient cancel failure
                        # look like success and reprice a still-live order
                        pending_actions[order.client_order_id] = PendingExpiry(
                            order, reason
                        )

            if not pending_actions:
                return

            logger.info(
                f"[ON_ORDER_EXPIRY_CHECK] Processing {len(pending_actions)} expired orders"
            )

            async with self.order_pools.get_order_backlog() as order_backlog:
                cancel_tasks = [
                    self.executor.cancel_single_order(
                        symbol=Symbol(action.order.symbol),
                        client_order_id=action.order.client_order_id,
                    )
                    for action in pending_actions.values()
                ]

                cancel_results = await asyncio.gather(
                    *cancel_tasks, return_exceptions=True
                )

                replacement_tasks = []

                for action, result in zip(pending_actions.values(), cancel_results):
                    if isinstance(result, BaseException):
                        logger.error(
                            f"[ON_ORDER_EXPIRY_CHECK] Critical Cancel Error: {result}"
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
                        order_backlog.append(expire_backlog)
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
                    replacement_tasks.append(new_task)

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
                            order_backlog.append(res)

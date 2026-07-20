import time
import logging
import random
import asyncio

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from pydantic.types import AwareDatetime
from cybotrade import Symbol
from cybotrade.io import ExchangeClient
from cybotrade.models import OrderSide, OrderUpdate, SymbolInfo

from adrs.oms.config import ConfigManager, Config
from adrs.oms.calculation import Calculate
from adrs.oms.ops.order_pool import (
    OrderBacklogs,
    OrderPoolHandler,
    CancelBacklogs,
    OrderDetails,
)
from adrs.oms.ops.order_utils import OrderUtils
from adrs.oms.rate_limit.rate_limiter import RateLimiter
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.error_policy import ErrorAction, ExchangeErrorPolicy

logger = logging.getLogger(__name__)

# Limit orders post away from the touch (buys below market, sells above), so the
# value at fill differs from value at the current market price. Derive the
# notional-based qty bounds from a price nudged this fraction toward a worse fill
# (buys: cheaper → need more qty to clear min_notional). Keeps an order from
# clearing min_notional at market yet getting rejected for order value once it
# rests at the limit price.
NOTIONAL_PRICE_SLIPPAGE = Decimal("0.005")

# Caps concurrent place/cancel requests per burst. Keeps the in-flight count
# at roughly the pool drain rate (~8/s buffered) x per-op latency, and bounds
# how many requests are already on the wire if the exchange rate-limits us
# while the local accounting thinks we are fine.
MAX_CONCURRENT_ORDER_OPS = 3


def make_package_id(portfolio_id: str) -> str:
    return f"{portfolio_id}_{time.time_ns() // 1_000_000}"


@dataclass
class PlacementContext:
    """
    Inputs available to the strategy hooks for one placement decision.

    New fields may be added over time; additions do not break existing
    subclass hook signatures.
    """

    symbol: Symbol
    side: OrderSide
    market_price: Decimal
    symbol_info: SymbolInfo
    # Exchange floor/ceiling per order, derived from lot size and notional limits
    min_qty: Decimal
    max_qty: Decimal


@dataclass
class CancelMultiResult:
    # Quantity still to place as a new order, computed from cancels that
    # actually succeeded - never from the pre-cancel selection.
    remainder: Decimal
    # How many of the chosen cancels did NOT succeed (still live/unknown on
    # the exchange). Callers must not place a same-tick replacement sized on
    # `remainder` while this is nonzero, or they risk placing on top of an
    # order that is still resting.
    failed_count: int


class OrderExecutor:
    def __init__(
        self,
        config_manager: ConfigManager,
        order_pools: OrderPoolHandler,
        rate_limiter: RateLimiter,
        error_policy: ExchangeErrorPolicy,
    ):
        self.exchange: ExchangeClient = config_manager.exchange
        self.config: Config = config_manager.config
        self.order_pools = order_pools
        self.symbol_infos = config_manager.symbol_infos
        self.rate_limiter = rate_limiter
        self.error_policy = error_policy
        self.package_id = make_package_id(self.config.portfolio_id)

    def update_package_id(self):
        self.package_id = make_package_id(self.config.portfolio_id)

    def _make_context(
        self, symbol: Symbol, side: OrderSide, market_price: Decimal
    ) -> PlacementContext:
        symbol_info = self.symbol_infos[symbol]
        # Notional bounds against the worst-case resting price, not the current
        # market: a buy that fills cheaper needs MORE qty to clear min_notional,
        # a sell that fills dearer needs LESS qty to stay under max_notional.
        # Using market_price directly lets an order pass here yet get rejected
        # for order value once it rests at the limit price.
        min_notional_price = market_price * (Decimal("1") - NOTIONAL_PRICE_SLIPPAGE)
        max_notional_price = market_price * (Decimal("1") + NOTIONAL_PRICE_SLIPPAGE)
        return PlacementContext(
            symbol=symbol,
            side=side,
            market_price=market_price,
            symbol_info=symbol_info,
            min_qty=max(
                symbol_info.min_limit_qty,
                symbol_info.min_notional / min_notional_price,
            ),
            max_qty=min(
                symbol_info.max_limit_qty,
                symbol_info.max_notional / max_notional_price,
            ),
        )

    async def split_order_quantity(
        self, qty: Decimal, ctx: PlacementContext
    ) -> list[Decimal]:
        """
        Strategy hook: break a target quantity into individual order sizes.

        The number of entries decides how many orders are placed; entries of
        zero or below are skipped. ctx.min_qty/max_qty are the exchange's
        per-order bounds; implementations may tighten them but not exceed.

        Naive default: a single order for the full quantity, chunked only
        when it exceeds the exchange's max order size.

        Every returned size is rounded (truncated) to the instrument's
        `quantity_precision` — exchanges reject a qty that isn't a multiple of
        the lot step (e.g. bybit "10001 Qty invalid"). Sizes that truncate below
        `min_qty` are dropped (too small to place).
        """
        precision = ctx.symbol_info.quantity_precision

        def round_qty(x: Decimal) -> Decimal:
            return Calculate.round_with_precision(x, precision)

        if qty <= ctx.max_qty:
            rounded = round_qty(qty)
            return [rounded] if rounded >= ctx.min_qty else []
        chunks, rest = divmod(qty, ctx.max_qty)
        sizes = [round_qty(ctx.max_qty)] * int(chunks)
        rest = round_qty(rest)
        if rest >= ctx.min_qty:
            sizes.append(rest)
        return sizes

    async def compute_limit_offsets(
        self, level: int, ctx: PlacementContext
    ) -> list[Decimal]:
        """
        Strategy hook: price offset from best bid/ask for each order,
        index-aligned with the sizes from split_order_quantity.

        Offset 0 posts at BBO; positive offsets post deeper into the book.

        Naive default: post everything at BBO.
        """
        return [Decimal("0")] * level

    async def cancel_single_order(
        self,
        symbol: Symbol,
        client_order_id: str,
    ) -> CancelBacklogs | None:
        """
        To cancel a specified order
        """
        endpoint = Endpoints.CANCEL_ORDER
        try:
            async with self.rate_limiter.guard(endpoint=endpoint):
                await self.exchange.cancel_order(
                    symbol=symbol,
                    client_order_id=client_order_id,
                )
            async with self.order_pools.get_order_pool() as order_pool:
                order_pool.pop(client_order_id, None)
            return None
        except Exception as e:
            action = self.error_policy.classify(e)
            async with self.order_pools.get_order_pool() as order_pool:
                already_left = client_order_id not in order_pool
                if action is ErrorAction.TERMINAL_SUCCESS:
                    order_pool.pop(client_order_id, None)

            if action is ErrorAction.TERMINAL_SUCCESS or already_left:
                # Order already gone on the exchange (110001) or our WS already
                # popped it; nothing left to cancel, retrying only burns rate limit
                logger.info(
                    f"Cancel for {client_order_id} treated as done "
                    f"(action={action.name}, already_left={already_left})"
                )
                return None
            if action is ErrorAction.FATAL:
                # Order may still be live, so keep it in the pool for the expiry/
                # delta logic, but stop retrying a cancel that will never succeed
                logger.error(
                    f"Cancel for {client_order_id} hit FATAL error, not retrying: {e}"
                )
                return None

            logger.warning(f"Failed to cancel order {client_order_id} because {e}")
            return CancelBacklogs(
                symbol=str(symbol),
                total_retries=1,
                client_order_id=client_order_id,
            )

    async def cancel_multi_limit_order(
        self, symbol: Symbol, target: Decimal, open_orders: list[OrderUpdate]
    ) -> CancelMultiResult:
        """
        Cancel pending orders to approach the target.

        open_orders comes from the caller's tick snapshot so the tick costs a
        single open-orders fetch instead of one per decision.

        The remainder is computed from cancels that actually succeeded, not
        from the pre-cancel selection: a cancel blocked by the rate limiter
        (or rejected by the exchange) leaves its order live, so counting it
        as freed would size a same-tick replacement on top of an order that
        never went away - doubling the position if both then fill.
        """
        # remove more than needed then return value to compensate it
        # REASON better to use the rate limits than to pay the fees twice
        if target == Decimal("0"):
            raise ValueError("Target shouldn't be zero")

        side = OrderSide.BUY if target > Decimal("0") else OrderSide.SELL
        to_be_removed_orders = [order for order in open_orders if order.side != side]
        indexed_orders = sorted(
            # BUY target: smallest opposite-side orders first to minimise over-cancel
            # SELL target: largest opposite-side orders first to reach target quickly
            enumerate(to_be_removed_orders),
            key=lambda x: x[1].remain_size,
            reverse=(side != OrderSide.BUY),
        )

        selected_sum = Decimal("0")
        chosen_indices: list[int] = []

        for index, order in indexed_orders:
            if side == OrderSide.BUY and selected_sum + order.remain_size < target:
                selected_sum += order.remain_size
                chosen_indices.append(index)
            elif side == OrderSide.SELL and selected_sum - order.remain_size > target:
                selected_sum -= order.remain_size
                chosen_indices.append(index)
            # To add the final order to exceed target if available
            else:
                selected_sum += (
                    order.remain_size if side == OrderSide.BUY else -order.remain_size
                )
                chosen_indices.append(index)
                break

        sem = asyncio.Semaphore(MAX_CONCURRENT_ORDER_OPS)

        async def _bounded_cancel(index: int):
            async with sem:
                return await self.cancel_single_order(
                    symbol=open_orders[index].symbol,
                    client_order_id=open_orders[index].client_order_id,
                )

        cancelled_sum = Decimal("0")
        failed_count = 0

        async with self.order_pools.get_order_backlog() as order_backlog:
            cancel_tasks = [_bounded_cancel(index) for index in chosen_indices]

            cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

            for index, result in zip(chosen_indices, cancel_results):
                if isinstance(result, BaseException):
                    logger.error(f"Failed to cancel order due to, {result}")
                    failed_count += 1
                    continue
                if isinstance(result, CancelBacklogs):
                    self.order_pools.dedup_append(order_backlog, result)
                    failed_count += 1
                    continue
                # result is None: cancel_single_order confirmed this order is
                # actually gone (or terminally gone) on the exchange.
                order = open_orders[index]
                cancelled_sum += (
                    order.remain_size if side == OrderSide.BUY else -order.remain_size
                )

        logger.info(
            f"[CANCEL_MULTI_LIMIT_ORDER] Cancelled total {cancelled_sum} {symbol} worth of open orders"
            + (f", {failed_count} cancel(s) failed" if failed_count else "")
        )
        return CancelMultiResult(
            remainder=target - cancelled_sum, failed_count=failed_count
        )

    async def place_single_limit_order(
        self,
        symbol: Symbol,
        offset: Decimal,
        qty: Decimal,
        side: OrderSide,
        symbol_info: SymbolInfo,
        replace_best_bid_ask_time: AwareDatetime | None = None,
        package_id: str | None = None,
        initial_price: Decimal | None = None,
        initial_time: datetime | None = None,
        client_order_id: str | None = None,
    ) -> OrderBacklogs | None:
        """
        Place a single limit order on the exchange

        OrderDetails replace_best_bid_ask always updates to the latest version
        unless replace_best_bid_ask_time is specified

        Package ID is for to be repriced orders, ignore if it is a new order

        client_order_id is for backlog retries only: reusing the original id lets
        the exchange dedupe when the first attempt was sent but its response was lost

        return
        None if successful
        OrderBacklogs if fail
        """
        post_order_endpoint = Endpoints.PLACE_ORDER
        get_depth_endpoint = Endpoints.GET_ORDERBOOK_SNAPSHOT

        current_time = datetime.now(timezone.utc)
        replace_order_interval_in_sec = current_time + timedelta(
            seconds=round(
                random.uniform(
                    self.config.min_limit_replace_interval,
                    self.config.max_limit_replace_interval,
                ),
                2,
            )
        )
        replace_best_bid_ask_time = (
            replace_best_bid_ask_time
            if replace_best_bid_ask_time is not None
            else current_time + timedelta(seconds=self.config.replace_best_bid_ask_time)
        )
        client_order_id = (
            client_order_id
            if client_order_id is not None
            else OrderUtils.make_client_order_id()
        )
        current_package_id = package_id if package_id else self.package_id
        try:
            order_book = await OrderUtils.get_order_book(
                exchange=self.exchange,
                pair=symbol,
                need_log=True,
                rate_limiter=self.rate_limiter,
                endpoint=get_depth_endpoint,
            )

            price = order_book[0] if side == OrderSide.BUY else order_book[1]
            adjusted_price = Calculate.align_price(
                limit_price=price,
                side=side,
                offset=offset,
                tick_size=symbol_info.tick_size,
                precision=symbol_info.price_precision,
            )

            # Register before the REST send: a WS fill can beat the REST
            # response, and validate_oms_state must not read that as a desync
            self.order_pools.register_pending_insert(client_order_id)
            async with self.rate_limiter.guard(endpoint=post_order_endpoint):
                await self.exchange.place_order(
                    limit=adjusted_price,
                    side=side,
                    quantity=qty,
                    symbol=symbol,
                    client_order_id=client_order_id,
                    post_only=True,
                )

        except Exception as e:
            if self.error_policy.classify(e) is ErrorAction.FATAL:
                # Nothing was placed; dropping is correct, retrying is futile
                logger.error(
                    f"[PLACE_SINGLE_LIMIT_ORDER] FATAL error, not retrying: {e}"
                )
                return None
            logger.warning(
                f"[PLACE_SINGLE_LIMIT_ORDER] Order placement failed, reason {e}"
            )
            return OrderBacklogs(
                offset=offset,
                ori_entry_time=datetime.now(timezone.utc),
                qty=qty,
                max_replace_limit_order_time=replace_order_interval_in_sec,
                side=side,
                replace_best_bid_ask_time=replace_best_bid_ask_time,
                symbol=str(symbol),
                total_retries=1,
                client_order_id=client_order_id,
                package_id=current_package_id,
                initial_price=initial_price,
                initial_time=initial_time,
            )

        logger.info(f"[PLACE_LIMIT] Placed {side} {qty} {symbol} @ {adjusted_price}")
        initial_price = initial_price if initial_price else adjusted_price
        initial_time = initial_time if initial_time else current_time
        async with self.order_pools.get_order_pool() as order_pool:
            order_pool[client_order_id] = OrderDetails(
                client_order_id=client_order_id,
                symbol=str(symbol),
                max_replace_limit_order_time=replace_order_interval_in_sec,
                replace_best_bid_ask_time=replace_best_bid_ask_time,
                remain_size=qty,
                side=side,
                price=adjusted_price,
                package_id=current_package_id,
                initial_price=initial_price,
                initial_time=initial_time,
            )
        self.order_pools.confirm_pending_insert(client_order_id)

        if current_package_id not in self.order_pools.order_records:
            self.order_pools.order_records[current_package_id] = []
        self.order_pools.order_records[current_package_id].append(
            (symbol, client_order_id, initial_price, initial_time)
        )

    async def get_current_price(self, symbol: Symbol) -> Decimal | None:
        """
        Canonical current-price fetch: reserves a rate-limit slot and returns
        None on any failure. All price reads go through here so behaviour is
        consistent (waits under contention, never raises to the caller).
        """
        endpoint = Endpoints.GET_ORDERBOOK_SNAPSHOT
        try:
            async with self.rate_limiter.reserve(endpoint=endpoint):
                return await self.exchange.get_current_price(symbol=symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch current price due to {e}")
            return None

    async def reprice_at_mid(
        self,
        symbol: Symbol,
        qty: Decimal,
        side: OrderSide,
        replace_best_bid_ask_time: AwareDatetime,
        package_id: str,
        initial_price: Decimal,
        initial_time: datetime | None = None,
    ) -> OrderBacklogs | None:
        """
        Replaces orders at a refreshed price so they are more likely to be filled.

        The offset comes from the compute_limit_offsets hook (level=1); the
        naive base hook makes this equivalent to repricing at BBO.

        The order being repriced was already cancelled, so a failure to compute
        the offset must not drop the quantity — fall back to BBO (offset 0)
        and let place_single_limit_order handle placement failures via backlog
        """
        symbol_info = self.symbol_infos[symbol]

        offset = Decimal("0")
        market_price = await self.get_current_price(symbol=symbol)
        if market_price is not None:
            try:
                ctx = self._make_context(
                    symbol=symbol, side=side, market_price=market_price
                )
                offset = (await self.compute_limit_offsets(1, ctx))[0]
            except Exception as e:
                logger.warning(
                    f"[REPRICE_AT_MID] Offset calculation for {symbol} failed due to {e}, falling back to BBO"
                )

        return await self.place_single_limit_order(
            symbol=symbol,
            offset=offset,
            qty=qty,
            side=side,
            symbol_info=symbol_info,
            replace_best_bid_ask_time=replace_best_bid_ask_time,
            package_id=package_id,
            initial_price=initial_price,
            initial_time=initial_time,
        )

    async def reprice_at_bbo(
        self,
        symbol: Symbol,
        qty: Decimal,
        side: OrderSide,
        replace_best_bid_ask_time: AwareDatetime,
        package_id: str,
        initial_price: Decimal | None = None,
        initial_time: datetime | None = None,
    ) -> OrderBacklogs | None:
        """
        Replaces expired orders immediately at the current Best Bid or Offer (BBO).

        This function is a fallback mechanism for orders that have passed their
        expiry time without being executed or cancelled by other logic.

        Rationale (Why at BBO?):
        The fact that the order expired implies the system critically needs this position filled.
        If a significant market movement occurred (causing a material change in delta),
        the order should have been cancelled by delta logic before it ever reached its expiry time.
        """
        return await self.place_single_limit_order(
            symbol=symbol,
            offset=Decimal("0"),
            qty=qty,
            side=side,
            symbol_info=self.symbol_infos[symbol],
            replace_best_bid_ask_time=replace_best_bid_ask_time,
            package_id=package_id,
            initial_price=initial_price,
            initial_time=initial_time,
        )

    async def place_multiple_limit_order(
        self,
        symbol: Symbol,
        quantity: Decimal,
    ):
        """
        Place a set of limit orders which total up to quantity.

        Sizing and pricing come from the split_order_quantity and
        compute_limit_offsets hooks; orchestration and failure handling stay
        here. A hook failure degrades to the naive defaults rather than
        dropping the rebalance.
        """

        order_side = OrderSide.BUY if quantity > Decimal("0") else OrderSide.SELL
        qty = abs(quantity)

        market_price = await self.get_current_price(symbol=symbol)
        if market_price is None:
            logger.warning(
                f"[PLACE_MULTI_LIMIT_ORDER] No market price for {symbol}, skipping this cycle; delta will retry next placement"
            )
            return

        ctx = self._make_context(
            symbol=symbol, side=order_side, market_price=market_price
        )

        try:
            random_order_size = await self.split_order_quantity(qty, ctx)
        except Exception as e:
            logger.warning(
                f"[PLACE_MULTI_LIMIT_ORDER] split_order_quantity failed due to {e}, using naive sizing"
            )
            # Pinned to the base implementation so a broken override cannot recurse
            random_order_size = await OrderExecutor.split_order_quantity(self, qty, ctx)

        try:
            limit_offsets = await self.compute_limit_offsets(
                len(random_order_size), ctx
            )
        except Exception as e:
            logger.warning(
                f"[PLACE_MULTI_LIMIT_ORDER] compute_limit_offsets failed due to {e}, falling back to BBO"
            )
            limit_offsets = [Decimal("0")] * len(random_order_size)

        if len(limit_offsets) != len(random_order_size):
            logger.warning(
                "[PLACE_MULTI_LIMIT_ORDER] Offset count does not match order count, falling back to BBO"
            )
            limit_offsets = [Decimal("0")] * len(random_order_size)

        logger.info(
            f"[PLACE_MULTI_LIMIT_ORDER] Placing {len(random_order_size)} limit orders, with price offset of {limit_offsets}"
        )

        sem = asyncio.Semaphore(MAX_CONCURRENT_ORDER_OPS)

        async def _bounded_place(i: int):
            async with sem:
                return await self.place_single_limit_order(
                    symbol=symbol,
                    offset=limit_offsets[i],
                    qty=random_order_size[i],
                    side=order_side,
                    symbol_info=ctx.symbol_info,
                )

        async with self.order_pools.get_order_backlog() as order_backlog:
            order_tasks = [
                _bounded_place(i)
                for i in range(len(random_order_size))
                if random_order_size[i] > Decimal("0")
            ]

            order_results = await asyncio.gather(*order_tasks, return_exceptions=True)
            for result in order_results:
                if isinstance(result, BaseException):
                    logger.error(f"Failed to place order due to, {result}")
                    continue
                if isinstance(result, OrderBacklogs):
                    order_backlog.append(result)

import logging
import asyncio
import random
import time

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pydantic import AwareDatetime, BaseModel, Field
from contextlib import asynccontextmanager

from adrs.oms.config import ConfigManager
from adrs.oms.rate_limit.rate_limiter import RateLimiter
from adrs.oms.rate_limit.exchange_limit_profiles import (
    Endpoints,
    OpenOrdersFetchProfile,
    OPEN_ORDERS_FETCH_PROFILES,
)

from cybotrade.io import ExchangeClient
from cybotrade import Symbol
from cybotrade.models import (
    OrderSide,
    OrderUpdate,
)

logger = logging.getLogger(__name__)

# How long a registered-but-unconfirmed placement counts as in-flight. Past
# this, an unknown WS update for the id is treated as a real desync again.
PENDING_INSERT_TTL_SEC = 60


class BacklogDetails(BaseModel):
    symbol: str
    total_retries: int
    next_retry_at: AwareDatetime | None = None  # None = due now
    client_order_id: str


class OrderBacklogs(BacklogDetails):
    side: OrderSide
    ori_entry_time: AwareDatetime
    qty: Decimal
    offset: Decimal
    replace_best_bid_ask_time: AwareDatetime
    max_replace_limit_order_time: AwareDatetime
    package_id: str
    initial_price: Decimal | None = None
    initial_time: datetime | None = None


class CancelBacklogs(BacklogDetails):
    pass


class ExpiredBacklogs(BacklogDetails):
    # For OrderBacklogs
    side: OrderSide
    qty: Decimal
    replace_best_bid_ask_time: AwareDatetime
    max_replace_limit_order_time: AwareDatetime
    # For CancelBacklogs
    is_bbo: bool
    # For Aegis
    package_id: str
    initial_price: Decimal
    initial_time: datetime


class OrderDetails(BaseModel):
    # Order Management
    symbol: str
    side: OrderSide
    price: Decimal
    remain_size: Decimal
    client_order_id: str
    replace_best_bid_ask_time: AwareDatetime
    max_replace_limit_order_time: AwareDatetime
    # Which signal group does it belong to
    package_id: str
    initial_price: Decimal
    initial_time: datetime
    # When this entry landed in the local pool; unlike initial_time it is not
    # carried across reprices, so resync can tell fresh entries from ghosts
    inserted_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class OpenOrdersSnapshot:
    taken_at: datetime  # when the REST pass began
    orders: dict[Symbol, list[OrderUpdate]]


class OrderPoolHandler:
    def __init__(
        self,
        exchange: ExchangeClient,
        config: ConfigManager,
        rate_limiter: RateLimiter,
    ):
        self.order_pool: dict[str, OrderDetails] = {}
        self._pool_lock = asyncio.Lock()
        # Placements whose REST response hasn't landed yet: a WS update can
        # beat the pool insert, and must not be read as a desync
        self.pending_inserts: dict[str, float] = {}  # client_order_id -> monotonic ts
        self.order_backlog: list[BacklogDetails] = []
        self._backlog_lock = asyncio.Lock()
        self.order_value_update: dict[str, Decimal] = {}
        self.order_records: dict[
            str, list[tuple[Symbol, str, Decimal, datetime]]
        ] = {}  # package id, list of client order id
        self.exchange = exchange
        self.config = config
        self.rate_limiter = rate_limiter

    @asynccontextmanager
    async def get_order_backlog(self):
        await self._backlog_lock.acquire()
        try:
            yield self.order_backlog
        finally:
            self._backlog_lock.release()

    @staticmethod
    def dedup_append(order_backlog: list[BacklogDetails], item: BacklogDetails) -> bool:
        """Append only if no entry with the same client_order_id is queued."""
        if any(b.client_order_id == item.client_order_id for b in order_backlog):
            return False
        order_backlog.append(item)
        return True

    def register_pending_insert(self, client_order_id: str):
        self.pending_inserts[client_order_id] = time.monotonic()

    def confirm_pending_insert(self, client_order_id: str):
        self.pending_inserts.pop(client_order_id, None)

    def is_pending_insert(self, client_order_id: str) -> bool:
        """
        Entries self-expire after PENDING_INSERT_TTL_SEC, so failure paths
        (ambiguous REST errors where the order may still exist) never need
        explicit cleanup and can't leak.
        """
        now = time.monotonic()
        expired = [
            cid
            for cid, ts in self.pending_inserts.items()
            if now - ts > PENDING_INSERT_TTL_SEC
        ]
        for cid in expired:
            del self.pending_inserts[cid]
        return client_order_id in self.pending_inserts

    @asynccontextmanager
    async def get_order_pool(self):
        await self._pool_lock.acquire()
        try:
            yield self.order_pool
        finally:
            self._pool_lock.release()

    async def snapshot(
        self,
    ) -> tuple[dict[str, "OrderDetails"], list["BacklogDetails"]]:
        """Return frozen copies of the pool and backlog for assertions in tests."""
        async with self.get_order_pool() as pool:
            pool_copy = dict(pool)
        async with self.get_order_backlog() as backlog:
            backlog_copy = list(backlog)
        return pool_copy, backlog_copy

    async def fetch_open_orders_snapshot(self) -> OpenOrdersSnapshot:
        """
        One REST pass over open orders, shared within a single event
        (placement tick / desync resync / init).

        A symbol absent from the result means its fetch failed; consumers
        keep their previous state for it rather than treating it as empty.
        """
        taken_at = datetime.now(timezone.utc)
        symbols = [
            Symbol(s) for s in self.config.config.base_asset_to_symbol_table.values()
        ]

        profile = OPEN_ORDERS_FETCH_PROFILES.get(
            self.config.config.credentials.exchange
        )
        if profile is not None and len(symbols) >= profile.min_symbols_for_bulk:
            orders = await self._fetch_bulk_open_orders(symbols, profile)
            if orders is not None:
                return OpenOrdersSnapshot(taken_at=taken_at, orders=orders)

        orders = {}
        for symbol in symbols:
            try:
                async with self.rate_limiter.guard(endpoint=Endpoints.GET_OPEN_ORDERS):
                    orders[symbol] = await self.exchange.get_open_orders(symbol=symbol)
            except Exception as e:
                logger.warning(f"Failed to fetch open orders for {symbol} due to {e}")
        return OpenOrdersSnapshot(taken_at=taken_at, orders=orders)

    async def _fetch_bulk_open_orders(
        self, symbols: list[Symbol], profile: OpenOrdersFetchProfile
    ) -> dict[Symbol, list[OrderUpdate]] | None:
        """
        All symbols in one request. None means bulk is unusable this round
        (error, or the response may be truncated at the exchange's page
        limit) and the caller must fall back to per-symbol fetches.
        """
        try:
            async with self.rate_limiter.guard(endpoint=Endpoints.GET_OPEN_ORDERS_ALL):
                all_orders = await self.exchange.get_open_orders(
                    symbol=None, **profile.bulk_kwargs
                )
        except Exception as e:
            logger.warning(
                f"Bulk open-orders fetch failed due to {e}, falling back to per-symbol"
            )
            return None
        if (
            profile.truncation_row_limit is not None
            and len(all_orders) >= profile.truncation_row_limit
        ):
            logger.warning(
                "Bulk open-orders response filled the page, possible truncation, falling back to per-symbol"
            )
            return None
        # Bulk saw everything, so absent symbol = no open orders (empty list,
        # unlike a failed per-symbol fetch which omits the key). Orders on
        # symbols outside the config table are dropped to match per-symbol
        # behavior — the OMS must not adopt orders it doesn't trade.
        orders: dict[Symbol, list[OrderUpdate]] = {symbol: [] for symbol in symbols}
        for order in all_orders:
            if order.symbol in orders:
                orders[order.symbol].append(order)
        return orders

    async def resync_order_pool(self, snapshot: OpenOrdersSnapshot):
        """
        Expensive function only resync when needed
        """
        current_time = datetime.now(timezone.utc)
        replace_order_interval_in_sec = current_time + timedelta(
            seconds=round(
                random.uniform(
                    self.config.config.min_limit_replace_interval,
                    self.config.config.max_limit_replace_interval,
                ),
                2,
            )
        )
        new_order_pool: dict[str, OrderDetails] = {}
        client_to_package = {
            record[1]: pkg_id
            for pkg_id, records in self.order_records.items()
            for record in records
        }
        try:
            # Snapshot the old pool once so timer carryover is read from stable
            # state, not the dict we are rebuilding mid-loop.
            old_order_pool = self.order_pool
            for order_list in snapshot.orders.values():
                for order in order_list:
                    prev = old_order_pool.get(order.client_order_id)
                    new_order_pool[order.client_order_id] = OrderDetails(
                        client_order_id=order.client_order_id,
                        replace_best_bid_ask_time=prev.replace_best_bid_ask_time
                        if prev is not None
                        else datetime.now(timezone.utc)
                        + timedelta(
                            seconds=self.config.config.replace_best_bid_ask_time
                        ),
                        max_replace_limit_order_time=(
                            prev.max_replace_limit_order_time
                            if prev is not None
                            else replace_order_interval_in_sec
                        ),
                        symbol=str(order.symbol),
                        remain_size=order.remain_size,
                        side=order.side,
                        price=order.price,
                        package_id=package_id
                        if (package_id := client_to_package.get(order.client_order_id))
                        else "",
                        initial_price=Decimal("0"),
                        initial_time=datetime.now(tz=timezone.utc),
                    )
            # Merge, then swap under the pool lock. An entry missing from the
            # snapshot is only a ghost if REST could have seen it: entries for
            # failed-fetch symbols and entries inserted after the REST pass
            # began are carried over, not dropped.
            fetched_symbols = {str(s) for s in snapshot.orders.keys()}
            async with self.get_order_pool():
                for cid, details in self.order_pool.items():
                    if cid in new_order_pool:
                        continue
                    if (
                        details.symbol not in fetched_symbols
                        or details.inserted_at > snapshot.taken_at
                    ):
                        new_order_pool[cid] = details
                self.order_pool = new_order_pool
        except Exception as e:
            logger.warning(f"Failed to resync_order_pool due to {e}")

import logging
import asyncio
import random

from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pydantic import AwareDatetime, BaseModel
from contextlib import asynccontextmanager

from adrs.oms.config import ConfigManager
from adrs.oms.rate_limit.rate_limiter import RateLimiter
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints

from cybotrade.io import ExchangeClient
from cybotrade import Symbol
from cybotrade.models import (
    OrderSide,
)

logger = logging.getLogger(__name__)


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


class OrderPoolHandler:
    def __init__(
        self,
        exchange: ExchangeClient,
        config: ConfigManager,
        rate_limiter: RateLimiter,
    ):
        self.order_pool: dict[str, OrderDetails] = {}
        self._pool_lock = asyncio.Lock()
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

    async def resync_order_pool(self):
        """
        Expensive function only resync when needed
        """
        endpoint = Endpoints.GET_OPEN_ORDERS
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
            for symbol in self.config.symbol_infos.keys():
                async with self.rate_limiter.guard(endpoint=endpoint):
                    order_list = await self.exchange.get_open_orders(symbol=symbol)
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
            # Swap the rebuilt pool in once, after the loop, under the pool lock.
            async with self.get_order_pool():
                self.order_pool = new_order_pool
        except Exception as e:
            logger.warning(f"Failed to resync_order_pool due to {e}")

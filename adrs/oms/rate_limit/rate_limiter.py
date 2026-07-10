import asyncio

from abc import abstractmethod, ABC
from collections import deque
from contextlib import asynccontextmanager
from decimal import Decimal
import time

import logging
from typing import Any, Dict, AsyncGenerator

from adrs.oms.config import ConfigManager
from adrs.oms.rate_limit.exchange_limit_profiles import (
    BinanceLimitProfile,
    BINANCE_FUTURES_COSTS,
    BybitLimitState,
    Endpoints,
    get_depth_weight,
    BybitRateLimitPool,
    BybitLimitProfile,
    BYBIT_FUTURES_COSTS,
)

from cybotrade.binance import BinanceLinearClient, BinanceError
from cybotrade.bybit import BybitLinearClient, BybitError

logger = logging.getLogger(__name__)

RESERVE_TIMEOUT_SEC = 1.5
_MIN_RESERVE_SLEEP = 0.005

# Only these header prefixes carry rate-limit signal; everything else (auth,
# account/IP identifiers, cookies) is dropped before logging.
_RATE_LIMIT_HEADER_PREFIXES = (
    "retry-after",
    "x-mbx-used-weight",
    "x-mbx-order-count",
    "x-bapi-limit",
)


def _redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Keep only rate-limit headers; never log raw exchange headers."""
    return {
        k: v
        for k, v in headers.items()
        if k.lower().startswith(_RATE_LIMIT_HEADER_PREFIXES)
    }


class LocalRateLimitError(Exception):
    """
    Raised when the local rate limiter blocks a request
    before it is sent to the exchange.
    """

    def __init__(self, message="Local rate limit exceeded"):
        self.message = message
        super().__init__(self.message)


# TODO add lock for multi process
class RateLimiter(ABC):
    # epoch ms until which all calls are locally blocked after a 418/429
    retry_after: int = 0

    def __init__(
        self,
        config: ConfigManager,
    ):
        self.config = config
        self.soft_limit_percentage = config.config.soft_limit_percent
        # Per-pool FIFO queue for reserve(); asyncio.Lock wakes waiters in
        # acquisition order, so only one counts down at a time (no stampede)
        self._reserve_locks: dict[Any, asyncio.Lock] = {}
        # Per-pool count of callers currently in reserve(); guard() yields
        # while this is > 0 so reserved calls take priority
        self._waiters: dict[Any, int] = {}

    @abstractmethod
    async def init(self): ...

    @abstractmethod
    def get_synced_time_ms(self) -> int:
        """
        Current exchange-synced time in epoch ms
        """
        ...

    @asynccontextmanager
    @abstractmethod
    async def guard(self, endpoint: Endpoints) -> AsyncGenerator[None, None]:
        """
        Context manager to wrap exchange calls with rate limit checks.

        Usage:
            async with self.rate_limiter.guard(Endpoints.PLACE_ORDER):
                await exchange.place_order(...)

        Raises:
            LocalRateLimitError: If local limits are exhausted (Pre-check).
        """
        yield

    def _reserve_lock(self, key: Any) -> asyncio.Lock:
        lock = self._reserve_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._reserve_locks[key] = lock
        return lock

    def _waiters_repr(self) -> str:
        """Per-pool reserve() queue depth, for the stats log; empty if idle."""
        active = {k: n for k, n in self._waiters.items() if n > 0}
        if not active:
            return ""
        parts = ", ".join(f"{getattr(k, 'name', k)}={n}" for k, n in active.items())
        return f" Reserving[{parts}]"

    @asynccontextmanager
    async def reserve(self, endpoint: Endpoints) -> AsyncGenerator[None, None]:
        """
        Like guard(), but waits for a slot instead of failing fast when the
        pool is full. While a caller is queued here, guard() callers on the
        same pool yield, so a reserved (delta-critical) call gets priority
        when capacity frees.

        Waiters on a pool are serialised by a lock, so they acquire slots in
        FIFO order and only one is ever counting down — no wake-up stampede.

        Usage:
            async with self.rate_limiter.reserve(Endpoints.GET_OPEN_ORDERS):
                await exchange.get_open_orders(...)

        Raises:
            LocalRateLimitError: on wait timeout, or immediately while
            retry_after is active — so callers keep their guard fallback.
        """
        key = self._pool_key(endpoint)
        deadline = time.monotonic() + RESERVE_TIMEOUT_SEC
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            async with self._reserve_lock(key):
                while True:
                    if self.retry_after >= self.get_synced_time_ms():
                        raise LocalRateLimitError(
                            f"reserve aborted, rate limiter cooling down {self}"
                        )
                    if self._has_capacity(endpoint):
                        self.record_usage(endpoint)
                        break
                    delay = self._next_free_delay(endpoint)
                    if time.monotonic() + delay > deadline:
                        raise LocalRateLimitError(
                            f"reserve timed out waiting for a slot {self}"
                        )
                    await asyncio.sleep(max(delay, _MIN_RESERVE_SLEEP))
        finally:
            self._waiters[key] -= 1
        try:
            yield
        except Exception as e:
            self._handle_call_error(e)
            raise e

    @abstractmethod
    def _pool_key(self, endpoint: Endpoints) -> Any:
        """Key identifying the contended pool, for reserve queueing/priority."""
        ...

    @abstractmethod
    def _has_capacity(self, endpoint: Endpoints, **kwargs) -> bool:
        """
        Raw capacity check for one call, ignoring retry_after and the reserve
        yield rule (reserve() and check_limits() layer those on top).
        """
        ...

    @abstractmethod
    def _next_free_delay(self, endpoint: Endpoints) -> float:
        """Seconds until the pool frees at least one slot for this endpoint."""
        ...

    @abstractmethod
    def _handle_call_error(self, e: Exception) -> None:
        """Fold an exchange rate-limit error into local retry_after state."""
        ...

    @abstractmethod
    async def on_resync_time(self):
        """
        To resync with server time
        """
        ...

    @abstractmethod
    def reset_limits(self):
        """
        To reset limits based on exchange rules
        """
        ...

    @abstractmethod
    def check_limits(self, endpoint: Endpoints, **kwargs) -> bool:
        """
        To determine whether to allow the endpoint request
        """
        ...

    @abstractmethod
    def record_usage(self, endpoint: Endpoints):
        """
        To record usage once endpoint request was successful
        """
        ...

    @abstractmethod
    def local_cache_error(self, headers: dict[str, Any]):
        """
        To record usage once endpoint request was failure, local store has been desyncronized
        """
        ...

    @abstractmethod
    def __repr__(self) -> str: ...


class BinanceRateLimiter(RateLimiter):
    def __init__(self, config: ConfigManager):
        if not isinstance(config.exchange, BinanceLinearClient):
            raise Exception("Exchange mismatch with rate limiter")
        super().__init__(config)
        exchange_info = config.exchange.exchange_info
        if exchange_info is None:
            raise Exception("Init the config before passing into Rate Limiter")

        rateLimits: list[dict[str, Any]] | None = exchange_info.get("rateLimits")
        if rateLimits is None:
            raise Exception("Check Binance Api, rateLimits shouldn't be None here")

        limit_profile = BinanceLimitProfile(
            request_weight_limit_per_minute=0,
            order_limit_per_minute=0,
            order_limit_per_10_sec=0,
        )

        for rule in rateLimits:
            try:
                limit_type = rule["rateLimitType"]
                interval = rule["interval"]
                interval_num = rule["intervalNum"]
                limit = rule["limit"]
            except Exception as e:
                logger.error(
                    f"[BinanceRateLimiter] Failed parsing for {rule} due to {e}"
                )
                continue

            if (
                limit_type == "REQUEST_WEIGHT"
                and interval == "MINUTE"
                and interval_num == 1
            ):
                limit_profile.request_weight_limit_per_minute = (
                    limit * self.soft_limit_percentage
                )
            elif limit_type == "ORDERS" and interval == "MINUTE" and interval_num == 1:
                limit_profile.order_limit_per_minute = (
                    limit * self.soft_limit_percentage
                )
            elif limit_type == "ORDERS" and interval == "SECOND" and interval_num == 10:
                limit_profile.order_limit_per_10_sec = (
                    limit * self.soft_limit_percentage
                )
            else:
                logger.warning(f"Unknown rate limit {rule}")
                continue

        self.limit_profile = limit_profile
        self.current_limit_state = BinanceLimitProfile(
            request_weight_limit_per_minute=0,
            order_limit_per_minute=0,
            order_limit_per_10_sec=0,
        )
        self.exchange = config.exchange
        self.exchange_time_offset = 0

        self.last_reset_10s_timestamp = 0
        self.last_reset_1m_timestamp = 0

        # to block all calls until
        self.retry_after = 0

    async def init(self):
        await self.on_resync_time()
        self.last_reset_10s_timestamp = self.get_synced_time_ms() // 10000  # 10 seconds
        self.last_reset_1m_timestamp = self.get_synced_time_ms() // 60000  # 1 minute

    @asynccontextmanager
    async def guard(self, endpoint: Endpoints) -> AsyncGenerator[None, None]:
        if not self.check_limits(endpoint=endpoint):
            raise LocalRateLimitError(f"Failed due to rate limits {self}")

        self.record_usage(endpoint=endpoint)
        try:
            yield
        except Exception as e:
            self._handle_call_error(e)
            raise e

    def _handle_call_error(self, e: Exception) -> None:
        if isinstance(e, BinanceError) and (e.code == 418 or e.code == 429):
            self.local_cache_error(e.response_headers if e.response_headers else {})

    def _pool_key(self, endpoint: Endpoints) -> Any:
        # Binance draws almost everything from the shared weight budget, so a
        # single queue suffices; reserve on Binance usually times out anyway
        return "binance_weight"

    async def on_resync_time(self):
        endpoint = Endpoints.GET_SERVER_TIME
        try:
            async with self.guard(endpoint=endpoint):
                exchange_time = await self.exchange.get_server_time()
                current_time = int(time.time() * 1000)
                self.exchange_time_offset = current_time - exchange_time
        except Exception as e:
            logger.warning(f"[ON_RESYNC_TIME] {e}")

    def get_synced_time_ms(self) -> int:
        current_time = int(time.time() * 1000)
        return current_time - self.exchange_time_offset

    def _next_free_delay(self, endpoint: Endpoints) -> float:
        # Weight and 1m orders reset on the minute boundary, 10s orders on the
        # 10s boundary. Return the sooner one and let the reserve loop recheck;
        # on Binance this usually exceeds the reserve timeout and degrades to
        # guard behaviour, which is acceptable.
        now = self.get_synced_time_ms()
        next_1m = ((now // 60000) + 1) * 60000
        next_10s = ((now // 10000) + 1) * 10000
        return (min(next_1m, next_10s) - now) / 1000.0

    def reset_limits(self):
        """
        Will reset limits based on time intervals

        Source: https://developers.binance.com/docs/binance-spot-api-docs/websocket-api/rate-limits
        """
        synced_time = self.get_synced_time_ms()
        synced_time_10s = synced_time // 10000  # 10 seconds
        synced_time_1m = synced_time // 60000  # 1 minute
        # if different means next interval has passed
        if synced_time_10s != self.last_reset_10s_timestamp:
            self.last_reset_10s_timestamp = synced_time_10s
            self.current_limit_state.order_limit_per_10_sec = 0
        if synced_time_1m != self.last_reset_1m_timestamp:
            self.last_reset_1m_timestamp = synced_time_1m
            self.current_limit_state.order_limit_per_minute = 0
            self.current_limit_state.request_weight_limit_per_minute = 0

    def find_cost_info(self, endpoint: Endpoints) -> tuple[int, int]:
        cost_info = BINANCE_FUTURES_COSTS.get(endpoint)
        if cost_info is None:
            logger.error(
                f"[CHECK_LIMITS] Severe error no cost info is defined for this endpoint: {endpoint.name}"
            )
            raise KeyError(f"Endpoint doesn't exist, {endpoint.name}")

        return (cost_info["weight"], cost_info["orders"])

    def check_limits(self, endpoint: Endpoints, **kwargs) -> bool:
        """
        Whether a guard() call may proceed: blocked while retry_after is
        active, and yields to anything queued in reserve() for this pool.
        POST_ORDER/ Order Creation is exempted from this rule
        """
        # Absulute condition if retry after is active will not do anything
        if self.retry_after >= self.get_synced_time_ms():
            return False
        # Yield to callers waiting in reserve() so reserved calls take priority
        if self._waiters.get(self._pool_key(endpoint), 0) > 0:
            return False
        return self._has_capacity(endpoint, **kwargs)

    def _has_capacity(self, endpoint: Endpoints, **kwargs) -> bool:
        self.reset_limits()
        try:
            (weight_cost, order_cost) = self.find_cost_info(endpoint=endpoint)
            params = {**kwargs}
            # Means it is dynamic
            if weight_cost == -1:
                if endpoint == Endpoints.GET_ORDERBOOK_SNAPSHOT:
                    weight_cost = (
                        get_depth_weight(params["depth"])
                        if "depth" in params.keys()
                        else get_depth_weight()
                    )
            # Check in decending order by timescale
            # Checking REQUEST_WEIGHT
            current_weight = self.current_limit_state.request_weight_limit_per_minute
            max_weight = self.limit_profile.request_weight_limit_per_minute
            if weight_cost != 0 and max_weight <= (current_weight + weight_cost):
                logger.warning(
                    f"[CHECK_LIMITS] REQUEST_WEIGHT 1m reached its limit\n{max_weight} <= {current_weight} + {weight_cost}"
                )
                return False
            # Checking ORDERS 1m
            current_orders_1m = self.current_limit_state.order_limit_per_minute
            max_orders_1m = self.limit_profile.order_limit_per_minute
            if order_cost != 0 and max_orders_1m <= (current_orders_1m + order_cost):
                logger.warning(
                    f"[CHECK_LIMITS] ORDERS 1m reached its limit\n{max_orders_1m} <= {current_orders_1m} + {order_cost}"
                )
                return False
            # Checking ORDERS 10s
            current_orders_10s = self.current_limit_state.order_limit_per_10_sec
            max_orders_10s = self.limit_profile.order_limit_per_10_sec
            if order_cost != 0 and max_orders_10s <= (current_orders_10s + order_cost):
                logger.warning(
                    f"[CHECK_LIMITS] ORDERS 10s reached its limit\n{max_orders_10s} <= {current_orders_10s} + {order_cost}"
                )
                return False
            # Passed all checks
            return True
        except Exception as e:
            logger.error(f"Failed to check limits due to, {e}")
            return False

    def record_usage(self, endpoint: Endpoints):
        """
        Update limit values after successful endpoint request
        """
        try:
            (weight_cost, order_cost) = self.find_cost_info(endpoint=endpoint)
            self.current_limit_state.order_limit_per_10_sec += order_cost
            self.current_limit_state.order_limit_per_minute += order_cost
            self.current_limit_state.request_weight_limit_per_minute += weight_cost
        except Exception as e:
            logger.error(f"Failed to record usage due to {e}")

    def local_cache_error(self, headers: dict[str, Any]):
        """
        Use when binance api return 429 (rate limit exhausted) and 418 (IP banned) status codes

        Will block any subsequent request based on retry after value from headers
        """

        logger.warning(f"[LOCAL CACHE ERROR] HEADERS {_redact_headers(headers)}")
        # Request Weight exhausted
        if "Retry-After" in headers.keys():
            self.retry_after = (
                self.get_synced_time_ms()
                + int(headers["Retry-After"]) * 1000
                + 1000  # safety buffer
            )
            self.current_limit_state.request_weight_limit_per_minute = (
                self.limit_profile.request_weight_limit_per_minute
            )
        # Order Limit exhausted
        else:
            # still fail after order_limit_per_10_sec just resetted
            # means it is the minute that was exhausted
            if self.current_limit_state.order_limit_per_10_sec == 0:
                self.current_limit_state.order_limit_per_minute = (
                    self.limit_profile.order_limit_per_minute
                )
            self.current_limit_state.order_limit_per_10_sec = (
                self.limit_profile.order_limit_per_10_sec
            )

    def __repr__(self) -> str:
        retry_message = ""
        if self.retry_after > self.get_synced_time_ms():
            retry_message = f" [RETRYING_AFTER: {self.retry_after}]"
        return (
            f"<RateLimitState "
            f"Weight(1m): {self.current_limit_state.request_weight_limit_per_minute}, "
            f"Orders(1m): {self.current_limit_state.order_limit_per_minute}, "
            f"Orders(10s): {self.current_limit_state.order_limit_per_10_sec}"
            f"{f'Retry-After: {self.retry_after}' if self.retry_after > self.get_synced_time_ms() else ''}"
            f"{self._waiters_repr()}"
            f"{retry_message}>"
        )


class BybitRateLimiter(RateLimiter):
    def __init__(self, config: ConfigManager):
        if not isinstance(config.exchange, BybitLinearClient):
            raise Exception("Exchange mismatch with rate limiter")

        super().__init__(config)
        limit_profile = BybitLimitProfile.with_buffer(
            buffer_pct=Decimal("1.0") - self.soft_limit_percentage
        )

        self.limit_profile = limit_profile
        self.current_limit_state: Dict[BybitRateLimitPool, BybitLimitState] = {
            limit_pool: BybitLimitState(timestamps=deque())
            for limit_pool in BybitRateLimitPool
        }
        self.exchange = config.exchange
        self.exchange_time_offset = 0

        # to block all calls until
        self.retry_after = 0

    async def init(self):
        await self.on_resync_time()

    @asynccontextmanager
    async def guard(self, endpoint: Endpoints) -> AsyncGenerator[None, None]:
        if not self.check_limits(endpoint=endpoint):
            raise LocalRateLimitError(f"Failed due to rate limits {self}")

        self.record_usage(endpoint=endpoint)
        try:
            yield
        except Exception as e:
            self._handle_call_error(e)
            raise e

    def _handle_call_error(self, e: Exception) -> None:
        if isinstance(e, BybitError) and (e.http_status == 403 or e.retCode == 10006):
            self.local_cache_error(e.response_headers if e.response_headers else {})

    def _pool_key(self, endpoint: Endpoints) -> Any:
        # Endpoints with a dedicated UID pool queue on it; the rest share the
        # IP-global pool that every call also counts against
        return BYBIT_FUTURES_COSTS.get(endpoint, BybitRateLimitPool.IP_GLOBAL)

    def _next_free_delay(self, endpoint: Endpoints) -> float:
        # Rolling 1s windows: a full pool frees when its oldest timestamp ages
        # out. Wait on the latest-freeing of the pools this call touches.
        now = self.get_synced_time_ms()
        interval_ms = self.limit_profile.interval * 1000
        pools = [BybitRateLimitPool.IP_GLOBAL]
        cost_info = BYBIT_FUTURES_COSTS.get(endpoint)
        if cost_info:
            pools.append(cost_info)
        delay_ms = 0
        for pool in pools:
            state = self.current_limit_state[pool]
            limit = self.limit_profile.limits[pool]
            if state.timestamps and len(state.timestamps) + 1 > limit:
                delay_ms = max(delay_ms, state.timestamps[0] + interval_ms - now)
        return max(delay_ms, 0) / 1000.0

    async def on_resync_time(self):
        endpoint = Endpoints.GET_SERVER_TIME
        try:
            async with self.guard(endpoint=endpoint):
                exchange_time = await self.exchange.get_server_time()
                current_time = int(time.time() * 1000)
                self.exchange_time_offset = current_time - exchange_time
        except Exception as e:
            logger.warning(f"[ON_RESYNC_TIME] {e}")

    def get_synced_time_ms(self) -> int:
        current_time = int(time.time() * 1000)
        return current_time - self.exchange_time_offset

    def reset_limits(self):
        """
        Will reset limits based on rolling time window

        Source: https://bybit-exchange.github.io/docs/v5/rate-limit
        """
        synced_time = self.get_synced_time_ms()
        for state in self.current_limit_state.values():
            # more than a second has passed
            while state.timestamps and synced_time - state.timestamps[0] >= 1000:
                state.timestamps.popleft()

    def check_limits(self, endpoint: Endpoints, **kwargs) -> bool:
        """
        Whether a guard() call may proceed: blocked while retry_after is
        active, and yields to anything queued in reserve() for this pool.
        POST_ORDER/ Order Creation is exempted from this rule
        """
        # Absulute condition if retry after is active will not do anything
        if self.retry_after >= self.get_synced_time_ms():
            return False
        # Yield to callers waiting in reserve() so reserved calls take priority
        if self._waiters.get(self._pool_key(endpoint), 0) > 0:
            return False
        return self._has_capacity(endpoint, **kwargs)

    def _has_capacity(self, endpoint: Endpoints, **kwargs) -> bool:
        self.reset_limits()
        try:
            cost_info = BYBIT_FUTURES_COSTS.get(endpoint)

            # IP rate limit takes priority
            ip_limit = BybitRateLimitPool.IP_GLOBAL
            if (
                len(self.current_limit_state[ip_limit].timestamps) + 1
                > self.limit_profile.limits[ip_limit]
            ):
                logger.warning(
                    f"[CHECK_LIMITS] {ip_limit.name} reached its limit\n{self.limit_profile.limits[ip_limit]} <= {len(self.current_limit_state[ip_limit].timestamps) + 1}"
                )
                return False

            if (
                cost_info
                and len(self.current_limit_state[cost_info].timestamps) + 1
                > self.limit_profile.limits[cost_info]
            ):
                logger.warning(
                    f"[CHECK_LIMITS] {cost_info.name} reached its limit\n{self.limit_profile.limits[cost_info]} <= {len(self.current_limit_state[cost_info].timestamps) + 1}"
                )
                return False
            # Passed all checks
            return True
        except Exception as e:
            logger.error(f"Failed to check limits due to, {e}")
            return False

    def record_usage(self, endpoint: Endpoints):
        """
        Update limit values after successful endpoint request
        """
        current_time = self.get_synced_time_ms()
        ip_limit = BybitRateLimitPool.IP_GLOBAL
        cost_info = BYBIT_FUTURES_COSTS.get(endpoint)
        self.current_limit_state[ip_limit].timestamps.append(current_time)
        if cost_info:
            self.current_limit_state[cost_info].timestamps.append(current_time)

    def local_cache_error(self, headers: dict[str, Any]):
        """
        Use when bybit api return 403 (IP rate limit exhausted) and retCode 10006 endpoint exhausted

        Will block any subsequent request based on retry after value from headers (if endpoint)
        Will block any subsequent request for 10 minutes (if IP)
        """

        logger.warning(f"[LOCAL CACHE ERROR] HEADERS {_redact_headers(headers)}")
        # UID ENDPOINT EXHAUSTED
        if "X-Bapi-Limit-Reset-Timestamp" in headers.keys():
            self.retry_after = (
                int(headers["X-Bapi-Limit-Reset-Timestamp"]) + 50  # safety buffer
            )
        # IP RATE LIMIT EXHAUSTED
        else:
            self.retry_after = (
                self.get_synced_time_ms()
                + (10 * 60 * 1000)  # 10 minute
                + 1000  # safety buffer
            )

    def __repr__(self) -> str:
        retry_message = ""
        if self.retry_after > self.get_synced_time_ms():
            retry_message = f" [RETRYING_AFTER: {self.retry_after}]"
        state_dump = ", ".join(
            [
                f"{k.name}: Usage: {len(v.timestamps)}"
                for k, v in self.current_limit_state.items()
            ]
        )
        return (
            f"<RateLimitState "
            f"{state_dump}"
            f"{f'Retry-After: {self.retry_after}' if self.retry_after > self.get_synced_time_ms() else ''}"
            f"{self._waiters_repr()}"
            f"{retry_message}>"
        )

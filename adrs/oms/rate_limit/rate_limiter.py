import asyncio

from abc import abstractmethod, ABC
from collections import deque
from contextlib import asynccontextmanager
from decimal import Decimal
import time

import logging
from typing import Any, Dict, AsyncGenerator

from adrs.oms.config import ConfigManager
from adrs.oms.rate_limit.error_policy import (
    BINANCE_ORDER_RATE_LIMIT_CODES,
    binance_banned_until_ms,
    is_binance_rate_limit_error,
)
from adrs.oms.rate_limit.exchange_limit_profiles import (
    BinanceLimitProfile,
    BINANCE_DEPTH_WEIGHTS,
    BINANCE_FUTURES_COSTS,
    BybitLimitState,
    DYNAMIC_WEIGHT,
    Endpoints,
    OMS_DEPTH_LIMIT,
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


def _get_header(headers: dict[str, Any], name: str) -> Any | None:
    """Case-insensitive header lookup; exchange clients don't guarantee casing."""
    if name in headers:
        return headers[name]
    lname = name.lower()
    for k, v in headers.items():
        if k.lower() == lname:
            return v
    return None


def _int_header(headers: dict[str, Any], name: str) -> int | None:
    """Case-insensitive header lookup coerced to int; None if absent or junk."""
    raw = _get_header(headers, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"[RATE_LIMIT] Ignoring unparseable {name} header: {raw!r}")
        return None


# A parsed cooldown deadline further out than this is treated as garbage rather
# than trusted, so one malformed header cannot park the OMS indefinitely.
#
# Sized off what Binance actually hands out, not off what feels reasonable: a
# repeat offender's IP ban escalates to three days, and a day-long horizon threw
# every one of those away as implausible. With no deadline left, _arm_cooldown
# fell back to the 65s blind cooldown, after which the OMS polled straight
# through the remaining ban and renewed it — the exact production failure this
# code exists to prevent, reserved for its most expensive case. Four days leaves
# headroom over the longest real ban while still catching absurd values.
_MAX_TRUSTED_COOLDOWN_MS = 4 * 24 * 60 * 60 * 1000
# Cooldown when the exchange says "too many requests" but supplies neither a
# Retry-After header nor a ban deadline. One weight window plus a margin: long
# enough to stop the bleeding, short enough not to strand the OMS.
_BLIND_COOLDOWN_MS = 65_000
_COOLDOWN_SAFETY_MS = 1_000

# Only order counts are reconciled from headers. x-mbx-used-weight-1m has no
# prefix constant on purpose: it is deliberately never read, because it does not
# report the egress IP's spend. See the REQUEST_WEIGHT branch in
# BinanceRateLimiter.__init__ for the measurements.
_BINANCE_ORDER_COUNT_HEADER_PREFIX = "x-mbx-order-count"

# Binance suffixes each usage header with the interval of the limiter it reports:
# X-MBX-USED-WEIGHT-(intervalNum)(intervalLetter). The letters are the first
# character of the exchangeInfo `interval` enum, so 1/MINUTE -> "1M" and
# 10/SECOND -> "10S".
_BINANCE_INTERVAL_LETTERS = {
    "SECOND": "S",
    "MINUTE": "M",
    "HOUR": "H",
    "DAY": "D",
}


def _binance_usage_header(prefix: str, interval: str, interval_num: int) -> str | None:
    """
    Usage header reporting the limiter at `interval_num` x `interval`, lowercased
    to match what reqwest hands back. None when the interval is one we have no
    letter for, so the caller can say so rather than build a header name that
    will never match.
    """
    letter = _BINANCE_INTERVAL_LETTERS.get(str(interval).upper())
    if letter is None:
        return None
    return f"{prefix}-{interval_num}{letter}".lower()


class LocalRateLimitError(Exception):
    """
    Raised when the local rate limiter blocks a request
    before it is sent to the exchange.
    """

    def __init__(self, message="Local rate limit exceeded"):
        self.message = message
        super().__init__(self.message)


# This limiter is per process, while an exchange's heaviest budgets are metered
# per source IP and every tenant on a shard shares one NAT address. Two things
# stand in for the cross-process lock this used to ask for: IP-scoped budgets are
# divided by Config.tenants_per_egress_ip, and on Binance the IP-wide spend is
# read off x-mbx-used-weight-1m, which already reflects what co-tenants have
# spent. They are deliberately tracked as two separate counters against two
# separate ceilings — this process's own tally against its divided share, and
# the IP-wide figure against the undivided IP budget. Folding one into the other
# compares an IP-wide count against a per-tenant budget and caps the whole shard
# at one tenant's share. A real shared token bucket would still beat both, since
# the static split cannot lend unused budget between quiet and busy tenants.
class RateLimiter(ABC):
    # epoch ms until which all calls are locally blocked after a rate-limit
    # error; see _arm_cooldown
    retry_after: int = 0

    def __init__(
        self,
        config: ConfigManager,
    ):
        self.config = config
        self.soft_limit_percentage = config.config.soft_limit_percent
        # Processes sharing this egress IP. IP-scoped budgets are split this
        # many ways; account-scoped ones are not. See Config for why.
        self.tenants_per_egress_ip = config.config.tenants_per_egress_ip
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
        else:
            self._on_call_success(endpoint)

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
    def record_usage(self, endpoint: Endpoints, **kwargs):
        """
        To record usage once endpoint request was successful
        """
        ...

    @abstractmethod
    def local_cache_error(self, headers: dict[str, Any], **kwargs: Any) -> None:
        """
        To record usage once endpoint request was failure, local store has been desyncronized

        Subclasses may take extra keyword-only detail from the error (Binance
        needs the body code and the ban deadline, which are not in the headers).
        """
        ...

    def _on_call_success(self, endpoint: Endpoints) -> None:
        """
        Hook for exchanges that can reconcile local state from a successful
        response (e.g. Bybit's rate-limit headers). No-op by default.
        """

    def _arm_cooldown(
        self,
        *deadlines_ms: int | None,
        reason: str,
        blind_cooldown_ms: int = _BLIND_COOLDOWN_MS,
    ) -> None:
        """
        Block every call until the latest of `deadlines_ms`.

        Rules that apply whichever exchange armed it:

        - An existing cooldown is never shortened. A later response can arrive
          carrying a nearer deadline (a 429's Retry-After after a 418's ban
          deadline, say) and must not release the block early.
        - A deadline beyond _MAX_TRUSTED_COOLDOWN_MS (four days, comfortably
          past Binance's longest real ban) is discarded as garbage rather than
          trusted, so one malformed header cannot park the OMS indefinitely.
        - If no usable deadline survives, back off for `blind_cooldown_ms`
          anyway. The exchange has said it is over budget, and continuing to
          send is what escalates a throttle into a ban. Callers whose pool
          refills quickly (a Bybit UID pool, on a one-second window) pass a
          correspondingly short fallback.
        """
        now = self.get_synced_time_ms()
        horizon = now + _MAX_TRUSTED_COOLDOWN_MS
        usable = [d for d in deadlines_ms if d is not None and now < d <= horizon]
        for rejected in (d for d in deadlines_ms if d is not None and d > horizon):
            logger.warning(
                f"[RATE_LIMIT] Ignoring implausible cooldown deadline {rejected} "
                f"({(rejected - now) / 3_600_000:.1f}h out)"
            )
        if not usable:
            usable = [now + blind_cooldown_ms]
            logger.warning(
                f"[RATE_LIMIT] {reason} carried no usable deadline, "
                f"backing off {blind_cooldown_ms / 1000:.1f}s"
            )
        self.retry_after = max(self.retry_after, *usable)
        logger.warning(
            f"[RATE_LIMIT] {reason}: cooling down for "
            f"{(self.retry_after - now) / 1000:.1f}s {self}"
        )

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

        # Header name -> the limit_profile/current_limit_state field it reports
        # usage for. Built from the rules below rather than hard-coded, because
        # Binance names each header after the limiter it belongs to: "Every
        # request will contain X-MBX-USED-WEIGHT-(intervalNum)(intervalLetter)
        # ... for all request rate limiters defined". Today USDⓈ-M advertises
        # REQUEST_WEIGHT 1/MINUTE and ORDERS at both 1/MINUTE and 10/SECOND, so
        # this resolves to x-mbx-used-weight-1m, x-mbx-order-count-1m and
        # x-mbx-order-count-10s -- but deriving it means the limit and the header
        # reporting its usage can never disagree about the interval.
        usage_headers: dict[str, str] = {}

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
                # REQUEST_WEIGHT is metered per source IP, so it is the one
                # budget that has to be shared with every co-tenant behind the
                # same NAT gateway. Floor, and never below 1, so a
                # generously-sized shard still leaves each tenant able to send.
                limit_profile.request_weight_limit_per_minute = max(
                    1,
                    int(
                        limit * self.soft_limit_percentage / self.tenants_per_egress_ip
                    ),
                )
                # Deliberately no header reconciliation for weight. Measured on a
                # live shard: x-mbx-used-weight-1m does not report this egress
                # IP's spend. Public endpoints answer with our own small figure
                # (/fapi/v1/time and /fapi/v1/exchangeInfo returned 8 and 9),
                # while signed account endpoints answer from some larger shared
                # pool in the same breath (/fapi/v3/positionRisk 1389,
                # /fapi/v2/balance 2293), and a /fapi/v1/time immediately after
                # dropped back to 10. The signed figures also swing about
                # tenfold between samples while our own traffic is unchanged.
                # Presence of an API key makes no difference -- one endpoint
                # probed with and without it returned 4/5, 6/7, 2/3, 4/5 -- so
                # the split is by endpoint family, not by authentication.
                #
                # Since the OMS reads signed endpoints constantly, adopting that
                # number meant a max() over unrelated readings, and it was
                # gating request admission: live readings sawtoothed to 3337 and
                # a direct probe hit 4433 against a 4800 refusal threshold. One
                # more step and the OMS would have stopped trading on a quantity
                # that measures nothing about us. Binance's own 429/418 plus
                # Retry-After is the authoritative "you exceeded" signal, and it
                # is already handled in note_error(); the local tally below is
                # what this process controls. Nothing here needs the header.
                continue
            elif limit_type == "ORDERS" and interval == "MINUTE" and interval_num == 1:
                # Order counts are metered per account, so each tenant's own API
                # key gets the whole budget; dividing would throttle for no
                # reason.
                limit_profile.order_limit_per_minute = int(
                    limit * self.soft_limit_percentage
                )
                field = "order_limit_per_minute"
                prefix = _BINANCE_ORDER_COUNT_HEADER_PREFIX
            elif limit_type == "ORDERS" and interval == "SECOND" and interval_num == 10:
                limit_profile.order_limit_per_10_sec = int(
                    limit * self.soft_limit_percentage
                )
                field = "order_limit_per_10_sec"
                prefix = _BINANCE_ORDER_COUNT_HEADER_PREFIX
            else:
                logger.warning(f"Unknown rate limit {rule}")
                continue

            header = _binance_usage_header(prefix, interval, interval_num)
            if header is None:
                logger.error(
                    f"[BinanceRateLimiter] No interval letter known for {interval!r}; "
                    f"{field} will be tracked from local estimates only"
                )
            else:
                usage_headers[header] = field

        # A REQUEST_WEIGHT rule we do not understand leaves the budget at 0, and
        # _has_capacity then refuses every single call — an OMS that runs, logs a
        # warning, and silently never trades. Fail at startup instead, where the
        # crash names the cause.
        if limit_profile.request_weight_limit_per_minute <= 0:
            raise Exception(
                "Binance exchangeInfo declared no REQUEST_WEIGHT limit this "
                f"limiter understands, so every request would be refused. "
                f"rateLimits={rateLimits}"
            )

        self.limit_profile = limit_profile
        self._usage_headers = usage_headers
        logger.info(
            f"[BinanceRateLimiter] Weight(1m) budget "
            f"{limit_profile.request_weight_limit_per_minute} "
            f"(this tenant's share, split "
            f"{self.tenants_per_egress_ip} way(s); tracked locally, because "
            f"x-mbx-used-weight-1m does not report this IP's spend), "
            f"Orders(1m) {limit_profile.order_limit_per_minute}, "
            f"Orders(10s) {limit_profile.order_limit_per_10_sec} (account-scoped). "
            f"Reconciling usage from {sorted(usage_headers)}"
        )
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
        else:
            self._on_call_success(endpoint)

    def _handle_call_error(self, e: Exception) -> None:
        if not isinstance(e, BinanceError) or not is_binance_rate_limit_error(e):
            return
        self.local_cache_error(
            e.response_headers if e.response_headers else {},
            code=e.code,
            banned_until_ms=binance_banned_until_ms(str(e)),
        )

    def _on_call_success(self, endpoint: Endpoints) -> None:
        headers = getattr(self.exchange, "last_response_headers", None)
        if headers:
            self._reconcile_from_headers(headers)

    def _reconcile_from_headers(self, headers: dict[str, Any]) -> None:
        """
        Adopt Binance's own usage counters from the response just returned.

        Order counts only. The weight header is never registered in
        _usage_headers — see the REQUEST_WEIGHT branch in __init__ for the
        measurements — so nothing here reads it and weight stays on the local
        tally that record_usage() maintains.

        Order counts are metered per account, so the header and the local tally
        really are two readings of one number and the header simply wins.

        Takes the max rather than overwriting: record_usage() charges a call
        before it is sent, while the header only reflects requests Binance has
        already counted, so overwriting would refund every in-flight
        reservation.

        Source: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
        """
        state = self.current_limit_state
        # reset_limits() first, or a counter this rolls forward can be zeroed
        # immediately afterwards by a window boundary that has already passed.
        window_before = self.last_reset_1m_timestamp
        self.reset_limits()
        if self.last_reset_1m_timestamp != window_before:
            # Our window rolled while this response was in flight, so the header
            # describes Binance's *previous* minute. Adopting a nearly-full
            # previous-minute count into a freshly zeroed window would block
            # weighted calls for up to a full minute for no reason. Skip it; the
            # next response re-establishes the figure milliseconds later.
            return

        # Header names come from the rateLimits rules parsed in __init__, so each
        # one lines up with the budget it reports usage against. An absent header
        # just means no reading this time (order counts only ride on order
        # responses), and the local estimate stands.
        for header, field in self._usage_headers.items():
            reported = _int_header(headers, header)
            if reported is None:
                continue
            setattr(state, field, max(getattr(state, field), reported))

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

    def find_cost_info(self, endpoint: Endpoints, **kwargs) -> tuple[int, int]:
        """
        (weight, orders) charged for one call to `endpoint`.

        Endpoints marked DYNAMIC_WEIGHT are resolved here from the request
        parameters, so the capacity check and the usage record can never
        disagree about what a call costs — before this, only the capacity check
        resolved them, and record_usage() would have added the raw -1 marker to
        the counter, crediting weight back on every call.
        """
        cost_info = BINANCE_FUTURES_COSTS.get(endpoint)
        if cost_info is None:
            logger.error(
                f"[CHECK_LIMITS] Severe error no cost info is defined for this endpoint: {endpoint.name}"
            )
            raise KeyError(f"Endpoint doesn't exist, {endpoint.name}")

        weight_cost = cost_info["weight"]
        if weight_cost == DYNAMIC_WEIGHT:
            weight_cost = self._resolve_dynamic_weight(endpoint, **kwargs)
        return (weight_cost, cost_info["orders"])

    def _resolve_dynamic_weight(self, endpoint: Endpoints, **kwargs) -> int:
        """
        Weight for an endpoint whose cost depends on its parameters.

        guard()/reserve() do not thread per-call parameters through, so in
        practice depth reads resolve to OMS_DEPTH_LIMIT — the depth
        OrderUtils.get_order_book actually requests, which is the OMS's only
        depth call site. Charging anything else here undercounts (a cheaper
        tier) or wastes budget (a heavier one); an explicit limit/depth kwarg,
        if one is ever threaded, still wins.
        """
        if endpoint == Endpoints.GET_ORDERBOOK_SNAPSHOT:
            limit = kwargs.get("limit", kwargs.get("depth"))
            return get_depth_weight(limit if limit is not None else OMS_DEPTH_LIMIT)
        logger.error(
            f"[CHECK_LIMITS] {endpoint.name} is marked dynamic but has no weight "
            f"rule; charging the heaviest known cost"
        )
        return max(get_depth_weight(BINANCE_DEPTH_WEIGHTS[-1][0]), 1)

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
            (weight_cost, order_cost) = self.find_cost_info(endpoint=endpoint, **kwargs)
            # Check in decending order by timescale
            # Checking REQUEST_WEIGHT
            # One ceiling: this process's own spend against this tenant's share.
            # There is deliberately no second, IP-wide check — the header it
            # would have to come from does not report this IP's spend (see the
            # REQUEST_WEIGHT branch in __init__), and refusing on it stalled
            # trading on a quantity unrelated to our traffic. Co-tenant pressure
            # is handled by dividing the budget, and real exhaustion by
            # Binance's 429/418 and the cooldown in note_error().
            current_weight = self.current_limit_state.request_weight_limit_per_minute
            max_weight = self.limit_profile.request_weight_limit_per_minute
            if weight_cost != 0 and max_weight <= (current_weight + weight_cost):
                logger.warning(
                    f"[CHECK_LIMITS] REQUEST_WEIGHT 1m reached this tenant's share\n{max_weight} <= {current_weight} + {weight_cost}"
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

    def record_usage(self, endpoint: Endpoints, **kwargs):
        """
        Update limit values after successful endpoint request
        """
        try:
            (weight_cost, order_cost) = self.find_cost_info(endpoint=endpoint, **kwargs)
            self.current_limit_state.order_limit_per_10_sec += order_cost
            self.current_limit_state.order_limit_per_minute += order_cost
            self.current_limit_state.request_weight_limit_per_minute += weight_cost
        except Exception as e:
            logger.error(f"Failed to record usage due to {e}")

    def local_cache_error(
        self,
        headers: dict[str, Any],
        *,
        code: int | None = None,
        banned_until_ms: int | None = None,
        **_: Any,
    ) -> None:
        """
        Use when Binance reports a rate-limit breach: HTTP 429 (budget
        exhausted) or 418 (IP banned), both carrying body code -1003, or -1015
        for an order-rate breach.

        Whichever it is, the local counters have provably disagreed with the
        exchange, so the budget the breach came from is marked spent.
        reset_limits() clears it again at the next window boundary.

        An order-rate breach is scoped to the account's order budget, so it only
        blocks placement and leaves reads working. Anything else is weight- or
        IP-scoped, and continuing to send during it is exactly what makes
        Binance extend a ban, so every endpoint is blocked until it lapses.
        """

        logger.warning(f"[LOCAL CACHE ERROR] HEADERS {_redact_headers(headers)}")

        if code in BINANCE_ORDER_RATE_LIMIT_CODES:
            self.current_limit_state.order_limit_per_10_sec = (
                self.limit_profile.order_limit_per_10_sec
            )
            self.current_limit_state.order_limit_per_minute = (
                self.limit_profile.order_limit_per_minute
            )
            logger.warning(
                f"[LOCAL CACHE ERROR] order rate exhausted (code {code}), "
                f"orders blocked until the next window {self}"
            )
            return

        self.current_limit_state.request_weight_limit_per_minute = (
            self.limit_profile.request_weight_limit_per_minute
        )

        # Retry-After is seconds; the ban deadline parsed out of the -1003
        # message is already epoch ms. A 418 often carries only the latter.
        retry_after_sec = _int_header(headers, "Retry-After")
        self._arm_cooldown(
            banned_until_ms + _COOLDOWN_SAFETY_MS
            if banned_until_ms is not None
            else None,
            self.get_synced_time_ms() + retry_after_sec * 1000 + _COOLDOWN_SAFETY_MS
            if retry_after_sec is not None
            else None,
            reason=f"Binance rate limit (code {code})",
        )

    def __repr__(self) -> str:
        retry_message = ""
        if self.retry_after > self.get_synced_time_ms():
            retry_message = f" [RETRYING_AFTER: {self.retry_after}]"
        return (
            f"<RateLimitState "
            f"Weight(1m): {self.current_limit_state.request_weight_limit_per_minute}"
            f"/{self.limit_profile.request_weight_limit_per_minute}, "
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
            buffer_pct=Decimal("1.0") - self.soft_limit_percentage,
            tenants_per_egress_ip=self.tenants_per_egress_ip,
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
        else:
            self._on_call_success(endpoint)

    def _handle_call_error(self, e: Exception) -> None:
        if isinstance(e, BybitError) and (e.http_status == 403 or e.retCode == 10006):
            self.local_cache_error(e.response_headers if e.response_headers else {})

    def _on_call_success(self, endpoint: Endpoints) -> None:
        cost_info = BYBIT_FUTURES_COSTS.get(endpoint)
        if cost_info is None:
            return
        headers = getattr(self.exchange, "last_response_headers", None)
        if headers:
            self._reconcile_uid_pool(cost_info, headers)

    def _reconcile_uid_pool(
        self, pool: BybitRateLimitPool, headers: dict[str, Any]
    ) -> None:
        """
        Overwrite this UID pool's tracked capacity with the exchange's own
        rate-limit headers from the response that was just returned for it.

        Source: https://bybit-exchange.github.io/docs/v5/rate-limit
        """
        limit = _get_header(headers, "X-Bapi-Limit")
        status = _get_header(headers, "X-Bapi-Limit-Status")
        if limit is None or status is None:
            return
        reset_ts = _get_header(headers, "X-Bapi-Limit-Reset-Timestamp")
        state = self.current_limit_state[pool]
        state.limit = int(limit)
        state.remaining = int(status)
        if reset_ts is not None:
            state.reset_ts = int(reset_ts)

    def _pool_key(self, endpoint: Endpoints) -> Any:
        # Endpoints with a dedicated UID pool queue on it; the rest share the
        # IP-global pool that every call also counts against
        return BYBIT_FUTURES_COSTS.get(endpoint, BybitRateLimitPool.IP_GLOBAL)

    def _next_free_delay(self, endpoint: Endpoints) -> float:
        now = self.get_synced_time_ms()
        interval_ms = self.limit_profile.interval * 1000
        delay_ms = 0

        # IP_GLOBAL: rolling 1s window, no exchange header to draw on. Frees
        # when its oldest timestamp ages out.
        ip_limit = BybitRateLimitPool.IP_GLOBAL
        ip_state = self.current_limit_state[ip_limit]
        ip_pool_limit = self.limit_profile.limits[ip_limit]
        if ip_state.timestamps and len(ip_state.timestamps) + 1 > ip_pool_limit:
            delay_ms = max(delay_ms, ip_state.timestamps[0] + interval_ms - now)

        # UID_* pool: exchange reports its own reset time once exhausted.
        cost_info = BYBIT_FUTURES_COSTS.get(endpoint)
        if cost_info:
            effective_remaining, _ = self._uid_pool_snapshot(cost_info, now)
            if effective_remaining < 1:
                delay_ms = max(
                    delay_ms, self.current_limit_state[cost_info].reset_ts - now
                )

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
        Rolls the IP_GLOBAL window. UID_* pools have no local window to roll
        any more; their capacity comes from the exchange's own headers (see
        _uid_pool_snapshot), refreshed lazily once their reset_ts passes.

        Source: https://bybit-exchange.github.io/docs/v5/rate-limit
        """
        synced_time = self.get_synced_time_ms()
        state = self.current_limit_state[BybitRateLimitPool.IP_GLOBAL]
        while state.timestamps and synced_time - state.timestamps[0] >= 1000:
            state.timestamps.popleft()

    def _uid_pool_snapshot(self, pool: BybitRateLimitPool, now: int) -> tuple[int, int]:
        """
        (effective_remaining, limit) for a UID_* pool as of `now`. Before the
        first response for this pool has been seen, capacity is assumed full
        at the hard-coded buffered limit. reset_ts is 0 until a real header
        sets it, so `now >= reset_ts` must not fire on that default — it
        would wipe out every optimistic decrement made between responses.
        """
        state = self.current_limit_state[pool]
        limit = (
            state.limit if state.limit is not None else self.limit_profile.limits[pool]
        )
        if state.remaining is None:
            return limit, limit
        if state.reset_ts and now >= state.reset_ts:
            return limit, limit
        return state.remaining, limit

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
            now = self.get_synced_time_ms()

            # IP rate limit takes priority; Bybit exposes no header for it,
            # so it stays on the local rolling window.
            ip_limit = BybitRateLimitPool.IP_GLOBAL
            if (
                len(self.current_limit_state[ip_limit].timestamps) + 1
                > self.limit_profile.limits[ip_limit]
            ):
                logger.warning(
                    f"[CHECK_LIMITS] {ip_limit.name} reached its limit\n{self.limit_profile.limits[ip_limit]} <= {len(self.current_limit_state[ip_limit].timestamps) + 1}"
                )
                return False

            if cost_info:
                effective_remaining, limit = self._uid_pool_snapshot(cost_info, now)
                if effective_remaining < 1:
                    logger.warning(
                        f"[CHECK_LIMITS] {cost_info.name} reached its limit\n0 remaining of {limit} (X-Bapi-Limit-Status)"
                    )
                    return False
            # Passed all checks
            return True
        except Exception as e:
            logger.error(f"Failed to check limits due to, {e}")
            return False

    def record_usage(self, endpoint: Endpoints, **kwargs):
        """
        Update limit values after successful endpoint request
        """
        current_time = self.get_synced_time_ms()
        ip_limit = BybitRateLimitPool.IP_GLOBAL
        self.current_limit_state[ip_limit].timestamps.append(current_time)

        cost_info = BYBIT_FUTURES_COSTS.get(endpoint)
        if cost_info:
            # Optimistic pre-call decrement so concurrent callers on the same
            # pool don't all admit off one stale header snapshot;
            # _on_call_success() overwrites this with the exchange's own
            # count once the response for this call lands.
            state = self.current_limit_state[cost_info]
            effective_remaining, _ = self._uid_pool_snapshot(cost_info, current_time)
            state.remaining = max(0, effective_remaining - 1)

    def local_cache_error(self, headers: dict[str, Any], **_: Any) -> None:
        """
        Use when bybit api return 403 (IP rate limit exhausted) and retCode 10006 endpoint exhausted

        A UID pool breach blocks until that pool's own reset timestamp, which is
        usually well under a second away. Only an IP breach — which Bybit
        reports with no reset header at all — falls back to the 10 minute block,
        matching the ban it hands out.

        The header lookup has to be case-insensitive: reqwest, which cybotrade's
        HTTP client is built on, lowercases header names, so matching
        "X-Bapi-Limit-Reset-Timestamp" exactly never fired and every UID breach
        took the IP branch and blocked all trading for 10 minutes.
        """

        logger.warning(f"[LOCAL CACHE ERROR] HEADERS {_redact_headers(headers)}")
        uid_pool_reset_ms = _int_header(headers, "X-Bapi-Limit-Reset-Timestamp")
        if uid_pool_reset_ms is not None:
            # UID ENDPOINT EXHAUSTED
            self._arm_cooldown(
                uid_pool_reset_ms + 50,  # safety buffer
                reason="Bybit UID pool exhausted",
                # The pool rolls on a one-second window, so a reset timestamp
                # that has already passed needs no more than that much backoff.
                blind_cooldown_ms=self.limit_profile.interval * 1000,
            )
            return
        # IP RATE LIMIT EXHAUSTED
        self._arm_cooldown(
            self.get_synced_time_ms() + (10 * 60 * 1000) + _COOLDOWN_SAFETY_MS,
            reason="Bybit IP rate limit exhausted",
        )

    def __repr__(self) -> str:
        retry_message = ""
        if self.retry_after > self.get_synced_time_ms():
            retry_message = f" [RETRYING_AFTER: {self.retry_after}]"

        def _pool_repr(pool: BybitRateLimitPool, state: BybitLimitState) -> str:
            if pool is BybitRateLimitPool.IP_GLOBAL:
                return f"{pool.name}: Usage: {len(state.timestamps)}"
            return f"{pool.name}: Remaining: {state.remaining}/{state.limit}"

        state_dump = ", ".join(
            _pool_repr(k, v) for k, v in self.current_limit_state.items()
        )
        return (
            f"<RateLimitState "
            f"{state_dump}"
            f"{f'Retry-After: {self.retry_after}' if self.retry_after > self.get_synced_time_ms() else ''}"
            f"{self._waiters_repr()}"
            f"{retry_message}>"
        )

"""
Rate limiting for Hyperliquid.

Hyperliquid rate-limits on two independent axes:

* an aggregated per-IP weight budget of 1200 per minute, which this limiter
  models exactly from a static cost table, and
* a cumulative per-address allowance of one request per USDC traded, with an
  initial buffer of 10,000 requests, which it does not.

The second is not a rate. It is a lifetime allowance that grows with traded
volume, so mirroring it would mean polling volume (paying weight for the
privilege) and acting on an estimate that is wrong either way: throttling when
there was headroom, or failing to protect when there was not. It is enormous
once trading -- $1M of volume earns 1M requests -- and its failure mode is a
clean, detectable throttle to one request per 10s. So it is handled reactively
via HyperliquidErrorPolicy, and counted here only to warn as the initial buffer
depletes. Counted for observability; never enforced as a limit we cannot see.
"""

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, TYPE_CHECKING

from cybotrade.exceptions import DeserializationError
from cybotrade.hyperliquid import HyperliquidClient, HyperliquidError

from adrs.oms.rate_limit.error_policy import is_hyperliquid_rate_limit_error
from adrs.oms.rate_limit.exchange_limit_profiles import (
    Endpoints,
    HYPERLIQUID_ADDRESS_ACTION_BUFFER,
    HYPERLIQUID_COSTS,
    HYPERLIQUID_IP_WEIGHT_PER_MINUTE,
    HYPERLIQUID_RATE_LIMIT_COOLDOWN_MS,
    HYPERLIQUID_WEIGHT_WINDOW_SEC,
    HyperliquidLimitProfile,
    HyperliquidRateLimitPool,
)
from adrs.oms.rate_limit.rate_limiter import LocalRateLimitError, RateLimiter

if TYPE_CHECKING:
    from adrs.oms.config import ConfigManager

logger = logging.getLogger(__name__)

# Fraction of the initial address buffer at which to start warning.
_ACTION_WARN_RATIO = 0.8

# Endpoints whose real cost is amortised over a window rather than charged per
# guard() call, because cybotrade serves the repeat calls from a client-side
# cache and they never reach Hyperliquid. See _effective_weight().
_AMORTISED_ENDPOINTS = frozenset({Endpoints.GET_SYMBOL_INFO})


class HyperliquidRateLimiter(RateLimiter):
    def __init__(self, config: "ConfigManager"):
        if not isinstance(config.exchange, HyperliquidClient):
            raise Exception("Exchange mismatch with rate limiter")
        super().__init__(config)

        # Weight is per IP, so a shard behind one NAT address must divide it;
        # the address allowance is account-scoped and is not divided. This is
        # the distinction the RateLimiter base class documents.
        ceiling = int(
            HYPERLIQUID_IP_WEIGHT_PER_MINUTE
            * self.soft_limit_percentage
            / self.tenants_per_egress_ip
        )
        self.limit_profile = HyperliquidLimitProfile(
            request_weight_limit_per_minute=ceiling,
            address_action_buffer=HYPERLIQUID_ADDRESS_ACTION_BUFFER,
        )
        # (epoch_ms, weight) charged, trimmed to the trailing window
        self.weight_window: deque[tuple[int, int]] = deque()
        self.address_actions = 0
        self._warned_actions = False
        # epoch_ms of the last full charge for each amortised endpoint; see
        # _effective_weight(). None until the first charge in a fresh window.
        self._amortised_charged_at: dict[Endpoints, int] = {}

        logger.info(
            f"[HYPERLIQUID_LIMITS] Weight(1m) {ceiling} (IP-scoped, split "
            f"{self.tenants_per_egress_ip} ways). Address actions counted "
            f"against a {HYPERLIQUID_ADDRESS_ACTION_BUFFER} buffer, not enforced."
        )

    # ---- clock ---------------------------------------------------------

    async def init(self):
        """
        Nothing to initialise.

        Hyperliquid publishes no server-time endpoint and nonces are
        client-generated, so there is no clock to sync against.
        """
        return None

    def get_synced_time_ms(self) -> int:
        """Local clock; see init() for why there is nothing to sync to."""
        return int(time.time() * 1000)

    async def on_resync_time(self):
        """No-op, for the same reason as init()."""
        return None

    # ---- window --------------------------------------------------------

    def current_weight(self) -> int:
        """Weight charged inside the trailing window, after trimming."""
        self._trim_window()
        return sum(weight for _, weight in self.weight_window)

    def _cost(self, endpoint: Endpoints) -> dict[str, int]:
        # KeyError rather than a zero default: an unpriced endpoint would be
        # charged nothing, and undercounting is what gets an IP banned.
        return HYPERLIQUID_COSTS[endpoint]

    def _effective_weight(self, endpoint: Endpoints) -> int:
        """
        Weight to charge for this call: the table price, except for an endpoint
        whose repeat calls cybotrade serves from its own cache.

        GET_SYMBOL_INFO is the one such endpoint. config.py's
        update_symbol_info() takes one guard per symbol, but cybotrade's
        HyperliquidClient caches `metaAndAssetCtxs` (METADATA_TTL, 5 minutes)
        and only the first call in a refresh reaches the exchange -- so a
        20-symbol sweep issues exactly one real weight-20 request while the
        per-symbol charge bills 400 of the minute's 1200. On the documented
        14-tenant shard the ceiling is 68/min, which the phantom charge can
        never fit: the sweep never completes, _symbol_info_refreshed_at is
        never stamped, and every tick retries it -- a livelock that starves
        PLACE_ORDER and CANCEL_ORDER. The Binance table documents the same
        phantom-charge failure.

        So the full weight is charged for the first call in a rolling window
        and 0 for the rest of that window, capping the charge at once per
        HYPERLIQUID_WEIGHT_WINDOW_SEC.

        This is deliberately conservative rather than exact. It is coupled to
        cybotrade's HyperliquidClient.METADATA_TTL: at 5 minutes, at most one
        real call occurs per 5 minutes while this charges up to one per minute
        -- a 5x overcharge, in the safe direction. If that TTL were ever
        shortened below HYPERLIQUID_WEIGHT_WINDOW_SEC (60s), two real calls
        could land inside one window and this would undercount, which is the
        direction that gets the shared egress IP banned. Re-derive the cap
        against METADATA_TTL before assuming it still holds.
        """
        weight = self._cost(endpoint)["weight"]
        if endpoint not in _AMORTISED_ENDPOINTS:
            return weight
        charged_at = self._amortised_charged_at.get(endpoint)
        if charged_at is None:
            return weight
        age_ms = self.get_synced_time_ms() - charged_at
        if 0 <= age_ms < HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000:
            return 0
        return weight

    def _trim_window(self):
        """Drop entries that have aged out of the trailing window."""
        cutoff = self.get_synced_time_ms() - HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000
        while self.weight_window and self.weight_window[0][0] <= cutoff:
            self.weight_window.popleft()

    def reset_limits(self):
        """
        Fully clear local state, as opposed to the passive time-based trimming
        _trim_window() does on every check. Public because the ABC requires it;
        nothing in adrs calls it today.

        WARNING: clearing the window discards weight the exchange is still
        counting. Hyperliquid's budget is a trailing 60s window on its side and
        there is no exchange-side signal to resync from (see init()), so a call
        here does not make that spend go away -- it only makes this limiter
        blind to it, and the next 60s of calls are admitted on top of weight
        already spent. That is an over-admission into a budget shared by every
        tenant on the egress IP. In particular it must not be called to recover
        from a connection event: a reconnect tells us nothing about what the
        exchange has already counted. Contrast _trim_window(), which only drops
        entries that have genuinely aged out and is safe on every check.
        """
        self.weight_window.clear()
        self.address_actions = 0
        self._warned_actions = False
        # Cleared too, so the next amortised call pays in full rather than
        # riding a window this reset just erased.
        self._amortised_charged_at.clear()

    def _pool_key(self, endpoint: Endpoints) -> Any:
        # One aggregated budget, so every call contends for the same pool.
        return HyperliquidRateLimitPool.IP_WEIGHT

    def _has_capacity(self, endpoint: Endpoints, **kwargs) -> bool:
        self._trim_window()
        # _effective_weight(), not the raw table price: record_usage() charges
        # the amortised figure, so projecting the raw one here would deny a
        # call that is about to cost nothing.
        projected = sum(w for _, w in self.weight_window) + self._effective_weight(
            endpoint
        )
        ceiling = self.limit_profile.request_weight_limit_per_minute
        if projected > ceiling:
            logger.warning(
                f"[CHECK_LIMITS] Hyperliquid weight window is full: "
                f"{projected} > {ceiling}"
            )
            return False
        return True

    def _next_free_delay(self, endpoint: Endpoints) -> float:
        """Seconds until the oldest entry ages out and frees weight."""
        self._trim_window()
        if self._has_capacity(endpoint):
            return 0.0
        if not self.weight_window:
            return 0.0
        oldest_ts = self.weight_window[0][0]
        expires_at = oldest_ts + HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000
        return max(0.0, (expires_at - self.get_synced_time_ms()) / 1000.0)

    def check_limits(self, endpoint: Endpoints, **kwargs) -> bool:
        """
        Whether a guard() call may proceed: blocked while retry_after is
        active, and yields to anything queued in reserve() for this pool.
        """
        # Absolute condition: while retry_after is active nothing proceeds. >=
        # rather than >, matching reserve() and both sibling limiters -- on the
        # exact millisecond of the deadline the hold still applies.
        if self.retry_after >= self.get_synced_time_ms():
            logger.warning(f"[CHECK_LIMITS] cooling down until {self.retry_after}")
            return False
        # Yield to callers waiting in reserve() so reserved calls take priority.
        # Without this the four reserve() sites lose their whole mechanism:
        # guard() callers keep taking the capacity a queued reserver is waiting
        # for, so delta-critical position and open-order reads are starved.
        if self._waiters.get(self._pool_key(endpoint), 0) > 0:
            return False
        return self._has_capacity(endpoint, **kwargs)

    def record_usage(self, endpoint: Endpoints, **kwargs):
        cost = self._cost(endpoint)
        now = self.get_synced_time_ms()
        weight = self._effective_weight(endpoint)
        if weight and endpoint in _AMORTISED_ENDPOINTS:
            # Opens the amortisation window for this endpoint; the rest of the
            # sweep rides it at 0. See _effective_weight().
            self._amortised_charged_at[endpoint] = now
        self.weight_window.append((now, weight))
        if cost["actions"]:
            self.address_actions += cost["actions"]
            self._maybe_warn_actions()

    def _maybe_warn_actions(self) -> None:
        buffer = self.limit_profile.address_action_buffer
        if self._warned_actions or self.address_actions < buffer * _ACTION_WARN_RATIO:
            return
        self._warned_actions = True
        logger.warning(
            f"[HYPERLIQUID_LIMITS] {self.address_actions} exchange actions sent "
            f"against an initial address buffer of {buffer}. The real allowance "
            f"grows with traded volume and is not observable here; if the "
            f"exchange starts throttling, that is why."
        )

    # ---- guard ---------------------------------------------------------

    @asynccontextmanager
    async def guard(self, endpoint: Endpoints) -> AsyncGenerator[None, None]:
        if not self.check_limits(endpoint=endpoint):
            raise LocalRateLimitError(f"Failed due to rate limits {self}")

        self.record_usage(endpoint=endpoint)
        try:
            async with asyncio.timeout(self.call_timeout_sec):
                yield
        except BaseException as e:
            self._handle_call_error(e, endpoint)
            raise e
        else:
            self._on_call_success(endpoint)

    # ---- error handling -----------------------------------------------

    def _handle_call_error(
        self, e: BaseException, endpoint: Endpoints | None = None
    ) -> None:
        """
        Arm the local cooldown when Hyperliquid is throttling us.

        Delegates the whole decision to local_cache_error(), which is the
        ABC-designated place for folding a failure into local state. Keeping
        one classifier means the two entry points cannot drift apart, and it
        stops local_cache_error() being dead code reachable from nowhere.

        Unlike the Bybit limiter there is no optimistic pre-call decrement to
        refund: the weight window records what was actually sent, and a request
        that failed still cost its weight at the exchange.
        """
        # Hyperliquid sends no rate-limit headers at all, so there is nothing
        # to pass; the exception itself carries every signal there is.
        if isinstance(e, Exception):
            self.local_cache_error({}, error=e)

    def local_cache_error(self, headers: dict[str, Any], **kwargs: Any) -> None:
        """
        Fold a rate-limit failure into local state.

        Hyperliquid returns no rate-limit headers at all -- an action failure
        arrives in the body of an HTTP 200 -- so `headers` is always empty here
        and the signal comes from the exception. Callers pass it as `error=...`,
        or as `message=...` when only the text is to hand.

        Classification is delegated to is_hyperliquid_rate_limit_error() rather
        than re-matched here, so the needles live in exactly one place; that
        also picks up the HTTP 429 case, which has no Hyperliquid body to match
        against.
        """
        error = kwargs.get("error")
        if error is None:
            message = str(kwargs.get("message") or "")
            if not message:
                return
            error = HyperliquidError(message)
        if not isinstance(error, Exception):
            return
        if isinstance(error, DeserializationError):
            # Deliberately broad. cybotrade's HyperliquidClient._post_info /
            # _post_exchange never look at the HTTP status: they hand the body
            # straight to json.loads, so a 429's non-JSON body (an edge/proxy
            # page, not a Hyperliquid payload) surfaces here as
            # DeserializationError and as nothing else. At this layer a
            # malformed response is therefore indistinguishable from a 429, so
            # it is treated as a suspected throttle. Holding briefly on a read
            # failure costs some latency; missing a real 429 means polling
            # through the throttle and renewing an IP ban that takes down every
            # tenant sharing the egress address.
            logger.warning(
                f"[HYPERLIQUID_LIMITS] undecodable response ({error}); treating "
                f"it as a suspected throttle, since a 429 body reaches us as "
                f"exactly this and nothing else."
            )
            self._arm_throttle_cooldown()
            return
        if is_hyperliquid_rate_limit_error(error):
            self._arm_throttle_cooldown()

    def _arm_throttle_cooldown(self) -> None:
        deadline = self.get_synced_time_ms() + HYPERLIQUID_RATE_LIMIT_COOLDOWN_MS
        self.retry_after = max(self.retry_after, deadline)
        logger.warning(
            f"[HYPERLIQUID_LIMITS] rate limited by the exchange; holding calls "
            f"until {self.retry_after}. Either axis can produce this: the "
            f"address allowance is cumulative and clears as traded volume "
            f"accrues, while an IP weight overrun clears as the window drains."
        )

    # ---- repr ----------------------------------------------------------

    def __repr__(self) -> str:
        retry = ""
        if self.retry_after > self.get_synced_time_ms():
            retry = f" [RETRYING_AFTER: {self.retry_after}]"
        return (
            f"HyperliquidRateLimiter(weight: {self.current_weight()}"
            f"/{self.limit_profile.request_weight_limit_per_minute}, "
            f"actions: {self.address_actions}"
            f"/{self.limit_profile.address_action_buffer}){retry}"
        )

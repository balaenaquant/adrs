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

from cybotrade.hyperliquid import HyperliquidClient

from adrs.oms.rate_limit.exchange_limit_profiles import (
    Endpoints,
    HYPERLIQUID_ADDRESS_ACTION_BUFFER,
    HYPERLIQUID_COSTS,
    HYPERLIQUID_IP_WEIGHT_PER_MINUTE,
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

    def _trim_window(self):
        """Drop entries that have aged out of the trailing window."""
        cutoff = self.get_synced_time_ms() - HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000
        while self.weight_window and self.weight_window[0][0] <= cutoff:
            self.weight_window.popleft()

    def reset_limits(self):
        """
        Fully clear local state, as opposed to the passive time-based trimming
        _trim_window() does on every check. There is no exchange-side signal
        to resync from (see init()), so this is only ever a hard local reset,
        e.g. on reconnect -- not part of the per-call capacity check.
        """
        self.weight_window.clear()
        self.address_actions = 0
        self._warned_actions = False

    def _pool_key(self, endpoint: Endpoints) -> Any:
        # One aggregated budget, so every call contends for the same pool.
        return HyperliquidRateLimitPool.IP_WEIGHT

    def _has_capacity(self, endpoint: Endpoints, **kwargs) -> bool:
        self._trim_window()
        projected = (
            sum(w for _, w in self.weight_window) + self._cost(endpoint)["weight"]
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
        if self.retry_after > self.get_synced_time_ms():
            logger.warning(f"[CHECK_LIMITS] cooling down until {self.retry_after}")
            return False
        return self._has_capacity(endpoint, **kwargs)

    def record_usage(self, endpoint: Endpoints, **kwargs):
        cost = self._cost(endpoint)
        self.weight_window.append((self.get_synced_time_ms(), cost["weight"]))
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

    # ---- error handling: completed in the next task ---------------------

    def _handle_call_error(
        self, e: BaseException, endpoint: Endpoints | None = None
    ) -> None:
        return None

    def local_cache_error(self, headers: dict[str, Any], **kwargs: Any) -> None:
        return None

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

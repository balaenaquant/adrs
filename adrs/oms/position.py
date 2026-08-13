import logging
import asyncio
import time

from decimal import Decimal
from datetime import datetime, timezone

from cybotrade import Symbol
from cybotrade.models import Position, OrderSide

from adrs.oms.config import ConfigManager
from adrs.oms.ops.order_pool import OpenOrdersSnapshot
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

Positions = dict[Symbol, Position]  # base_asset -> Position

# How stale an exchange-position read may be before update_exchange() goes back
# to the REST API. Positions are only ever as fresh as the last poll anyway, so
# this changes nothing for the placement tick (which runs far slower than this),
# and collapses a burst of websocket-driven refreshes into a single request.
POSITION_REFRESH_TTL_SEC = 1.0

# How stale the REST anchor may be before the order-sizing path stops trusting
# the streamed position and forces a read.
#
# ACCOUNT_UPDATE is event-driven, so silence is ambiguous -- a quiet account and a
# dead socket look identical -- and unlike the price feed there is no busy stream
# on the user-data socket to use as a heartbeat: its ping is every 3 minutes and
# the listenKey keepalive every 30. So the 60s REST reconcile is the liveness
# mechanism, and this is the window it is trusted for. Must exceed that 60s
# cadence, or the sizing path forces a read on most ticks and the saving is lost.
POSITION_ANCHOR_MAX_AGE_SEC = 90.0


class PositionManager:
    def __init__(self, config: ConfigManager, rate_limiter: RateLimiter):
        # The OrderPlacementManager will update these states
        self.exchange: Positions = {}
        self.pending: Positions = {}

        self.desired: Positions = {}
        self.config = config
        self.rate_limiter = rate_limiter
        self.delta_lock = asyncio.Lock()
        # Serialises exchange-position refreshes so concurrent callers share one
        # REST call instead of each spending GET_POSITION weight
        self._refresh_lock = asyncio.Lock()
        self._exchange_refreshed_at: float | None = None  # monotonic, None = never

    async def delta_calculation(
        self, snapshot: OpenOrdersSnapshot
    ) -> dict[Symbol, Decimal]:
        """
        Pending and Exchange values would be updated from REST
        Would fall back to WS local copy if rate limits has been reached

        Case 1:
          desire: +3
          exchange: -1
          pending: +2
          delta = 3 - (-1 + 2) = 2

        Case 2:
          desire: -3
          exchange: +1
          pending: -2
          delta = -3 - (1 - 2) = -2
        """
        # Prevents conditions where orders being replaced and calculation runs in the middle of it
        async with self.delta_lock:
            self.update_pending(snapshot)
            # Trust the streamed position while the REST anchor is fresh. The
            # anchor (set only by a real read, refreshed every 60s by
            # on_aegis_update) is the liveness proof; once it ages out this
            # forces a read exactly as it always did.
            await self.update_exchange(max_age_sec=POSITION_ANCHOR_MAX_AGE_SEC)
            deltas = {}
            for symbol, position in self.desired.items():
                exchange_pos = self.exchange[symbol]
                pending_pos = self.pending[symbol]
                delta = position.quantity - (
                    exchange_pos.quantity + pending_pos.quantity
                )
                deltas[symbol] = delta

            return deltas

    def compute_base_quantity(self, price: Decimal, weightage: Decimal) -> Decimal:
        """
        The quantity being return can be -ve/+ve depending on the side.
        If the position is sell, it would be negative and positive otherwise.
        """
        initial_balance, leverage = (
            self.config.config.initial_balance,
            self.config.config.leverage,
        )
        return (initial_balance * leverage * weightage) / price

    def _positions_fresh_within(self, max_age_sec: float) -> bool:
        if self._exchange_refreshed_at is None or max_age_sec <= 0:
            return False
        return (time.monotonic() - self._exchange_refreshed_at) < max_age_sec

    async def update_exchange(self, max_age_sec: float = POSITION_REFRESH_TTL_SEC):
        """
        Get the latest positions available from the exchange.

        Skips the call when the last successful read is younger than
        `max_age_sec`, and makes concurrent callers share one request rather than
        each spending GET_POSITION weight. A burst of websocket order updates
        used to fire one /fapi/v2/positionRisk per event — on Binance that is
        weight 5 apiece, against a budget metered per IP and shared with every
        co-tenant on the shard. Pass max_age_sec=0 to force a read.

        A failed read leaves the timestamp alone, so the next caller retries
        rather than trusting positions that were never fetched.
        """
        if self._positions_fresh_within(max_age_sec):
            return
        async with self._refresh_lock:
            # Re-check: whoever held the lock may have just refreshed, in which
            # case every caller queued behind them can use that result.
            if self._positions_fresh_within(max_age_sec):
                return
            endpoint = Endpoints.GET_POSITION
            try:
                async with self.rate_limiter.reserve(endpoint=endpoint):
                    exchange_positions = await self.config.exchange.get_positions()
                    for position in exchange_positions:
                        self.exchange[position.symbol] = position
                self._exchange_refreshed_at = time.monotonic()
            except Exception as e:
                logger.warning(f"Failed to update exchange due to {e}")

    def apply_stream_positions(self, positions: list[Position]) -> None:
        """
        Overwrite the exchange position for each symbol in a stream frame.

        Absolute, never accumulated: this replaces the incremental fill
        accounting, which drifted by construction. Applying the same frame twice
        is a no-op, which is the property that makes the replacement worth making.

        Frames are partial -- they carry what changed -- so symbols absent from
        `positions` keep their previous value rather than being zeroed. A position
        that has closed arrives as quantity 0 and is applied, not skipped: skipping
        it would leave the OMS sizing orders against a position it no longer holds.

        Deliberately does NOT stamp the REST anchor. Only a real REST read may do
        that, because the anchor is what proves liveness -- if the stream stamped
        it, a dead REST path would look healthy forever.
        """
        for position in positions:
            self.exchange[position.symbol] = position

    def update_pending(self, snapshot: OpenOrdersSnapshot):
        """
        Derive pending positions from an open-orders snapshot.

        Symbols absent from the snapshot had a failed fetch; their previous
        pending state is kept rather than zeroed.
        """
        for symbol, pending_orders in snapshot.orders.items():
            quantity = Decimal("0")
            for order in pending_orders:
                quantity += (
                    order.remain_size
                    if order.side == OrderSide.BUY
                    else -order.remain_size
                )
            entry_price = (
                Decimal("0") if len(pending_orders) == 0 else pending_orders[-1].price
            )
            self.pending[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=entry_price,
                updated_time=datetime.now(timezone.utc),
            )

import logging
import asyncio

from decimal import Decimal
from datetime import datetime, timezone

from cybotrade import Symbol
from cybotrade.models import Position, OrderSide

from adrs.oms.config import ConfigManager
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

Positions = dict[Symbol, Position]  # base_asset -> Position


class PositionManager:
    def __init__(self, config: ConfigManager, rate_limiter: RateLimiter):
        # The OrderPlacementManager will update these states
        self.exchange: Positions = {}
        self.pending: Positions = {}

        self.desired: Positions = {}
        self.config = config
        self.rate_limiter = rate_limiter
        self.delta_lock = asyncio.Lock()

    async def delta_calculation(self) -> dict[Symbol, Decimal]:
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
            await self.update_pending()
            await self.update_exchange()
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

    async def update_exchange(self):
        """
        Get the latest positions available from the exchange
        """
        endpoint = Endpoints.GET_POSITION
        try:
            async with self.rate_limiter.guard(endpoint=endpoint):
                exchange_positions = await self.config.exchange.get_positions()
                for position in exchange_positions:
                    self.exchange[position.symbol] = position
        except Exception as e:
            logger.warning(f"Failed to update exchange due to {e}")

    async def update_pending(self):
        """
        Get the latest open orders from the exchange
        """
        endpoint = Endpoints.GET_OPEN_ORDERS
        try:
            for s in self.config.config.base_asset_to_symbol_table.values():
                async with self.rate_limiter.guard(endpoint=endpoint):
                    symbol = Symbol(s)
                    pending_orders = await self.config.exchange.get_open_orders(symbol)
                    quantity = Decimal("0")
                    for order in pending_orders:
                        quantity += (
                            order.remain_size
                            if order.side == OrderSide.BUY
                            else -order.remain_size
                        )
                    entry_price = (
                        Decimal("0")
                        if len(pending_orders) == 0
                        else pending_orders[-1].price
                    )
                    self.pending[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        updated_time=datetime.now(timezone.utc),
                    )
        except Exception as e:
            logger.warning(f"Failed to update pending due to {e}")

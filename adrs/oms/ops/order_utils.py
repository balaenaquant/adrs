import uuid
import logging
import asyncio

from decimal import Decimal
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from cybotrade import Symbol
from cybotrade.io import ExchangeClient

from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints

if TYPE_CHECKING:
    from adrs.oms.rate_limit.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class OrderUtils:
    def __init__(self):
        pass

    @staticmethod
    async def get_order_book(
        exchange: ExchangeClient,
        pair: Symbol,
        need_log: bool,
        rate_limiter: "RateLimiter",
        endpoint: Endpoints = Endpoints.GET_ORDERBOOK_SNAPSHOT,
    ) -> list[Decimal]:
        # Guard each attempt so every actual snapshot request is counted by the
        # rate limiter, not just the first. Was 1 + 5 retries counted as 1.
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                async with rate_limiter.guard(endpoint=endpoint):
                    orderbook = await exchange.get_orderbook_snapshot(symbol=pair)
                (best_bid, best_ask) = (
                    max(map(lambda level: level.price, orderbook.bids)),
                    min(map(lambda level: level.price, orderbook.asks)),
                )
                if need_log:
                    logger.info(f"best_bid: {best_bid}, best_ask: {best_ask}")
                return [best_bid, best_ask]
            except Exception as e:
                last_err = e
                logger.error(f"Failed to fetch order book: {e}")
                if attempt < 5:
                    await asyncio.sleep(1)
        raise Exception(f"Failed to fetch order book after 6 attempts: {last_err}")

    @staticmethod
    def make_client_order_id():
        return f"{uuid.uuid4().hex}"

    @staticmethod
    def convert_ms_to_datetime(milliseconds):
        try:
            seconds = milliseconds / 1000.0
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except Exception as e:
            logger.error(
                f"[CONVERT_DATETIME] Failed to convert_ms_to_datetime due to: {e}"
            )
            return datetime.now(timezone.utc)

import uuid
import logging

from decimal import Decimal
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from cybotrade import Symbol
from cybotrade.io import ExchangeClient

from adrs.oms.rate_limit.exchange_limit_profiles import OMS_DEPTH_LIMIT, Endpoints

if TYPE_CHECKING:
    from adrs.oms.price_feed import PriceFeed
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
        price_feed: "PriceFeed | None" = None,
    ) -> list[Decimal]:
        # The websocket feed answers for free when it is proven live. A miss
        # (feed down, symbol unseen, or past the staleness backstop) falls
        # through to REST below, which is exactly the old behaviour.
        if price_feed is not None:
            quote = price_feed.get(pair)
            if quote is not None:
                if need_log:
                    logger.info(f"best_bid: {quote.bid}, best_ask: {quote.ask} (ws)")
                return [quote.bid, quote.ask]

        # reserve() waits for a rate-limit slot rather than failing, so
        # contention no longer needs a retry loop. Genuine fetch errors
        # propagate to the caller (placement backlogs it, expiry skips it).
        #
        # This is the OMS's only depth call site, and the limit it sends is the
        # same constant the limiter charges for — the two cannot drift.
        async with rate_limiter.reserve(endpoint=endpoint):
            orderbook = await exchange.get_orderbook_snapshot(
                symbol=pair, limit=OMS_DEPTH_LIMIT
            )
        (best_bid, best_ask) = (
            max(map(lambda level: level.price, orderbook.bids)),
            min(map(lambda level: level.price, orderbook.asks)),
        )
        if need_log:
            logger.info(f"best_bid: {best_bid}, best_ask: {best_ask}")
        return [best_bid, best_ask]

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

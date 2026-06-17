import uuid
import logging
import asyncio

from decimal import Decimal
from datetime import datetime, timezone

from cybotrade import Symbol
from cybotrade.io import ExchangeClient

logger = logging.getLogger(__name__)


class OrderUtils:
    def __init__(self):
        pass

    @staticmethod
    async def get_order_book(
        exchange: ExchangeClient,
        pair: Symbol,
        need_log: bool,
    ) -> list[Decimal]:
        try:
            orderbook = await exchange.get_orderbook_snapshot(symbol=pair)
            (best_bid, best_ask) = (
                max(map(lambda level: level.price, orderbook.bids)),
                min(map(lambda level: level.price, orderbook.asks)),
            )
            if need_log:
                logger.info(f"best_bid: {best_bid}, best_ask: {best_ask}")
            return [best_bid, best_ask]
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            best_bid = Decimal(str(0.0))
            best_ask = Decimal(str(0.0))
            for _ in range(0, 5):
                try:
                    orderbook = await exchange.get_orderbook_snapshot(symbol=pair)
                    (best_bid, best_ask) = (
                        max(map(lambda level: level.price, orderbook.bids)),
                        min(map(lambda level: level.price, orderbook.asks)),
                    )
                    logger.debug(f"best_bid: {best_bid}, best_ask: {best_ask}")
                    return [best_bid, best_ask]
                except Exception as e:
                    await asyncio.sleep(1)
                    logger.error(f"Failed to fetch order book: {e}")
            raise Exception()

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

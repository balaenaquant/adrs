from decimal import Decimal
from pydantic import BaseModel, Field
from typing import Any, Dict, Deque
from enum import Enum, auto

from cybotrade.models import Exchange


# Helper function for dynamic weight calculation (optional)
def get_depth_weight(limit: int = 1000) -> int:
    """Calculates REQUEST_WEIGHT for /fapi/v1/depth"""
    if limit <= 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


class Endpoints(Enum):
    # Market Data
    GET_SERVER_TIME = auto()
    GET_SYMBOL_INFO = auto()
    GET_ORDERBOOK_SNAPSHOT = auto()

    # Account & Trade
    PLACE_ORDER = auto()
    CANCEL_ORDER = auto()
    GET_ORDER_DETAILS = auto()

    # Account Info
    GET_WALLET_BALANCE = auto()
    GET_POSITION = auto()

    # Order Info
    GET_OPEN_ORDERS = auto()  # Getting open orders for one symbol
    GET_OPEN_ORDERS_ALL = auto()  # Getting open orders for all symbols at once


BINANCE_FUTURES_COSTS: dict[Endpoints, dict[str, int]] = {
    Endpoints.GET_SYMBOL_INFO: {"weight": 20, "orders": 0},
    Endpoints.GET_ORDERBOOK_SNAPSHOT: {"weight": 1, "orders": 0},
    Endpoints.PLACE_ORDER: {"weight": 0, "orders": 1},
    Endpoints.CANCEL_ORDER: {"weight": 1, "orders": 1},
    Endpoints.GET_ORDER_DETAILS: {"weight": 1, "orders": 0},
    Endpoints.GET_WALLET_BALANCE: {"weight": 5, "orders": 0},
    Endpoints.GET_POSITION: {"weight": 5, "orders": 0},
    Endpoints.GET_OPEN_ORDERS: {"weight": 1, "orders": 0},
    Endpoints.GET_OPEN_ORDERS_ALL: {"weight": 40, "orders": 0},
    Endpoints.GET_SERVER_TIME: {"weight": 1, "orders": 0},
}


class OpenOrdersFetchProfile(BaseModel):
    """Per-exchange strategy for fetching an all-symbols open-orders snapshot."""

    # Symbol count at which one bulk call beats per-symbol calls
    min_symbols_for_bulk: int = 1
    # Extra kwargs for get_open_orders(symbol=None), e.g. Bybit's settleCoin
    bulk_kwargs: dict[str, Any] = Field(default_factory=dict)
    # A response with this many rows may be truncated by the exchange's page
    # limit; the caller must fall back to per-symbol fetches
    truncation_row_limit: int | None = None


OPEN_ORDERS_FETCH_PROFILES: dict[Exchange, OpenOrdersFetchProfile] = {
    # Rate limit is per request, so bulk wins from the second symbol onward.
    # /v5/order/realtime without symbol requires settleCoin and pages at 50.
    Exchange.BYBIT_LINEAR: OpenOrdersFetchProfile(
        min_symbols_for_bulk=2,
        bulk_kwargs={"settleCoin": "USDT", "limit": 50},
        truncation_row_limit=50,
    ),
    # All-symbols openOrders costs weight 40 vs 1 per symbol and returns
    # everything in one page; only worth it past the break-even symbol count
    Exchange.BINANCE_LINEAR: OpenOrdersFetchProfile(min_symbols_for_bulk=40),
}
# Exchanges without a profile (Kucoin, EdgeX) always fetch per symbol.


class BinanceLimitProfile(BaseModel):
    request_weight_limit_per_minute: int
    order_limit_per_10_sec: int
    order_limit_per_minute: int


class BybitRateLimitPool(Enum):
    IP_GLOBAL = auto()

    UID_PLACE = auto()
    UID_CANCEL = auto()
    UID_POSITION = auto()
    UID_WALLET = auto()
    UID_OPEN_ORDERS = auto()


# Store the hard limits outside the class (or as a class variable)
DEFAULT_HARD_LIMITS = {
    BybitRateLimitPool.IP_GLOBAL: 120,
    BybitRateLimitPool.UID_PLACE: 10,
    BybitRateLimitPool.UID_CANCEL: 10,
    BybitRateLimitPool.UID_POSITION: 50,
    BybitRateLimitPool.UID_WALLET: 50,
    BybitRateLimitPool.UID_OPEN_ORDERS: 50,
}


class BybitLimitProfile(BaseModel):
    limits: Dict[BybitRateLimitPool, int] = Field(
        default_factory=lambda: DEFAULT_HARD_LIMITS.copy()
    )
    interval: int = 1

    @classmethod
    def with_buffer(cls, buffer_pct: Decimal = Decimal("0.2"), interval: int = 1):
        """
        Creates a profile with limits reduced by the buffer percentage.
        """
        multiplier = Decimal("1.0") - buffer_pct
        safe_limits = {
            pool: max(1, int(limit * multiplier))
            for pool, limit in DEFAULT_HARD_LIMITS.items()
        }
        return cls(limits=safe_limits, interval=interval)


class BybitLimitState(BaseModel):
    timestamps: Deque[int]  # time


# Follows per endpoint uid rolling window, also has a ip rate limit
# IP 600/5min
BYBIT_FUTURES_COSTS: dict[Endpoints, BybitRateLimitPool] = {
    Endpoints.PLACE_ORDER: BybitRateLimitPool.UID_PLACE,
    Endpoints.CANCEL_ORDER: BybitRateLimitPool.UID_CANCEL,
    Endpoints.GET_POSITION: BybitRateLimitPool.UID_POSITION,
    Endpoints.GET_WALLET_BALANCE: BybitRateLimitPool.UID_WALLET,
    Endpoints.GET_OPEN_ORDERS: BybitRateLimitPool.UID_OPEN_ORDERS,
    Endpoints.GET_OPEN_ORDERS_ALL: BybitRateLimitPool.UID_OPEN_ORDERS,
    Endpoints.GET_ORDER_DETAILS: BybitRateLimitPool.UID_OPEN_ORDERS,
}

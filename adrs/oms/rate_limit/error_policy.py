import re
from abc import ABC, abstractmethod
from enum import Enum, auto

from cybotrade.binance import BinanceError
from cybotrade.bybit import BybitError

# Binance reports rate limiting through the JSON body code, never the HTTP
# status: -1003 arrives with both 429 (request budget exhausted) and 418 (IP
# banned), and -1015 with an order-rate breach. Matching 418/429 against
# BinanceError.code -- which only ever holds the body code -- can therefore
# never fire, which is how an OMS came to poll straight through an hour-long IP
# ban, renewing it with every request.
BINANCE_RATE_LIMIT_CODES = frozenset({-1003, -1015})
# Scoped to the account's order budget, so these only need to stop order
# placement; reads can carry on.
BINANCE_ORDER_RATE_LIMIT_CODES = frozenset({-1015})
# Checked as well as the body code, because a 429/418 raised by Binance's edge
# rather than its matching engine can carry a body with no code field at all.
BINANCE_RATE_LIMIT_HTTP_STATUSES = frozenset({418, 429})

# Binance embeds the ban deadline in the -1003 message, e.g. "Way too many
# requests; IP(1.2.3.4) banned until 1786361397856." That epoch-ms deadline is
# the only authoritative recovery time on the response, since a 418 frequently
# carries no Retry-After header.
_BINANCE_BANNED_UNTIL_RE = re.compile(r"banned until (\d{13,})")


def is_binance_rate_limit_error(exc: BinanceError) -> bool:
    """Whether this Binance error means we are being rate limited or banned."""
    if exc.code in BINANCE_RATE_LIMIT_CODES:
        return True
    # cybotrade only began carrying the HTTP status on BinanceError in 2.0.19;
    # on older versions the attribute is absent and the body code is all there
    # is to go on.
    return getattr(exc, "http_status", None) in BINANCE_RATE_LIMIT_HTTP_STATUSES


def binance_banned_until_ms(message: str) -> int | None:
    """Epoch-ms ban deadline from a -1003 message, or None if it carries none."""
    match = _BINANCE_BANNED_UNTIL_RE.search(message)
    if match is None:
        return None
    return int(match.group(1))


class ErrorAction(Enum):
    TERMINAL_SUCCESS = auto()  # order already gone; treat as done, no retry
    RETRY = auto()  # transient/unknown; backlog + backoff (default)
    RATE_LIMITED = auto()  # rate code; cooldown is armed in the limiter's guard()
    FATAL = auto()  # unrecoverable for these params; drop + log, no retry


# Unlisted codes fall through to RETRY, matching legacy behaviour. FATAL entries
# stop retries, so only add a code here once its terminal nature is confirmed.
BYBIT_ERROR_ACTIONS: dict[int, ErrorAction] = {
    110001: ErrorAction.TERMINAL_SUCCESS,  # order not exists or too late to cancel
    10006: ErrorAction.RATE_LIMITED,  # too many visits
}

# Keyed on Binance's body codes. The 418/429 that used to sit here were HTTP
# statuses and so never matched; a rate-limit error fell through to RETRY, and
# the backlog kept re-sending orders every 2s for the length of the ban.
BINANCE_ERROR_ACTIONS: dict[int, ErrorAction] = {
    -2011: ErrorAction.TERMINAL_SUCCESS,  # unknown order (already gone)
    **{code: ErrorAction.RATE_LIMITED for code in BINANCE_RATE_LIMIT_CODES},
}


class ExchangeErrorPolicy(ABC):
    """Maps an exchange exception to the behaviour the OMS should take."""

    default_action: ErrorAction = ErrorAction.RETRY

    @abstractmethod
    def classify(self, exc: Exception) -> ErrorAction: ...


class BybitErrorPolicy(ExchangeErrorPolicy):
    def classify(self, exc: Exception) -> ErrorAction:
        if isinstance(exc, BybitError):
            if exc.http_status == 403:
                return ErrorAction.RATE_LIMITED
            if exc.retCode is not None:
                return BYBIT_ERROR_ACTIONS.get(exc.retCode, self.default_action)
        return self.default_action


class BinanceErrorPolicy(ExchangeErrorPolicy):
    def classify(self, exc: Exception) -> ErrorAction:
        if not isinstance(exc, BinanceError):
            return self.default_action
        # Status first: a 429/418 whose body has no code at all is still a rate
        # limit, and must not be retried as though it were transient.
        if is_binance_rate_limit_error(exc):
            return ErrorAction.RATE_LIMITED
        if exc.code is not None:
            return BINANCE_ERROR_ACTIONS.get(exc.code, self.default_action)
        return self.default_action


class DefaultErrorPolicy(ExchangeErrorPolicy):
    """Everything retries — legacy behaviour for exchanges without a policy."""

    def classify(self, exc: Exception) -> ErrorAction:
        return self.default_action

from abc import ABC, abstractmethod
from enum import Enum, auto

from cybotrade.binance import BinanceError
from cybotrade.bybit import BybitError


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

BINANCE_ERROR_ACTIONS: dict[int, ErrorAction] = {
    -2011: ErrorAction.TERMINAL_SUCCESS,  # unknown order (already gone)
    418: ErrorAction.RATE_LIMITED,
    429: ErrorAction.RATE_LIMITED,
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
        if isinstance(exc, BinanceError) and exc.code is not None:
            return BINANCE_ERROR_ACTIONS.get(exc.code, self.default_action)
        return self.default_action


class DefaultErrorPolicy(ExchangeErrorPolicy):
    """Everything retries — legacy behaviour for exchanges without a policy."""

    def classify(self, exc: Exception) -> ErrorAction:
        return self.default_action

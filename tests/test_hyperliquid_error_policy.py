import pytest
from cybotrade.hyperliquid import HyperliquidError

from adrs.oms.rate_limit.error_policy import (
    ErrorAction,
    HyperliquidErrorPolicy,
    is_hyperliquid_rate_limit_error,
)


@pytest.fixture
def policy():
    return HyperliquidErrorPolicy()


def test_already_gone_is_terminal_success(policy):
    """
    Verified live against mainnet: cancelling an unknown order id returns
    "Order was never placed, already canceled, or filled. asset=0" inside an
    HTTP 200. The order is gone, which is what the caller wanted, so this is
    TERMINAL_SUCCESS -- the same classification Bybit's 110001 gets. Treating
    it as FATAL would drop and log an order the OMS should count as done.
    """
    exc = HyperliquidError(
        "Order was never placed, already canceled, or filled. asset=0"
    )
    assert policy.classify(exc) == ErrorAction.TERMINAL_SUCCESS


@pytest.mark.parametrize(
    "message",
    ["Rate limit exceeded for address", "Too many requests, slow down"],
)
def test_rate_limit_messages(policy, message):
    exc = HyperliquidError(message)
    assert policy.classify(exc) == ErrorAction.RATE_LIMITED
    assert is_hyperliquid_rate_limit_error(exc) is True


@pytest.mark.parametrize(
    "message",
    [
        "Insufficient margin to place order",
        "Price too far from oracle price",
        "Reduce only order would increase position",
    ],
)
def test_unrecoverable_messages_are_fatal(policy, message):
    """
    These fail for the parameters given, not transiently. Retrying re-sends the
    same rejected order every couple of seconds for as long as the backlog
    lives, so they must stop the retry loop.
    """
    assert policy.classify(HyperliquidError(message)) == ErrorAction.FATAL


def test_unknown_message_falls_through_to_retry(policy):
    """Unlisted errors keep the legacy retry-everything behaviour."""
    exc = HyperliquidError("Some error Hyperliquid added last Tuesday")
    assert policy.classify(exc) == ErrorAction.RETRY
    assert is_hyperliquid_rate_limit_error(exc) is False


def test_matching_is_case_insensitive(policy):
    exc = HyperliquidError("INSUFFICIENT MARGIN to place order")
    assert policy.classify(exc) == ErrorAction.FATAL


def test_non_hyperliquid_exception_falls_through(policy):
    assert policy.classify(ValueError("unrelated")) == ErrorAction.RETRY
    assert is_hyperliquid_rate_limit_error(ValueError("rate limit")) is False


def test_rate_limit_helper_requires_a_hyperliquid_error():
    """
    The helper is used by the limiter to arm its cooldown. It must not fire on
    an arbitrary exception whose text happens to contain the words, or an
    unrelated failure would stall every call for 10 seconds.
    """
    assert is_hyperliquid_rate_limit_error(RuntimeError("rate limit")) is False

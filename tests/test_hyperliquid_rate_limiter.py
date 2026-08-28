import pytest
from decimal import Decimal
from types import SimpleNamespace

from cybotrade.hyperliquid import HyperliquidClient, HyperliquidError

from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.hyperliquid_limiter import HyperliquidRateLimiter
from adrs.oms.rate_limit.rate_limiter import LocalRateLimitError

KEY = "0x0123456789012345678901234567890123456789012345678901234567890123"
ADDRESS = "0x" + "42" * 20


def _limiter(tenants: int = 1, soft: str = "0.8") -> HyperliquidRateLimiter:
    """
    Drive the real __init__ so the budget derivation is under test rather than
    a copy of its arithmetic. Mirrors _binance_real_init in
    tests/test_rate_limiter_backoff.py.
    """
    client = HyperliquidClient(  # no IO on construct
        private_key=KEY, account_address=ADDRESS
    )
    config = SimpleNamespace(
        config=SimpleNamespace(
            soft_limit_percent=Decimal(soft), tenants_per_egress_ip=tenants
        ),
        exchange=client,
    )
    return HyperliquidRateLimiter(config)  # type: ignore[arg-type]


def test_rejects_a_mismatched_exchange():
    config = SimpleNamespace(
        config=SimpleNamespace(
            soft_limit_percent=Decimal("0.8"), tenants_per_egress_ip=1
        ),
        exchange=object(),
    )
    with pytest.raises(Exception, match="Exchange mismatch"):
        HyperliquidRateLimiter(config)  # type: ignore[arg-type]


def test_weight_budget_is_soft_limited_and_split_across_tenants():
    """
    The 1200/min is per IP. A shard is one NAT IP with many tenants, so each
    must claim only its share or they ban each other -- the same reasoning as
    the Binance split.
    """
    assert _limiter(tenants=1).limit_profile.request_weight_limit_per_minute == 960
    shared = _limiter(tenants=14)
    assert shared.limit_profile.request_weight_limit_per_minute == 68
    assert shared.limit_profile.request_weight_limit_per_minute * 14 <= 1200


def test_address_buffer_is_not_split_across_tenants():
    """The address budget is account-scoped, so it is not divided."""
    assert _limiter(tenants=14).limit_profile.address_action_buffer == 10_000


def test_synced_time_is_the_local_clock():
    import time

    limiter = _limiter()
    before = int(time.time() * 1000)
    got = limiter.get_synced_time_ms()
    assert before <= got <= int(time.time() * 1000)


def test_one_pool_for_every_endpoint():
    limiter = _limiter()
    keys = {limiter._pool_key(e) for e in Endpoints}
    assert len(keys) == 1


def test_record_usage_charges_the_documented_weight():
    limiter = _limiter()
    limiter.record_usage(endpoint=Endpoints.GET_ORDERBOOK_SNAPSHOT)  # 2
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)  # 20
    assert limiter.current_weight() == 22


def test_only_exchange_calls_increment_the_action_count():
    limiter = _limiter()
    limiter.record_usage(endpoint=Endpoints.GET_POSITION)
    assert limiter.address_actions == 0
    limiter.record_usage(endpoint=Endpoints.PLACE_ORDER)
    limiter.record_usage(endpoint=Endpoints.CANCEL_ORDER)
    assert limiter.address_actions == 2


def test_capacity_is_exhausted_at_the_ceiling():
    limiter = _limiter()  # 960 weight
    # 48 symbol-info reads at weight 20 = 960, exactly the ceiling
    for _ in range(48):
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    assert limiter.current_weight() == 960
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is False
    # a weight-0 endpoint still fits
    assert limiter._has_capacity(Endpoints.GET_SERVER_TIME) is True


def test_capacity_available_below_the_ceiling():
    limiter = _limiter()
    for _ in range(47):
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)  # 940
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is True  # 960 fits
    limiter.record_usage(endpoint=Endpoints.GET_ORDERBOOK_SNAPSHOT)  # 942
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is False  # 962 does not


def test_window_expires_after_sixty_seconds(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    for _ in range(48):
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is False

    # 59.9s later the window still holds everything
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 59_900)
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is False

    # past 60s the entries age out
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 60_001)
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is True
    assert limiter.current_weight() == 0


def test_next_free_delay_reports_when_the_window_frees(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    for _ in range(48):
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)

    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 20_000)
    # oldest entry ages out 60s after it was recorded, i.e. 40s from now
    assert limiter._next_free_delay(Endpoints.GET_SYMBOL_INFO) == pytest.approx(40.0)


def test_next_free_delay_is_zero_with_capacity():
    limiter = _limiter()
    assert limiter._next_free_delay(Endpoints.PLACE_ORDER) == 0.0


def test_reset_limits_clears_state():
    limiter = _limiter()
    limiter.record_usage(endpoint=Endpoints.PLACE_ORDER)
    limiter.reset_limits()
    assert limiter.current_weight() == 0
    assert limiter.address_actions == 0


@pytest.mark.asyncio
async def test_guard_admits_and_charges():
    limiter = _limiter()
    async with limiter.guard(Endpoints.PLACE_ORDER):
        pass
    assert limiter.current_weight() == 1
    assert limiter.address_actions == 1


@pytest.mark.asyncio
async def test_guard_raises_when_exhausted():
    limiter = _limiter()
    for _ in range(48):
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    with pytest.raises(LocalRateLimitError):
        async with limiter.guard(Endpoints.GET_SYMBOL_INFO):
            pass


def test_repr_shows_usage_against_both_budgets():
    limiter = _limiter()
    limiter.record_usage(endpoint=Endpoints.PLACE_ORDER)
    text = repr(limiter)
    assert "1/960" in text
    assert "1/10000" in text


def test_throttle_error_arms_the_cooldown(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)

    limiter._handle_call_error(
        HyperliquidError("Rate limit exceeded for address"), Endpoints.PLACE_ORDER
    )
    # Hyperliquid throttles a spent address to one request every 10s
    assert limiter.retry_after == now + 10_000
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is False


def test_cooldown_expires(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter._handle_call_error(HyperliquidError("Rate limit exceeded"), None)

    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 10_001)
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is True


def test_non_throttle_error_does_not_arm_the_cooldown(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter._handle_call_error(
        HyperliquidError("Insufficient margin to place order"), Endpoints.PLACE_ORDER
    )
    assert limiter.retry_after == 0
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is True


def test_unrelated_exception_does_not_arm_the_cooldown(monkeypatch):
    """
    A timeout or a bug must not stall every call for ten seconds. Only a
    Hyperliquid throttle signal arms the cooldown.
    """
    limiter = _limiter()
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: 1_700_000_000_000)
    limiter._handle_call_error(TimeoutError("slow"), Endpoints.PLACE_ORDER)
    assert limiter.retry_after == 0


def test_local_cache_error_arms_from_the_message(monkeypatch):
    """
    Hyperliquid sends no rate-limit headers, so this folds the message in
    instead of reading the header dict. A silent no-op here would look like an
    oversight rather than a property of the exchange.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.local_cache_error({}, message="Rate limit exceeded for address")
    assert limiter.retry_after == now + 10_000


def test_local_cache_error_ignores_an_unrelated_message(monkeypatch):
    limiter = _limiter()
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: 1_700_000_000_000)
    limiter.local_cache_error({}, message="something else")
    assert limiter.retry_after == 0


@pytest.mark.asyncio
async def test_guard_arms_the_cooldown_on_a_throttle(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)

    with pytest.raises(HyperliquidError):
        async with limiter.guard(Endpoints.PLACE_ORDER):
            raise HyperliquidError("Rate limit exceeded for address")

    assert limiter.retry_after == now + 10_000

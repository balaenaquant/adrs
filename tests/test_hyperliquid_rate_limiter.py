import asyncio
import pytest
from decimal import Decimal
from types import SimpleNamespace

from cybotrade.exceptions import DeserializationError
from cybotrade.hyperliquid import HyperliquidClient, HyperliquidError

from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.hyperliquid_limiter import HyperliquidRateLimiter
from adrs.oms.rate_limit.rate_limiter import LocalRateLimitError

KEY = "0x0123456789012345678901234567890123456789012345678901234567890123"
ADDRESS = "0x" + "42" * 20

# A weight-20 endpoint that is NOT amortised, for the tests that need to fill
# the window. GET_SYMBOL_INFO is charged at most once per window (see
# _effective_weight), so it cannot be used to accumulate weight.
HEAVY = Endpoints.GET_OPEN_ORDERS
# The cooldown any rate-limit signal arms. Long enough to outlast a whole IP
# weight window, because the signal does not say which axis fired.
COOLDOWN_MS = 65_000


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
    # 48 open-orders reads at weight 20 = 960, exactly the ceiling
    for _ in range(48):
        limiter.record_usage(endpoint=HEAVY)
    assert limiter.current_weight() == 960
    assert limiter._has_capacity(HEAVY) is False
    # a weight-0 endpoint still fits
    assert limiter._has_capacity(Endpoints.GET_SERVER_TIME) is True


def test_capacity_available_below_the_ceiling():
    limiter = _limiter()
    for _ in range(47):
        limiter.record_usage(endpoint=HEAVY)  # 940
    assert limiter._has_capacity(HEAVY) is True  # 960 fits
    limiter.record_usage(endpoint=Endpoints.GET_ORDERBOOK_SNAPSHOT)  # 942
    assert limiter._has_capacity(HEAVY) is False  # 962 does not


def test_window_expires_after_sixty_seconds(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    for _ in range(48):
        limiter.record_usage(endpoint=HEAVY)
    assert limiter._has_capacity(HEAVY) is False

    # 59.9s later the window still holds everything
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 59_900)
    assert limiter._has_capacity(HEAVY) is False

    # past 60s the entries age out
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 60_001)
    assert limiter._has_capacity(HEAVY) is True
    assert limiter.current_weight() == 0


def test_next_free_delay_reports_when_the_window_frees(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    for _ in range(48):
        limiter.record_usage(endpoint=HEAVY)

    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 20_000)
    # oldest entry ages out 60s after it was recorded, i.e. 40s from now
    assert limiter._next_free_delay(HEAVY) == pytest.approx(40.0)


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
        limiter.record_usage(endpoint=HEAVY)
    with pytest.raises(LocalRateLimitError):
        async with limiter.guard(HEAVY):
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
    # 65s, not 10s: the message cannot tell an address throttle (10s) from an
    # IP weight overrun (up to a full 60s window), so the longer hold is taken.
    assert limiter.retry_after == now + COOLDOWN_MS
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is False


def test_cooldown_expires(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter._handle_call_error(HyperliquidError("Rate limit exceeded"), None)

    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + COOLDOWN_MS + 1)
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is True


def test_cooldown_still_holds_on_its_exact_deadline_millisecond(monkeypatch):
    """
    >= not >, matching reserve() and both sibling limiters. On the deadline
    millisecond itself the hold still applies; admitting there would let a call
    out one tick early into a budget the exchange may still be counting.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter._handle_call_error(HyperliquidError("Rate limit exceeded"), None)
    deadline = now + COOLDOWN_MS

    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: deadline)
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is False
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: deadline + 1)
    assert limiter.check_limits(Endpoints.PLACE_ORDER) is True


def test_an_http_429_arms_the_cooldown(monkeypatch):
    """
    cybotrade's _post_info never inspects the HTTP status, so a 429 on an info
    call has no Hyperliquid body to match needles against -- the status is the
    only signal there is. Missing it means the reactive half of the design
    never fires for reads, and the OMS polls straight through the throttle.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    # A body with no needle in it at all: only the status says rate limit
    limiter._handle_call_error(
        HyperliquidError("<html>nginx</html>", status=429), Endpoints.GET_POSITION
    )
    assert limiter.retry_after == now + COOLDOWN_MS
    assert limiter.check_limits(Endpoints.GET_POSITION) is False


def test_a_deserialization_error_arms_the_cooldown(monkeypatch):
    """
    Deliberately broad. cybotrade hands the response body straight to
    json.loads without looking at the status, so a 429's non-JSON body reaches
    this layer as DeserializationError and as nothing else -- indistinguishable
    from any other malformed response. Holding briefly on a read failure costs
    latency; missing a real 429 renews an IP ban across every tenant on the
    shared egress address.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter._handle_call_error(
        DeserializationError("Failed to load json: Expecting value"),
        Endpoints.GET_ORDERBOOK_SNAPSHOT,
    )
    assert limiter.retry_after == now + COOLDOWN_MS
    assert limiter.check_limits(Endpoints.GET_ORDERBOOK_SNAPSHOT) is False


def test_an_unrelated_value_error_does_not_arm_the_cooldown(monkeypatch):
    """
    The broadening stops at DeserializationError. An ordinary bug must not
    stall every call for a minute.
    """
    limiter = _limiter()
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: 1_700_000_000_000)
    limiter._handle_call_error(ValueError("unrelated"), Endpoints.GET_POSITION)
    assert limiter.retry_after == 0
    assert limiter.check_limits(Endpoints.GET_POSITION) is True


@pytest.mark.asyncio
async def test_guard_arms_the_cooldown_on_a_deserialization_error(monkeypatch):
    """End to end: the suspected-throttle path fires from inside guard()."""
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)

    with pytest.raises(DeserializationError):
        async with limiter.guard(Endpoints.GET_ORDERBOOK_SNAPSHOT):
            raise DeserializationError("Failed to load json: Expecting value")

    assert limiter.retry_after == now + COOLDOWN_MS


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
    assert limiter.retry_after == now + COOLDOWN_MS


def test_handle_call_error_routes_through_local_cache_error(monkeypatch):
    """
    One classifier, two entry points. _handle_call_error() delegates rather
    than re-matching, so local_cache_error() is reachable and the needles
    cannot drift between the two.
    """
    limiter = _limiter()
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: 1_700_000_000_000)
    seen: list[tuple[dict, dict]] = []
    real = limiter.local_cache_error

    def spy(headers, **kwargs):
        seen.append((headers, kwargs))
        return real(headers, **kwargs)

    monkeypatch.setattr(limiter, "local_cache_error", spy)
    exc = HyperliquidError("Rate limit exceeded for address")
    limiter._handle_call_error(exc, Endpoints.PLACE_ORDER)

    assert len(seen) == 1
    headers, kwargs = seen[0]
    # Hyperliquid sends no rate-limit headers, so there is nothing to pass
    assert headers == {}
    assert kwargs["error"] is exc
    assert limiter.retry_after > 0


def test_local_cache_error_arms_from_an_error_object(monkeypatch):
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.local_cache_error({}, error=HyperliquidError("nope", status=429))
    assert limiter.retry_after == now + COOLDOWN_MS


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

    assert limiter.retry_after == now + COOLDOWN_MS


# ---- GET_SYMBOL_INFO amortisation ---------------------------------------


def test_symbol_info_sweep_charges_once_per_window(monkeypatch):
    """
    config.py's update_symbol_info() takes one guard per symbol, but cybotrade
    caches metaAndAssetCtxs for 5 minutes, so a 20-symbol sweep issues exactly
    one real weight-20 call. Charging per symbol bills 400 of the minute's
    1200 for one request -- a phantom charge, the same failure the Binance cost
    table already documents.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)

    for _ in range(20):
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)

    assert limiter.current_weight() == 20  # not 400


def test_symbol_info_sweep_completes_on_a_fourteen_tenant_shard(monkeypatch):
    """
    The livelock this fixes. At the documented 14-tenant shard the ceiling is
    68/min, so a per-symbol charge of 20 exhausts it after three symbols: the
    sweep never finishes, _symbol_info_refreshed_at is never stamped, and every
    on_order_placement tick retries it -- starving PLACE_ORDER and CANCEL_ORDER
    behind a refresh that can never complete.
    """
    limiter = _limiter(tenants=14)
    assert limiter.limit_profile.request_weight_limit_per_minute == 68
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)

    for _ in range(20):
        assert limiter.check_limits(endpoint=Endpoints.GET_SYMBOL_INFO) is True
        limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)

    assert limiter.current_weight() == 20
    # And there is still headroom left for the trading path
    assert limiter.check_limits(endpoint=Endpoints.PLACE_ORDER) is True


def test_symbol_info_charges_again_in_a_later_window(monkeypatch):
    """
    The amortisation is capped at once per weight window, not once forever. It
    is coupled to cybotrade's 5-minute METADATA_TTL: charging up to once per
    minute against one real call per five is a 5x overcharge, in the safe
    direction.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    assert limiter.current_weight() == 20

    # Still inside the window: free
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 59_999)
    assert limiter._effective_weight(Endpoints.GET_SYMBOL_INFO) == 0

    # Past the window: charged in full again
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now + 60_001)
    assert limiter._effective_weight(Endpoints.GET_SYMBOL_INFO) == 20
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    assert limiter.current_weight() == 20  # the first charge has aged out


def test_amortisation_does_not_leak_to_other_endpoints(monkeypatch):
    """Only GET_SYMBOL_INFO is cached client-side; every other read pays each time."""
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    for _ in range(3):
        limiter.record_usage(endpoint=HEAVY)
    assert limiter.current_weight() == 20 + 60


def test_capacity_check_agrees_with_the_amortised_charge(monkeypatch):
    """
    _has_capacity() must project the same figure record_usage() charges, or a
    sweep is denied for weight that is about to cost nothing.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    # Fill the rest of the window to exactly the ceiling with real weight
    for _ in range(47):
        limiter.record_usage(endpoint=HEAVY)
    assert limiter.current_weight() == 960
    # A weight-20 read is refused, but the free symbol-info read still fits
    assert limiter._has_capacity(HEAVY) is False
    assert limiter._has_capacity(Endpoints.GET_SYMBOL_INFO) is True


def test_reset_limits_reopens_the_amortisation_window(monkeypatch):
    """A reset must not leave a free ride behind; the next read pays in full."""
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    assert limiter._effective_weight(Endpoints.GET_SYMBOL_INFO) == 0
    limiter.reset_limits()
    assert limiter._effective_weight(Endpoints.GET_SYMBOL_INFO) == 20


def test_a_clock_step_backwards_charges_in_full(monkeypatch):
    """
    If the clock moves backwards the elapsed time is meaningless, so the charge
    is taken rather than skipped -- overcharging is the safe direction.
    """
    limiter = _limiter()
    now = 1_700_000_000_000
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now)
    limiter.record_usage(endpoint=Endpoints.GET_SYMBOL_INFO)
    monkeypatch.setattr(limiter, "get_synced_time_ms", lambda: now - 5_000)
    assert limiter._effective_weight(Endpoints.GET_SYMBOL_INFO) == 20


# ---- reserve() priority --------------------------------------------------


def test_guard_yields_while_a_reserver_is_queued():
    """
    check_limits() must defer to anything parked in reserve() on this pool.
    Without it the four reserve() sites (position.py, order_pool.py,
    order_utils.py) lose their whole mechanism: guard() callers keep taking the
    capacity a queued reserver is waiting for. Same shape as the Binance and
    Bybit limiters, which both implement this rule.
    """
    limiter = _limiter()
    key = limiter._pool_key(HEAVY)
    # Simulate a caller parked in reserve() on this pool
    limiter._waiters[key] = 1
    # Capacity is free, but guard must still defer to the queued reserver
    assert limiter._has_capacity(HEAVY) is True
    assert limiter.check_limits(endpoint=HEAVY) is False
    # Once the reserver leaves, guard proceeds again
    limiter._waiters[key] = 0
    assert limiter.check_limits(endpoint=HEAVY) is True


@pytest.mark.asyncio
async def test_a_queued_reserver_is_not_starved_by_a_guard_caller(monkeypatch):
    """
    End to end: with the window full and a reserver waiting, freed capacity
    goes to the reserver -- a guard() caller racing for it is refused.
    """
    monkeypatch.setattr(
        HyperliquidRateLimiter, "_next_free_delay", lambda self, ep: 0.005
    )
    limiter = _limiter()
    key = limiter._pool_key(HEAVY)
    for _ in range(48):
        limiter.record_usage(endpoint=HEAVY)  # window full at 960
    assert limiter._has_capacity(HEAVY) is False
    admitted: list[str] = []

    async def reserver():
        async with limiter.reserve(endpoint=HEAVY):
            admitted.append("reserve")

    async def guarder():
        # Wait until the reserver is actually parked in the queue
        while limiter._waiters.get(key, 0) == 0:
            await asyncio.sleep(0)
        limiter.weight_window.clear()  # capacity frees
        assert limiter._waiters[key] > 0  # the reserver is still queued
        assert limiter._has_capacity(HEAVY) is True  # and capacity is genuinely free
        try:
            async with limiter.guard(HEAVY):
                admitted.append("guard")
        except LocalRateLimitError:
            admitted.append("guard-denied")

    await asyncio.gather(reserver(), guarder())

    assert "guard" not in admitted  # the guard caller was made to wait
    assert "guard-denied" in admitted
    assert "reserve" in admitted  # and the reserver got the slot


# ---- reset_limits() contract --------------------------------------------


def test_reset_limits_docstring_warns_against_recovering_from_a_reconnect():
    """
    The method is public because the ABC requires it and nothing calls it
    today, so its docstring is the only thing steering the first caller.
    Clearing the window discards spend Hyperliquid is still counting inside its
    own trailing 60s, which over-admits into a budget shared across the shard.
    The docstring must warn about that rather than offer "on reconnect" as an
    example of when to do it.
    """
    # Normalised, so the assertions do not depend on where the lines wrap
    doc = " ".join((HyperliquidRateLimiter.reset_limits.__doc__ or "").split())
    # The warning, naming what is lost and why it over-admits
    assert "WARNING" in doc
    assert "discards weight the exchange is still counting" in doc
    assert "must not be called to recover from a connection event" in doc
    # The invitation it replaced must be gone
    assert "e.g. on reconnect" not in doc
    # And it still contrasts with the safe, passive path
    assert "_trim_window()" in doc

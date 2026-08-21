"""
Tests for rate-limit *backoff*: what the limiter does once an exchange says no.

These exist because the backoff path was unreachable in production. Binance
reports rate limiting with a negative body code (-1003) and BinanceError.code
only ever holds that body code, but the handler matched `code == 418 or code ==
429` — HTTP statuses. Nothing ever matched, retry_after was never armed, and the
OMS polled straight through an hour-long IP ban, renewing it with every request.
The header lookups had the mirror-image problem: they matched canonical casing
against headers that arrive lowercased from reqwest.

So each test here names the production signal it stands in for, and the fixtures
use the casing a real response carries (lowercase), not the casing that made the
old tests pass.
"""

import asyncio
from collections import deque
from decimal import Decimal
from types import SimpleNamespace

from cybotrade.binance import BinanceError, BinanceLinearClient
from cybotrade.bybit import BybitError

from adrs.oms.rate_limit.rate_limiter import (
    BinanceRateLimiter,
    BybitRateLimiter,
    _BLIND_COOLDOWN_MS,
    _COOLDOWN_JITTER_MAX_MS,
    _COOLDOWN_SAFETY_MS,
    _COOLDOWN_RAMP_MS,
)
from adrs.oms.rate_limit.exchange_limit_profiles import (
    BinanceLimitProfile,
    BybitLimitProfile,
    BybitLimitState,
    BybitRateLimitPool,
    Endpoints,
    OMS_DEPTH_LIMIT,
    get_depth_weight,
)

NOW_MS = 1_786_000_000_000
BAN_UNTIL_MS = NOW_MS + 3_600_000  # an hour out, as Binance actually hands out

# Verbatim shape of the -1003 body seen in production logs.
IP_BAN_MESSAGE = (
    "Error Code -1003: {'code': -1003, 'msg': 'Way too many requests; "
    f"IP(130.176.187.76) banned until {BAN_UNTIL_MS}. "
    "Please use the websocket for live updates to avoid bans.'}"
)


class FakeBinanceError(BinanceError):
    """BinanceError with settable fields; the real one parses an http.Response."""

    def __init__(self, code=None, http_status=None, headers=None, message=""):
        self._code = code
        self._hs = http_status
        self.response_headers = headers or {}
        self.message = message
        Exception.__init__(self, message)

    @property
    def code(self):
        return self._code

    @property
    def http_status(self):
        return self._hs


class FakeBybitError(BybitError):
    def __init__(self, retCode=None, http_status=None, headers=None):
        self._rc = retCode
        self._hs = http_status
        self.response_headers = headers or {}

    @property
    def retCode(self):
        return self._rc

    @property
    def http_status(self):
        return self._hs


def _binance(*, weight_limit=1920, order_1m=960, order_10s=240) -> BinanceRateLimiter:
    """
    BinanceRateLimiter with __init__ skipped and a fixed clock.

    `weight_limit` is this tenant's (possibly divided) share. There is no
    IP-wide budget: weight is tracked purely locally, because
    x-mbx-used-weight-1m does not report the egress IP's spend.
    """
    lim = BinanceRateLimiter.__new__(BinanceRateLimiter)
    lim._reserve_locks = {}
    lim._waiters = {}
    lim.retry_after = 0
    lim.exchange_time_offset = 0
    lim.exchange = None
    lim.limit_profile = BinanceLimitProfile(
        request_weight_limit_per_minute=weight_limit,
        order_limit_per_minute=order_1m,
        order_limit_per_10_sec=order_10s,
    )
    lim.current_limit_state = BinanceLimitProfile(
        request_weight_limit_per_minute=0,
        order_limit_per_minute=0,
        order_limit_per_10_sec=0,
    )
    lim.last_reset_10s_timestamp = NOW_MS // 10_000
    lim.last_reset_1m_timestamp = NOW_MS // 60_000
    # What __init__ derives from USDⓈ-M's advertised rateLimits; asserted
    # against the real constructor in
    # test_usage_header_names_are_derived_from_the_advertised_rate_limits.
    # Order counts only -- the weight header is deliberately absent.
    lim._usage_headers = {
        "x-mbx-order-count-1m": "order_limit_per_minute",
        "x-mbx-order-count-10s": "order_limit_per_10_sec",
    }
    lim._now = NOW_MS
    lim.get_synced_time_ms = lambda: lim._now
    return lim


def _bybit() -> BybitRateLimiter:
    lim = BybitRateLimiter.__new__(BybitRateLimiter)
    lim._reserve_locks = {}
    lim._waiters = {}
    lim.retry_after = 0
    lim.exchange_time_offset = 0
    lim.exchange = None
    lim.limit_profile = BybitLimitProfile(
        limits={pool: 10 for pool in BybitRateLimitPool}, interval=1
    )
    lim.current_limit_state = {
        pool: BybitLimitState(timestamps=deque()) for pool in BybitRateLimitPool
    }
    lim._now = NOW_MS
    lim.get_synced_time_ms = lambda: lim._now
    return lim


# --- Binance: the error that was never recognised -------------------------


def test_body_code_1003_arms_the_cooldown():
    """
    The regression that mattered. -1003 is what Binance actually sends; before
    the fix this left retry_after at 0 and the OMS kept polling.
    """
    lim = _binance()
    lim._handle_call_error(FakeBinanceError(code=-1003, message=IP_BAN_MESSAGE))
    assert lim.retry_after > lim._now
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is False


def test_cooldown_runs_to_the_ban_deadline_in_the_message():
    """
    A 418 usually carries no Retry-After, so the "banned until <epoch ms>" in
    the message is the only authoritative recovery time on the response.
    """
    lim = _binance()
    lim._handle_call_error(
        FakeBinanceError(code=-1003, http_status=418, message=IP_BAN_MESSAGE)
    )
    assert lim.retry_after >= BAN_UNTIL_MS
    # ...and not wildly beyond it. The allowance is the safety margin plus the
    # de-synchronisation jitter, expressed in terms of the constants rather than
    # a magic number so it cannot silently drift away from them.
    assert (
        lim.retry_after <= BAN_UNTIL_MS + _COOLDOWN_SAFETY_MS + _COOLDOWN_JITTER_MAX_MS
    )


def test_retry_after_header_is_read_despite_lowercase_casing():
    """reqwest lowercases header names; "Retry-After" never matched literally."""
    lim = _binance()
    lim._handle_call_error(
        FakeBinanceError(
            code=-1003,
            http_status=429,
            headers={"retry-after": "120"},
            message="Error Code -1003: {'code': -1003, 'msg': 'Too many requests'}",
        )
    )
    assert lim.retry_after >= lim._now + 120_000


def test_http_429_with_no_body_code_still_arms_the_cooldown():
    """A 429 from Binance's edge can arrive with no code field at all."""
    lim = _binance()
    lim._handle_call_error(FakeBinanceError(code=None, http_status=429))
    assert lim.retry_after >= lim._now + _BLIND_COOLDOWN_MS


def test_rate_limit_with_no_deadline_still_backs_off():
    """
    Previously a -1003 without a Retry-After header fell into the order-limit
    branch and set no cooldown at all, so the caller retried immediately.
    """
    lim = _binance()
    lim._handle_call_error(FakeBinanceError(code=-1003, message="no deadline here"))
    assert lim.retry_after >= lim._now + _BLIND_COOLDOWN_MS


def test_order_rate_breach_blocks_orders_but_not_reads():
    """
    -1015 is metered against the account's order budget, so reads must keep
    working; blocking everything would stall position tracking needlessly.
    """
    lim = _binance()
    lim._handle_call_error(FakeBinanceError(code=-1015))
    assert lim.retry_after == 0
    assert lim.check_limits(endpoint=Endpoints.PLACE_ORDER) is False
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


def test_an_unrelated_error_arms_nothing():
    lim = _binance()
    lim._handle_call_error(FakeBinanceError(code=-2011))  # unknown order
    assert lim.retry_after == 0
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


def test_a_later_nearer_deadline_never_shortens_the_cooldown():
    lim = _binance()
    lim._handle_call_error(
        FakeBinanceError(code=-1003, http_status=418, message=IP_BAN_MESSAGE)
    )
    banned_until = lim.retry_after
    # A 429 for a different endpoint lands afterwards with a 1s Retry-After
    lim._handle_call_error(
        FakeBinanceError(code=-1003, http_status=429, headers={"retry-after": "1"})
    )
    assert lim.retry_after == banned_until


def test_multi_day_ban_deadline_is_honoured():
    """
    Binance escalates a repeat offender's IP ban to three days. A one-day trust
    horizon threw those deadlines away as implausible, leaving _arm_cooldown to
    fall back to the 65s blind cooldown — after which the OMS polls straight
    through the rest of the ban and renews it. The worst case has to be the one
    the deadline is trusted for.
    """
    lim = _binance()
    two_days_out = NOW_MS + 2 * 24 * 3600 * 1000
    lim._handle_call_error(
        FakeBinanceError(
            code=-1003,
            http_status=418,
            message=f"banned until {two_days_out}",
        )
    )
    assert lim.retry_after >= two_days_out
    assert lim.retry_after <= two_days_out + 5_000
    # ...and specifically not the blind cooldown that used to replace it
    assert lim.retry_after > lim._now + _BLIND_COOLDOWN_MS


def test_implausible_ban_deadline_is_rejected():
    """One malformed value must not park the OMS for a decade."""
    lim = _binance()
    lim._handle_call_error(
        FakeBinanceError(
            code=-1003,
            message=f"banned until {NOW_MS + 10 * 365 * 24 * 3600 * 1000}",
        )
    )
    assert lim.retry_after <= lim._now + _BLIND_COOLDOWN_MS + 1_000


# --- Binance: adopting the exchange's own counters -------------------------


def test_the_used_weight_header_is_ignored_entirely():
    """
    x-mbx-used-weight-1m is never read, however large it gets.

    It does not report the egress IP's spend. Probed from one pod on one shard,
    within seconds: /fapi/v1/time answered 8 and /fapi/v1/exchangeInfo 9, while
    /fapi/v3/positionRisk answered 1389 and /fapi/v2/balance 2293, and a
    /fapi/v1/time straight afterwards dropped back to 10. The signed figures
    also moved about tenfold between samples while our own traffic was
    unchanged, so they track some shared pool, not this IP.

    Since the OMS polls signed endpoints constantly, believing that number
    throttled trading on a quantity unrelated to our traffic.
    """
    lim = _binance()
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True

    lim.exchange = SimpleNamespace(
        last_response_headers={"x-mbx-used-weight-1m": "999999"}
    )
    lim._on_call_success(Endpoints.GET_POSITION)

    # Not adopted anywhere: not as a separate counter, not into our own tally.
    assert not hasattr(lim, "_ip_wide_used_weight")
    assert lim.current_limit_state.request_weight_limit_per_minute == 0
    # And it cannot refuse a read.
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


def test_a_huge_weight_header_never_stalls_the_oms():
    """
    Regression test for the live near-miss: readings sawtoothed to 3337 and a
    direct probe hit 4433 against a 4800 refusal threshold, one step from the
    OMS refusing every weighted call each minute. Repeated large readings must
    leave admission untouched.
    """
    lim = _binance(weight_limit=4800)
    for reading in ("1389", "2293", "3337", "4433", "4799"):
        lim.exchange = SimpleNamespace(
            last_response_headers={"x-mbx-used-weight-1m": reading}
        )
        lim._on_call_success(Endpoints.GET_POSITION)
        assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True
    # Only what this process actually spent is tracked, and it spent nothing:
    # _on_call_success does not charge, record_usage does.
    assert lim.current_limit_state.request_weight_limit_per_minute == 0


def test_this_tenants_own_share_is_still_enforced():
    """
    The split is what remains of the co-tenant protection and must survive.
    Weight is charged locally by record_usage(), never adopted from a header.
    """
    lim = _binance(weight_limit=137)  # 2400 * 0.8 / 14
    for _ in range(27):  # 27 * 5 = 135 of 137
        lim.record_usage(Endpoints.GET_POSITION)
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is False


def test_reconcile_keeps_in_flight_reservations():
    """
    record_usage() charges before the request is sent, but the header only counts
    requests Binance has already processed. Overwriting would refund every
    in-flight reservation, so the higher of the two wins.
    """
    lim = _binance()
    lim.current_limit_state.order_limit_per_minute = 500
    lim.exchange = SimpleNamespace(last_response_headers={"x-mbx-order-count-1m": "20"})
    lim._on_call_success(Endpoints.PLACE_ORDER)
    assert lim.current_limit_state.order_limit_per_minute == 500


def test_the_weight_window_boundary_zeroes_our_own_tally():
    """
    Our own weight tally is what admission is judged against, so it must roll
    with the minute or a spent window would keep refusing into the next one.
    """
    lim = _binance(weight_limit=10)
    lim.record_usage(Endpoints.GET_POSITION)  # weight 5
    lim.record_usage(Endpoints.GET_POSITION)  # 10 of 10
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is False
    lim._now += 60_000
    lim.reset_limits()
    assert lim.current_limit_state.request_weight_limit_per_minute == 0
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


def test_header_from_the_previous_window_is_not_adopted_after_a_roll():
    """
    reset_limits() rolls on our synced clock; the header describes Binance's
    window as of when it processed the request. A response that straddles the
    boundary would otherwise drop a nearly-full previous-minute order count into
    a freshly zeroed window and block orders for up to a full minute.
    """
    lim = _binance()
    lim.exchange = SimpleNamespace(
        last_response_headers={"x-mbx-order-count-1m": "959"}
    )
    lim._now += 60_000  # the window rolled while the response was in flight
    lim._on_call_success(Endpoints.PLACE_ORDER)
    assert lim.current_limit_state.order_limit_per_minute == 0
    assert lim.check_limits(endpoint=Endpoints.PLACE_ORDER) is True

    # The very next response re-establishes the figure within the new window
    lim._on_call_success(Endpoints.PLACE_ORDER)
    assert lim.current_limit_state.order_limit_per_minute == 959


def test_order_count_headers_are_adopted_too():
    lim = _binance()
    lim.exchange = SimpleNamespace(
        last_response_headers={
            "x-mbx-order-count-1m": "900",
            "x-mbx-order-count-10s": "200",
        }
    )
    lim._on_call_success(Endpoints.PLACE_ORDER)
    assert lim.current_limit_state.order_limit_per_minute == 900
    assert lim.current_limit_state.order_limit_per_10_sec == 200


def test_junk_header_is_ignored_rather_than_crashing():
    lim = _binance()
    lim.current_limit_state.order_limit_per_minute = 42
    lim.current_limit_state.request_weight_limit_per_minute = 7
    lim.exchange = SimpleNamespace(
        last_response_headers={"x-mbx-order-count-1m": "not-a-number"}
    )
    lim._on_call_success(Endpoints.PLACE_ORDER)
    assert lim.current_limit_state.order_limit_per_minute == 42
    assert lim.current_limit_state.request_weight_limit_per_minute == 7


# --- Binance: weight accounting -------------------------------------------


def test_orderbook_read_is_charged_what_binance_charges():
    """
    /fapi/v1/depth is charged for the depth the OMS actually requests. guard()/
    reserve() thread no parameters, so an unparameterised depth read must resolve
    to OMS_DEPTH_LIMIT — the one limit OrderUtils.get_order_book ever sends.

    Both assertions matter: the literal pins the tier (2, not the 10 the 500-level
    default costs, and not the 1 it was once charged), and the derived one stops
    the request limit and the charge drifting apart in silence. Changing
    OMS_DEPTH_LIMIT without moving the request is the undercount that gets the
    IP banned.
    """
    lim = _binance()
    weight, orders = lim.find_cost_info(Endpoints.GET_ORDERBOOK_SNAPSHOT)
    assert (weight, orders) == (2, 0)
    assert weight == get_depth_weight(OMS_DEPTH_LIMIT)


def test_an_explicit_depth_limit_still_wins_over_the_oms_default():
    """A caller that threads its own limit is charged for that limit, not ours."""
    lim = _binance()
    assert lim.find_cost_info(Endpoints.GET_ORDERBOOK_SNAPSHOT, limit=1000) == (20, 0)
    assert lim.find_cost_info(Endpoints.GET_ORDERBOOK_SNAPSHOT, depth=500) == (10, 0)


def test_record_usage_charges_the_dynamic_weight_not_the_marker():
    """
    The dynamic marker is -1. record_usage() never resolved it, so a depth read
    used to *credit* a weight back and the counter drifted downward forever.
    """
    lim = _binance()
    lim.record_usage(Endpoints.GET_ORDERBOOK_SNAPSHOT)
    assert lim.current_limit_state.request_weight_limit_per_minute == get_depth_weight(
        OMS_DEPTH_LIMIT
    )
    assert lim.current_limit_state.request_weight_limit_per_minute == 2


def test_depth_weight_tiers_match_the_published_table():
    # Source: Binance USDⓈ-M /fapi/v1/depth
    assert [get_depth_weight(n) for n in (5, 50, 100, 500, 1000)] == [2, 2, 5, 10, 20]


def test_exchange_info_is_charged_its_real_weight():
    lim = _binance()
    assert lim.find_cost_info(Endpoints.GET_SYMBOL_INFO) == (1, 0)


# --- IP-scoped budgets are shared, account-scoped ones are not -------------


# Binance USDⓈ-M values, as /fapi/v1/exchangeInfo reports them.
BINANCE_EXCHANGE_INFO = {
    "rateLimits": [
        {
            "rateLimitType": "REQUEST_WEIGHT",
            "interval": "MINUTE",
            "intervalNum": 1,
            "limit": 2400,
        },
        {
            "rateLimitType": "ORDERS",
            "interval": "MINUTE",
            "intervalNum": 1,
            "limit": 1200,
        },
        {
            "rateLimitType": "ORDERS",
            "interval": "SECOND",
            "intervalNum": 10,
            "limit": 300,
        },
    ]
}


def _binance_real_init(tenants: int) -> BinanceRateLimiter:
    """
    Drive the real BinanceRateLimiter.__init__ so the budget derivation itself
    is under test, rather than a copy of its arithmetic.
    """
    client = BinanceLinearClient(api_key="k", api_secret="s")  # no IO on construct
    client.exchange_info = BINANCE_EXCHANGE_INFO
    config = SimpleNamespace(
        config=SimpleNamespace(
            soft_limit_percent=Decimal("0.8"), tenants_per_egress_ip=tenants
        ),
        exchange=client,
    )
    return BinanceRateLimiter(config)  # type: ignore[arg-type]


def test_binance_weight_is_split_across_shard_tenants_but_orders_are_not():
    """
    REQUEST_WEIGHT is per IP; order counts are per account. A shard is one NAT
    IP with up to 14 tenants, so three tenants each claiming 1920 weight/min
    admitted 2.4x what the IP allows and they banned each other.
    """
    solo = _binance_real_init(1)
    shared = _binance_real_init(14)

    assert solo.limit_profile.request_weight_limit_per_minute == 1920
    assert shared.limit_profile.request_weight_limit_per_minute == 137
    # 14 tenants at 137 each stays inside the real 2400/min IP ceiling
    assert shared.limit_profile.request_weight_limit_per_minute * 14 <= 2400
    # The old behaviour did not, by a factor of 11
    assert solo.limit_profile.request_weight_limit_per_minute * 14 > 2400

    # Order budgets are per API key, so they are identical either way
    assert (
        solo.limit_profile.order_limit_per_minute
        == shared.limit_profile.order_limit_per_minute
        == 960
    )
    assert (
        solo.limit_profile.order_limit_per_10_sec
        == shared.limit_profile.order_limit_per_10_sec
        == 240
    )


def test_binance_budget_defaults_to_undivided():
    """Dedicated-tier tenants own their IP; the default must not throttle them."""
    lim = _binance_real_init(1)
    assert lim.limit_profile.request_weight_limit_per_minute == 1920


def test_no_ip_wide_budget_is_derived_at_all():
    """
    Splitting the budget is the whole of the co-tenant protection now. There is
    no second, undivided IP ceiling, because the header it would be judged
    against does not report this IP's spend.
    """
    lim = _binance_real_init(14)
    assert lim.limit_profile.request_weight_limit_per_minute == 137  # 2400*0.8/14
    assert not hasattr(lim, "_ip_weight_limit_per_minute")
    assert not hasattr(lim, "_ip_wide_used_weight")


def test_a_busy_shard_does_not_stall_every_tenant_on_it():
    """
    End to end through the real constructor: 14 tenants, a 137 share each, and
    Binance answering with a large weight figure. Reads must still be allowed —
    that number is not this IP's spend, and refusing on it stalled the OMS.
    """
    lim = _binance_real_init(14)
    # Pin the clock (init() is what normally seeds these) so a real minute
    # boundary cannot roll mid-test.
    lim.get_synced_time_ms = lambda: NOW_MS
    lim.last_reset_1m_timestamp = NOW_MS // 60_000
    lim.last_reset_10s_timestamp = NOW_MS // 10_000
    for reading in ("200", "1918", "4433"):
        lim.exchange = SimpleNamespace(
            last_response_headers={"x-mbx-used-weight-1m": reading}
        )
        lim._on_call_success(Endpoints.GET_POSITION)
        assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


# --- usage headers are named after the limiters they report -----------------


def test_usage_header_names_are_derived_from_the_advertised_rate_limits():
    """
    Binance names each header after the limiter it belongs to --
    X-MBX-USED-WEIGHT-(intervalNum)(intervalLetter), "for all request rate
    limiters defined" -- so the names have to come from the same rateLimits rules
    the budgets come from. Hard-coding them is two copies of one assumption, and
    a header that stops matching fails silently: reconciliation just no-ops and
    the local estimate is trusted again.

    USDⓈ-M advertises three rules today (verified against live
    /fapi/v1/exchangeInfo): REQUEST_WEIGHT 1/MINUTE, ORDERS 1/MINUTE, ORDERS
    10/SECOND. Only the order ones are reconciled — REQUEST_WEIGHT is
    deliberately left out, because x-mbx-used-weight-1m does not report this
    egress IP's spend and refusing on it stalled the OMS.
    """
    lim = _binance_real_init(1)
    assert lim._usage_headers == {
        "x-mbx-order-count-1m": "order_limit_per_minute",
        "x-mbx-order-count-10s": "order_limit_per_10_sec",
    }
    assert "x-mbx-used-weight-1m" not in lim._usage_headers


def test_usage_header_templating():
    from adrs.oms.rate_limit.rate_limiter import _binance_usage_header as h

    assert h("x-mbx-used-weight", "MINUTE", 1) == "x-mbx-used-weight-1m"
    assert h("x-mbx-order-count", "SECOND", 10) == "x-mbx-order-count-10s"
    assert h("x-mbx-used-weight", "DAY", 1) == "x-mbx-used-weight-1d"
    # An interval we have no letter for yields nothing, rather than a header name
    # that could never match a response
    assert h("x-mbx-used-weight", "FORTNIGHT", 1) is None


def test_missing_request_weight_rule_fails_at_startup():
    """
    A REQUEST_WEIGHT rule the parser does not recognise left the budget at 0, and
    _has_capacity then refuses every call — an OMS that starts, logs one warning
    and silently never trades. Far better to crash where the cause is named.
    """
    import pytest

    client = BinanceLinearClient(api_key="k", api_secret="s")
    client.exchange_info = {
        "rateLimits": [
            # REQUEST_WEIGHT at an interval this limiter does not handle
            {
                "rateLimitType": "REQUEST_WEIGHT",
                "interval": "MINUTE",
                "intervalNum": 5,
                "limit": 12_000,
            }
        ]
    }
    config = SimpleNamespace(
        config=SimpleNamespace(
            soft_limit_percent=Decimal("0.8"), tenants_per_egress_ip=1
        ),
        exchange=client,
    )
    with pytest.raises(Exception, match="REQUEST_WEIGHT"):
        BinanceRateLimiter(config)  # type: ignore[arg-type]


def test_bybit_ip_pool_is_split_but_uid_pools_are_not():
    solo = BybitLimitProfile.with_buffer(
        buffer_pct=Decimal("0.2"), tenants_per_egress_ip=1
    )
    shared = BybitLimitProfile.with_buffer(
        buffer_pct=Decimal("0.2"), tenants_per_egress_ip=14
    )
    ip = BybitRateLimitPool.IP_GLOBAL
    assert solo.limits[ip] == 96  # 120 * 0.8
    assert shared.limits[ip] == 6  # and 14 * 6 <= 120
    assert shared.limits[ip] * 14 <= 120
    # UID pools are per API key
    for pool in BybitRateLimitPool:
        if pool is not ip:
            assert solo.limits[pool] == shared.limits[pool]


def test_budget_split_never_starves_a_tenant_completely():
    profile = BybitLimitProfile.with_buffer(
        buffer_pct=Decimal("0.2"), tenants_per_egress_ip=10_000
    )
    assert all(limit >= 1 for limit in profile.limits.values())


# --- Bybit: the header casing bug -----------------------------------------


def test_bybit_uid_breach_waits_for_its_pool_not_ten_minutes():
    """
    The reset header arrives lowercased, so the literal match failed and every
    UID breach took the IP branch — blocking all trading for 10 minutes over a
    pool that refills in under a second.
    """
    lim = _bybit()
    reset_at = lim._now + 300
    lim._handle_call_error(
        FakeBybitError(
            retCode=10006,
            headers={"x-bapi-limit-reset-timestamp": str(reset_at)},
        )
    )
    assert lim.retry_after < lim._now + 1_000
    assert lim.retry_after >= reset_at


def test_bybit_ip_breach_still_blocks_for_ten_minutes():
    """No reset header at all is how Bybit reports the IP-level ban."""
    lim = _bybit()
    lim._handle_call_error(FakeBybitError(http_status=403))
    assert lim.retry_after >= lim._now + 10 * 60 * 1000


def test_bybit_stale_reset_timestamp_backs_off_one_window_only():
    lim = _bybit()
    lim._handle_call_error(
        FakeBybitError(
            retCode=10006,
            headers={"x-bapi-limit-reset-timestamp": str(lim._now - 5_000)},
        )
    )
    assert lim._now < lim.retry_after <= lim._now + 1_000


# --- Bybit: a UID pool must not be wedgeable at zero -----------------------


def _bybit_wallet_headers():
    """Bybit really does return these on /v5/account/wallet-balance (observed live)."""
    return {
        "x-bapi-limit": "50",
        "x-bapi-limit-status": "49",
        "x-bapi-limit-reset-timestamp": "9999999999999",
    }


def test_a_non_rate_limit_failure_does_not_permanently_wedge_a_uid_pool():
    """
    Reported by a user as "UID_WALLET: 0/None, never recovers".

    guard() decrements the pool BEFORE the call and only adopts the exchange's
    headers in the `else` branch, so any failure inside the body decrements
    without ever learning the real quota. With `limit`/`reset_ts` unset,
    _uid_pool_snapshot's `if state.reset_ts and now >= state.reset_ts` can never
    fire, so once remaining reaches 0 the pool refuses forever -- for the life of
    the process, even after the underlying call starts working again.

    The wallet pool was uniquely exposed because its guard wrapped a ClickHouse
    write (create_equity) as well as the exchange call, so an aegis outage -- not
    a Bybit problem at all -- was enough to walk it to zero one tick per minute.
    """
    lim = _bybit()
    lim.exchange = SimpleNamespace(last_response_headers=_bybit_wallet_headers())
    pool = BybitRateLimitPool.UID_WALLET
    ceiling = lim.limit_profile.limits[pool]

    async def _tick(body_raises: bool):
        lim._now += 1_100  # roll IP_GLOBAL's 1s window; isolate the UID pool
        try:
            async with lim.guard(endpoint=Endpoints.GET_WALLET_BALANCE):
                if body_raises:
                    raise RuntimeError("clickhouse insert failed")
        except RuntimeError:
            pass

    # Enough non-rate-limit failures to drain the pool several times over
    for _ in range(ceiling * 2 + 5):
        asyncio.run(_tick(body_raises=True))

    # It must still admit: nothing here was a rate-limit signal from Bybit.
    assert lim.check_limits(endpoint=Endpoints.GET_WALLET_BALANCE) is True


def test_a_wedged_pool_recovers_once_calls_succeed_again():
    """The user's actual symptom: it never came back."""
    lim = _bybit()
    lim.exchange = SimpleNamespace(last_response_headers=_bybit_wallet_headers())
    ceiling = lim.limit_profile.limits[BybitRateLimitPool.UID_WALLET]

    async def _fail():
        lim._now += 1_100
        try:
            async with lim.guard(endpoint=Endpoints.GET_WALLET_BALANCE):
                raise RuntimeError("clickhouse insert failed")
        except RuntimeError:
            pass

    async def _succeed():
        lim._now += 1_100
        async with lim.guard(endpoint=Endpoints.GET_WALLET_BALANCE):
            pass

    for _ in range(ceiling + 5):
        asyncio.run(_fail())
    asyncio.run(_succeed())  # would raise LocalRateLimitError while wedged

    state = lim.current_limit_state[BybitRateLimitPool.UID_WALLET]
    assert state.limit == 50  # header finally adopted
    assert lim.check_limits(endpoint=Endpoints.GET_WALLET_BALANCE) is True


def test_a_real_bybit_rate_limit_breach_is_still_respected():
    """
    The self-healing must not swallow a genuine breach: a 403/10006 carries a
    reset timestamp, and until that passes the pool stays shut.
    """
    lim = _bybit()
    future = lim._now + 5_000
    lim.local_cache_error(
        {
            "x-bapi-limit": "50",
            "x-bapi-limit-status": "0",
            "x-bapi-limit-reset-timestamp": str(future),
        }
    )
    assert lim.check_limits(endpoint=Endpoints.GET_WALLET_BALANCE) is False


# --- cooldown release: jitter + ramp ---------------------------------------


def test_cooldown_release_is_jittered_so_co_tenants_do_not_wake_together():
    """
    Every process sharing a rate-limit bucket parses the SAME deadline out of the
    same ban message, so without jitter they all resume on the same millisecond
    and hammer the bucket in lockstep. On Binance testnet the bucket is a shared
    CloudFront PoP, so the co-tenants are not even ours.

    Jitter must only ever DELAY: the existing contract is that a cooldown is
    never shortened.
    """
    deadline = NOW_MS + 60_000
    seen = set()
    for _ in range(40):
        lim = _binance()
        lim._arm_cooldown(deadline, reason="test")
        assert lim.retry_after >= deadline, "jitter must never shorten a cooldown"
        assert lim.retry_after <= deadline + _COOLDOWN_JITTER_MAX_MS
        seen.add(lim.retry_after)
    assert len(seen) > 1, "identical deadlines produced identical wake-ups: no jitter"


def test_jitter_still_never_shortens_an_existing_cooldown():
    lim = _binance()
    far = NOW_MS + 600_000
    lim._arm_cooldown(far, reason="418 ban")
    armed = lim.retry_after
    lim._arm_cooldown(NOW_MS + 1_000, reason="429 retry-after")  # nearer
    assert lim.retry_after == armed


def test_weight_budget_ramps_back_up_after_a_cooldown_rather_than_snapping():
    """
    Resuming at the full budget the instant a ban expires is what turns one ban
    into the next. The first window after release runs on a reduced ceiling that
    climbs back to normal.
    """
    lim = _binance(weight_limit=1000)
    lim._arm_cooldown(NOW_MS + 1_000, reason="test")
    released_at = lim.retry_after

    lim._now = released_at - 1
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is False  # still cooling

    # Just released: ceiling is throttled, not full
    lim._now = released_at + 1
    early = lim._effective_weight_ceiling()
    assert early < 1000

    # Part-way through the ramp it is higher, but still not full
    lim._now = released_at + _COOLDOWN_RAMP_MS // 2
    mid = lim._effective_weight_ceiling()
    assert early < mid < 1000

    # Past the ramp, back to the real budget
    lim._now = released_at + _COOLDOWN_RAMP_MS + 1
    assert lim._effective_weight_ceiling() == 1000


def test_the_ramp_actually_refuses_calls_the_full_budget_would_allow():
    lim = _binance(weight_limit=1000)
    lim._arm_cooldown(NOW_MS + 1_000, reason="test")
    lim._now = lim.retry_after + 1
    # Spend up to just under the throttled ceiling
    lim.current_limit_state.request_weight_limit_per_minute = (
        lim._effective_weight_ceiling()
    )
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is False
    # The same spend is fine once the ramp is over
    lim._now = lim.retry_after + _COOLDOWN_RAMP_MS + 1
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True


def test_a_limiter_that_never_cooled_down_is_never_ramped():
    lim = _binance(weight_limit=1000)
    assert lim.retry_after == 0
    assert lim._effective_weight_ceiling() == 1000

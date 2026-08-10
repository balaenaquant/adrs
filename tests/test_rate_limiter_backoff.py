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

from collections import deque
from decimal import Decimal
from types import SimpleNamespace

from cybotrade.binance import BinanceError, BinanceLinearClient
from cybotrade.bybit import BybitError

from adrs.oms.rate_limit.rate_limiter import (
    BinanceRateLimiter,
    BybitRateLimiter,
    _BLIND_COOLDOWN_MS,
)
from adrs.oms.rate_limit.exchange_limit_profiles import (
    BinanceLimitProfile,
    BybitLimitProfile,
    BybitLimitState,
    BybitRateLimitPool,
    Endpoints,
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
    """BinanceRateLimiter with __init__ skipped and a fixed clock."""
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
    # test_usage_header_names_are_derived_from_the_advertised_rate_limits
    lim._usage_headers = {
        "x-mbx-used-weight-1m": "request_weight_limit_per_minute",
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
    # ...and not wildly beyond it
    assert lim.retry_after <= BAN_UNTIL_MS + 5_000


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


def test_used_weight_header_overrides_the_local_estimate():
    """
    x-mbx-used-weight-1m is metered per source IP, so it already includes every
    co-tenant behind the same NAT gateway — the traffic a per-process tally
    cannot see. Adopting it is what lets one process notice the shard is hot.
    """
    lim = _binance()
    # This process's own tally says it has spent nothing, so the read is allowed
    assert lim.current_limit_state.request_weight_limit_per_minute == 0
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is True

    # ...but Binance reports 1918 of the 1920 budget already gone, because
    # co-tenants behind this IP have been spending it.
    lim.exchange = SimpleNamespace(
        last_response_headers={"x-mbx-used-weight-1m": "1918"}
    )
    lim._on_call_success(Endpoints.GET_POSITION)
    assert lim.current_limit_state.request_weight_limit_per_minute == 1918
    # A weight-5 position read no longer fits, and is now correctly refused
    assert lim.check_limits(endpoint=Endpoints.GET_POSITION) is False


def test_reconcile_keeps_in_flight_reservations():
    """
    record_usage() charges before the request is sent, but the header only counts
    requests Binance has already processed. Overwriting would refund every
    in-flight reservation, so the higher of the two wins.
    """
    lim = _binance()
    lim.current_limit_state.request_weight_limit_per_minute = 500
    lim.exchange = SimpleNamespace(last_response_headers={"x-mbx-used-weight-1m": "20"})
    lim._on_call_success(Endpoints.GET_POSITION)
    assert lim.current_limit_state.request_weight_limit_per_minute == 500


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
    lim.current_limit_state.request_weight_limit_per_minute = 42
    lim.exchange = SimpleNamespace(
        last_response_headers={"x-mbx-used-weight-1m": "not-a-number"}
    )
    lim._on_call_success(Endpoints.GET_POSITION)
    assert lim.current_limit_state.request_weight_limit_per_minute == 42


# --- Binance: weight accounting -------------------------------------------


def test_orderbook_read_is_charged_what_binance_charges():
    """
    /fapi/v1/depth at the limit cybotrade actually causes (500) costs weight 10,
    not the 1 it was charged. get_current_price resolves to this call, so the
    undercount was 10x on the busiest read path in the OMS.
    """
    lim = _binance()
    weight, orders = lim.find_cost_info(Endpoints.GET_ORDERBOOK_SNAPSHOT)
    assert (weight, orders) == (10, 0)


def test_record_usage_charges_the_dynamic_weight_not_the_marker():
    """
    The dynamic marker is -1. record_usage() never resolved it, so a depth read
    used to *credit* a weight back and the counter drifted downward forever.
    """
    lim = _binance()
    lim.record_usage(Endpoints.GET_ORDERBOOK_SNAPSHOT)
    assert lim.current_limit_state.request_weight_limit_per_minute == 10


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


# --- usage headers are named after the limiters they report -----------------


def test_usage_header_names_are_derived_from_the_advertised_rate_limits():
    """
    Binance names each header after the limiter it belongs to --
    X-MBX-USED-WEIGHT-(intervalNum)(intervalLetter), "for all request rate
    limiters defined" -- so the names have to come from the same rateLimits rules
    the budgets come from. Hard-coding them is two copies of one assumption, and
    a header that stops matching fails silently: reconciliation just no-ops and
    the local estimate is trusted again.

    These are the three USDⓈ-M advertises today (verified against live
    /fapi/v1/exchangeInfo): REQUEST_WEIGHT 1/MINUTE, ORDERS 1/MINUTE, ORDERS
    10/SECOND.
    """
    lim = _binance_real_init(1)
    assert lim._usage_headers == {
        "x-mbx-used-weight-1m": "request_weight_limit_per_minute",
        "x-mbx-order-count-1m": "order_limit_per_minute",
        "x-mbx-order-count-10s": "order_limit_per_10_sec",
    }


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

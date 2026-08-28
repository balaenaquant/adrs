import pytest

from adrs.oms.rate_limit.exchange_limit_profiles import (
    Endpoints,
    HYPERLIQUID_ADDRESS_ACTION_BUFFER,
    HYPERLIQUID_COSTS,
    HYPERLIQUID_IP_WEIGHT_PER_MINUTE,
    HYPERLIQUID_RATE_LIMIT_COOLDOWN_MS,
    HYPERLIQUID_WEIGHT_WINDOW_SEC,
    HyperliquidLimitProfile,
    HyperliquidRateLimitPool,
    exchange_request_weight,
)


def test_documented_constants():
    assert HYPERLIQUID_IP_WEIGHT_PER_MINUTE == 1200
    assert HYPERLIQUID_WEIGHT_WINDOW_SEC == 60
    assert HYPERLIQUID_ADDRESS_ACTION_BUFFER == 10_000


def test_cooldown_outlasts_a_whole_weight_window():
    """
    The rate-limit signal does not say which axis fired. An address throttle
    clears in 10s, but an overrun of the 1200/min IP weight budget needs the
    whole 60s window to drain, and resuming mid-ban renews it -- for every
    tenant on the shared egress IP, not just this process. So the cooldown must
    cover the longer case: one full window plus a margin, the same 65s the base
    class uses for its blind fallback.
    """
    assert HYPERLIQUID_RATE_LIMIT_COOLDOWN_MS == 65_000
    assert HYPERLIQUID_RATE_LIMIT_COOLDOWN_MS > HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000


def test_every_endpoint_has_a_cost():
    """
    A missing entry would be charged as zero weight, which is an undercount --
    the failure mode that gets an IP banned. Every Endpoints member must be
    priced explicitly, even the ones Hyperliquid does not have.
    """
    assert set(HYPERLIQUID_COSTS) == set(Endpoints)
    for endpoint, cost in HYPERLIQUID_COSTS.items():
        assert set(cost) == {"weight", "actions"}, endpoint
        assert cost["weight"] >= 0
        assert cost["actions"] >= 0


@pytest.mark.parametrize(
    "endpoint,weight,actions",
    [
        # clearinghouseState / l2Book are the cheap tier at 2
        (Endpoints.GET_ORDERBOOK_SNAPSHOT, 2, 0),
        (Endpoints.GET_WALLET_BALANCE, 2, 0),
        (Endpoints.GET_POSITION, 2, 0),
        # 20, not the cheap tier's 2, because both of adrs's guarded call sites
        # (oms.py, ops/order_placement_manager.py) call
        # get_order_details_from_history -- `historicalOrders`, weight 20 --
        # and never get_order_details (`orderStatus`, weight 2). The entry is
        # priced for the call that is actually made. Endpoints is shared with
        # the other exchanges, so splitting the two paths into separate members
        # is out of scope; overcharging the unused cheap path is harmless, while
        # a 10x undercount on a repeated reconciliation loop bans the IP.
        (Endpoints.GET_ORDER_DETAILS, 20, 0),
        # metaAndAssetCtxs and frontendOpenOrders are not in the cheap tier
        (Endpoints.GET_SYMBOL_INFO, 20, 0),
        (Endpoints.GET_OPEN_ORDERS, 20, 0),
        (Endpoints.GET_OPEN_ORDERS_ALL, 20, 0),
        # POST /exchange is weight 1 for a single-order batch, and one action
        (Endpoints.PLACE_ORDER, 1, 1),
        (Endpoints.CANCEL_ORDER, 1, 1),
        # Hyperliquid has no server-time endpoint; the limiter uses the local clock
        (Endpoints.GET_SERVER_TIME, 0, 0),
    ],
)
def test_costs(endpoint, weight, actions):
    assert HYPERLIQUID_COSTS[endpoint] == {"weight": weight, "actions": actions}


def test_only_exchange_endpoints_cost_an_address_action():
    charged = {e for e, c in HYPERLIQUID_COSTS.items() if c["actions"] > 0}
    assert charged == {Endpoints.PLACE_ORDER, Endpoints.CANCEL_ORDER}


@pytest.mark.parametrize(
    "batch_length,expected",
    [(1, 1), (39, 1), (40, 2), (79, 2), (80, 3), (400, 11)],
)
def test_exchange_request_weight_formula(batch_length, expected):
    """Hyperliquid charges 1 + floor(batch_length / 40) for POST /exchange."""
    assert exchange_request_weight(batch_length) == expected


def test_exchange_request_weight_defaults_to_a_single_order():
    assert exchange_request_weight() == 1


def test_single_pool():
    """Every Hyperliquid call contends for the one aggregated weight window."""
    assert [p.name for p in HyperliquidRateLimitPool] == ["IP_WEIGHT"]


def test_profile_is_constructible():
    profile = HyperliquidLimitProfile(
        request_weight_limit_per_minute=960,
        address_action_buffer=HYPERLIQUID_ADDRESS_ACTION_BUFFER,
    )
    assert profile.request_weight_limit_per_minute == 960
    assert profile.address_action_buffer == 10_000

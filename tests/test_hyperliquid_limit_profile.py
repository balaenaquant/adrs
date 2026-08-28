import pytest

from adrs.oms.rate_limit.exchange_limit_profiles import (
    Endpoints,
    HYPERLIQUID_ADDRESS_ACTION_BUFFER,
    HYPERLIQUID_COSTS,
    HYPERLIQUID_IP_WEIGHT_PER_MINUTE,
    HYPERLIQUID_WEIGHT_WINDOW_SEC,
    HyperliquidLimitProfile,
    HyperliquidRateLimitPool,
    exchange_request_weight,
)


def test_documented_constants():
    assert HYPERLIQUID_IP_WEIGHT_PER_MINUTE == 1200
    assert HYPERLIQUID_WEIGHT_WINDOW_SEC == 60
    assert HYPERLIQUID_ADDRESS_ACTION_BUFFER == 10_000


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
        # clearinghouseState / l2Book / orderStatus are the cheap tier at 2
        (Endpoints.GET_ORDERBOOK_SNAPSHOT, 2, 0),
        (Endpoints.GET_ORDER_DETAILS, 2, 0),
        (Endpoints.GET_WALLET_BALANCE, 2, 0),
        (Endpoints.GET_POSITION, 2, 0),
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

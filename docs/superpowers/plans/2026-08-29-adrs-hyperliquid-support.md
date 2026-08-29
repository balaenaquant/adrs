# Hyperliquid Support in adrs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the adrs OMS trade Hyperliquid perpetuals at production parity with Bybit and Binance — client, private stream, public top-of-book feed, rate limiting and error policy.

**Architecture:** Hyperliquid is wired into the existing `Credentials` dispatch in `adrs/oms/config.py`. A new `HyperliquidRateLimiter` models Hyperliquid's 1200/min IP weight budget exactly from a static cost table, in its own module rather than growing the 1211-line `rate_limiter.py`. Hyperliquid's second budget — a cumulative address allowance of one request per USDC traded — is handled reactively through `HyperliquidErrorPolicy` and only counted locally for observability.

**Tech Stack:** Python 3.14, pydantic, `cybotrade>=2.4.0` (`cybotrade.hyperliquid`), pytest + pytest-asyncio, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-29-adrs-hyperliquid-support-design.md`

## Global Constraints

- `cybotrade>=2.4.0` — `cybotrade.hyperliquid` does not exist below that. `uv.lock` currently pins 2.3.1.
- adrs's `uv-lock` pre-commit hook rewrites `uv.lock` on any dependency change, so `uv lock` must run before committing a pin bump or the commit fails.
- Hyperliquid IP weight limit: **1200 per minute**, aggregated across all calls.
- Weight per call: `l2Book` / `clearinghouseState` / `orderStatus` = **2**; other info calls (`metaAndAssetCtxs`, `frontendOpenOrders`) = **20**; `userRole` = **60**; `POST /exchange` = **`1 + floor(batch_length / 40)`**.
- Address budget: **1 request per 1 USDC traded cumulatively**, initial buffer **10,000** requests; exhausted → **one request per 10 seconds**.
- Budget derivation matches `BinanceRateLimiter`: `int(limit * soft_limit_percent / tenants_per_egress_ip)`. IP-scoped budgets are divided by tenants; account-scoped ones are not.
- `Credentials.api_key` carries the Hyperliquid **master account address** and is **required** — `to_exchange_event()` is synchronous and cannot await cybotrade's `userRole` resolution.
- Unknown exchange errors fall through to `ErrorAction.RETRY` (the existing `default_action`). Never guess a classification.
- Run tests from the repo root: `uv run pytest`.

## File Structure

| File | Responsibility |
|---|---|
| `adrs/oms/rate_limit/exchange_limit_profiles.py` | **Modify** — add `HYPERLIQUID_COSTS`, `HyperliquidLimitProfile`, `HyperliquidRateLimitPool`, weight constants, `exchange_request_weight()` |
| `adrs/oms/rate_limit/error_policy.py` | **Modify** — add `is_hyperliquid_rate_limit_error()`, `HYPERLIQUID_ERROR_ACTIONS`, `HyperliquidErrorPolicy` |
| `adrs/oms/rate_limit/hyperliquid_limiter.py` | **Create** — `HyperliquidRateLimiter` |
| `adrs/oms/config.py` | **Modify** — five `Exchange.HYPERLIQUID` dispatch cases |
| `pyproject.toml` + `uv.lock` | **Modify** — `cybotrade>=2.4.0` |
| `examples/run_oms.py` | **Modify** — show how to select the limiter |
| `tests/test_hyperliquid_limit_profile.py` | **Create** |
| `tests/test_hyperliquid_error_policy.py` | **Create** |
| `tests/test_hyperliquid_rate_limiter.py` | **Create** |
| `tests/test_hyperliquid_config.py` | **Create** |

`hyperliquid_limiter.py` imports from `exchange_limit_profiles` and `error_policy` and from `rate_limiter` (for the `RateLimiter` ABC and `LocalRateLimitError`); nothing imports back into it except `examples/`. The existing `BinanceRateLimiter` and `BybitRateLimiter` are not moved — unrelated churn.

---

### Task 1: Dependency bump, cost table and limit profile

**Files:**
- Modify: `pyproject.toml` (the `dependencies` list, `cybotrade>=2.3.1`)
- Modify: `uv.lock` (regenerated, not hand-edited)
- Modify: `adrs/oms/rate_limit/exchange_limit_profiles.py` (append after the Bybit section)
- Test: `tests/test_hyperliquid_limit_profile.py`

**Interfaces:**
- Consumes: `Endpoints` (existing enum in the same file).
- Produces:
  - `HYPERLIQUID_IP_WEIGHT_PER_MINUTE: int = 1200`
  - `HYPERLIQUID_WEIGHT_WINDOW_SEC: int = 60`
  - `HYPERLIQUID_ADDRESS_ACTION_BUFFER: int = 10_000`
  - `HYPERLIQUID_THROTTLE_COOLDOWN_MS: int = 10_000`
  - `exchange_request_weight(batch_length: int = 1) -> int`
  - `class HyperliquidRateLimitPool(Enum)` with a single member `IP_WEIGHT`
  - `HYPERLIQUID_COSTS: dict[Endpoints, dict[str, int]]` with keys `"weight"` and `"actions"`
  - `class HyperliquidLimitProfile(BaseModel)` with `request_weight_limit_per_minute: int` and `address_action_buffer: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyperliquid_limit_profile.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hyperliquid_limit_profile.py -v`
Expected: FAIL — `ImportError: cannot import name 'HYPERLIQUID_COSTS'`

- [ ] **Step 3: Bump the dependency and relock**

In `pyproject.toml`, change `"cybotrade>=2.3.1",` to `"cybotrade>=2.4.0",`. Then:

```bash
uv lock
uv sync
uv run python -c "from cybotrade.hyperliquid import HyperliquidClient; print('ok')"
```

Expected: prints `ok`. If it does not, the lock did not move to 2.4.0 — check `grep -A1 'name = "cybotrade"' uv.lock`.

- [ ] **Step 4: Write the implementation**

Append to `adrs/oms/rate_limit/exchange_limit_profiles.py`:

```python
# ---------------------------------------------------------------------------
# Hyperliquid
# ---------------------------------------------------------------------------

# Hyperliquid rate-limits on two independent axes. This module prices the
# first: an aggregated per-IP weight budget. The second -- a cumulative
# address allowance of one request per USDC traded, with an initial buffer --
# is not a rate and is handled reactively by HyperliquidErrorPolicy; see the
# design doc for why it is not modelled here.
HYPERLIQUID_IP_WEIGHT_PER_MINUTE = 1200
HYPERLIQUID_WEIGHT_WINDOW_SEC = 60

# Requests an address may make before it has traded anything. Not enforced --
# the limiter counts against it only to warn, because the true allowance grows
# with traded volume and is not observable without paying weight to poll it.
HYPERLIQUID_ADDRESS_ACTION_BUFFER = 10_000

# Hyperliquid throttles an address that exhausts its allowance to one request
# every 10 seconds, so that is the cooldown to arm on the signal.
HYPERLIQUID_THROTTLE_COOLDOWN_MS = 10_000

# Info calls in Hyperliquid's cheap tier (weight 2): l2Book, allMids,
# clearinghouseState, orderStatus, spotClearinghouseState, exchangeStatus.
# Everything else documented is 20, and userRole is 60.
_HL_CHEAP_INFO_WEIGHT = 2
_HL_INFO_WEIGHT = 20


def exchange_request_weight(batch_length: int = 1) -> int:
    """
    Weight Hyperliquid charges for one POST /exchange call.

    `1 + floor(batch_length / 40)`, so a single order costs 1 and batching is
    close to free. The OMS places one order per call today; the formula lives
    here so that batching later cannot silently undercount.
    """
    return 1 + batch_length // 40


class HyperliquidRateLimitPool(Enum):
    # Hyperliquid publishes one aggregated weight budget rather than
    # per-endpoint pools, so there is exactly one contended pool. The enum
    # exists so _pool_key() returns something self-describing in logs, matching
    # BybitRateLimitPool.
    IP_WEIGHT = auto()


# Weight is per IP; actions are per account address. Both are charged on the
# request whether or not it succeeds. Every Endpoints member is listed: a
# missing entry would be charged as zero, and undercounting is what gets an IP
# banned.
HYPERLIQUID_COSTS: dict[Endpoints, dict[str, int]] = {
    # metaAndAssetCtxs -- needed rather than plain `meta` because the
    # Hyperliquid tick size is derived from the current mark price. Weight 20,
    # and update_symbol_info() guards per symbol, so a 20-symbol refresh spends
    # 400 of the minute's 1200.
    Endpoints.GET_SYMBOL_INFO: {"weight": _HL_INFO_WEIGHT, "actions": 0},
    # l2Book, the cheap tier
    Endpoints.GET_ORDERBOOK_SNAPSHOT: {"weight": _HL_CHEAP_INFO_WEIGHT, "actions": 0},
    # POST /exchange, single-order batch
    Endpoints.PLACE_ORDER: {"weight": exchange_request_weight(), "actions": 1},
    Endpoints.CANCEL_ORDER: {"weight": exchange_request_weight(), "actions": 1},
    # orderStatus, the cheap tier
    Endpoints.GET_ORDER_DETAILS: {"weight": _HL_CHEAP_INFO_WEIGHT, "actions": 0},
    # both read clearinghouseState, the cheap tier
    Endpoints.GET_WALLET_BALANCE: {"weight": _HL_CHEAP_INFO_WEIGHT, "actions": 0},
    Endpoints.GET_POSITION: {"weight": _HL_CHEAP_INFO_WEIGHT, "actions": 0},
    # frontendOpenOrders is not in the cheap tier
    Endpoints.GET_OPEN_ORDERS: {"weight": _HL_INFO_WEIGHT, "actions": 0},
    Endpoints.GET_OPEN_ORDERS_ALL: {"weight": _HL_INFO_WEIGHT, "actions": 0},
    # Hyperliquid has no server-time endpoint. Nonces are client-generated and
    # the reference SDK uses the local clock, so nothing is ever requested here.
    Endpoints.GET_SERVER_TIME: {"weight": 0, "actions": 0},
}


class HyperliquidLimitProfile(BaseModel):
    request_weight_limit_per_minute: int
    address_action_buffer: int = HYPERLIQUID_ADDRESS_ACTION_BUFFER
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_hyperliquid_limit_profile.py -v`
Expected: PASS (all parametrised cases)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock adrs/oms/rate_limit/exchange_limit_profiles.py tests/test_hyperliquid_limit_profile.py
git commit -m "feat(hyperliquid): add rate limit cost table and profile"
```

---

### Task 2: `HyperliquidErrorPolicy`

**Files:**
- Modify: `adrs/oms/rate_limit/error_policy.py`
- Test: `tests/test_hyperliquid_error_policy.py`

**Interfaces:**
- Consumes: `ErrorAction`, `ExchangeErrorPolicy` (same file); `HyperliquidError` from `cybotrade.hyperliquid`.
- Produces:
  - `HYPERLIQUID_ERROR_ACTIONS: tuple[tuple[str, ErrorAction], ...]`
  - `is_hyperliquid_rate_limit_error(exc: Exception) -> bool`
  - `class HyperliquidErrorPolicy(ExchangeErrorPolicy)` with `classify(exc) -> ErrorAction`

Hyperliquid has no stable numeric error code — it reports failure in the body of an HTTP 200 — so classification is by case-insensitive substring, ordered, with the first match winning.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyperliquid_error_policy.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hyperliquid_error_policy.py -v`
Expected: FAIL — `ImportError: cannot import name 'HyperliquidErrorPolicy'`

- [ ] **Step 3: Write the implementation**

Add the import at the top of `adrs/oms/rate_limit/error_policy.py`, beside the existing cybotrade error imports:

```python
from cybotrade.hyperliquid import HyperliquidError
```

Then append:

```python
# Hyperliquid reports failure in the body of an HTTP 200 and offers no stable
# numeric code, so classification is by substring. Ordered: first match wins.
# A whitelist with an inert default -- anything unmatched keeps the legacy
# RETRY behaviour rather than being guessed at.
#
# Only the first entry is confirmed against the live exchange. The rest are the
# categories the OMS must not retry blindly; drop any that cannot be confirmed
# rather than keep a match that silently never fires.
HYPERLIQUID_ERROR_ACTIONS: tuple[tuple[str, ErrorAction], ...] = (
    # Verified live: returned when cancelling an order id that is already gone.
    # The caller wanted it gone, so this is success, not failure -- the same
    # reading as Bybit's 110001.
    ("was never placed, already canceled, or filled", ErrorAction.TERMINAL_SUCCESS),
    ("rate limit", ErrorAction.RATE_LIMITED),
    ("too many requests", ErrorAction.RATE_LIMITED),
    ("insufficient margin", ErrorAction.FATAL),
    ("too far", ErrorAction.FATAL),
    ("reduce only", ErrorAction.FATAL),
)


def _hyperliquid_action(exc: Exception) -> ErrorAction | None:
    """The mapped action for a HyperliquidError, or None if nothing matched."""
    if not isinstance(exc, HyperliquidError):
        return None
    message = (exc.message or "").lower()
    for needle, action in HYPERLIQUID_ERROR_ACTIONS:
        if needle in message:
            return action
    return None


def is_hyperliquid_rate_limit_error(exc: Exception) -> bool:
    """
    Whether this error is Hyperliquid throttling us.

    Used by HyperliquidRateLimiter to arm its cooldown, so it deliberately
    requires a HyperliquidError: an unrelated exception whose text happens to
    contain "rate limit" must not stall every call for ten seconds.
    """
    return _hyperliquid_action(exc) is ErrorAction.RATE_LIMITED


class HyperliquidErrorPolicy(ExchangeErrorPolicy):
    def classify(self, exc: Exception) -> ErrorAction:
        action = _hyperliquid_action(exc)
        return self.default_action if action is None else action
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hyperliquid_error_policy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add adrs/oms/rate_limit/error_policy.py tests/test_hyperliquid_error_policy.py
git commit -m "feat(hyperliquid): add error policy"
```

---

### Task 3: `HyperliquidRateLimiter` — construction and weight admission

**Files:**
- Create: `adrs/oms/rate_limit/hyperliquid_limiter.py`
- Test: `tests/test_hyperliquid_rate_limiter.py`

**Interfaces:**
- Consumes: `RateLimiter`, `LocalRateLimitError` from `adrs.oms.rate_limit.rate_limiter`; everything Task 1 produced; `HyperliquidClient` from `cybotrade.hyperliquid`.
- Produces:
  - `class HyperliquidRateLimiter(RateLimiter)` with `__init__(self, config: ConfigManager)`
  - attributes `limit_profile: HyperliquidLimitProfile`, `weight_window: deque[tuple[int, int]]`, `address_actions: int`
  - methods `init`, `get_synced_time_ms`, `guard`, `_pool_key`, `_has_capacity`, `_next_free_delay`, `on_resync_time`, `reset_limits`, `check_limits`, `record_usage`, `__repr__` (Task 4 adds `_handle_call_error` and `local_cache_error`)

Task 4 completes the class. To keep the module importable at every commit, `_handle_call_error` and `local_cache_error` are written in this task as minimal bodies and replaced in Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyperliquid_rate_limiter.py
import pytest
from decimal import Decimal
from types import SimpleNamespace

from cybotrade.hyperliquid import HyperliquidClient

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hyperliquid_rate_limiter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adrs.oms.rate_limit.hyperliquid_limiter'`

- [ ] **Step 3: Write the implementation**

```python
# adrs/oms/rate_limit/hyperliquid_limiter.py
"""
Rate limiting for Hyperliquid.

Hyperliquid rate-limits on two independent axes:

* an aggregated per-IP weight budget of 1200 per minute, which this limiter
  models exactly from a static cost table, and
* a cumulative per-address allowance of one request per USDC traded, with an
  initial buffer of 10,000 requests, which it does not.

The second is not a rate. It is a lifetime allowance that grows with traded
volume, so mirroring it would mean polling volume (paying weight for the
privilege) and acting on an estimate that is wrong either way: throttling when
there was headroom, or failing to protect when there was not. It is enormous
once trading -- $1M of volume earns 1M requests -- and its failure mode is a
clean, detectable throttle to one request per 10s. So it is handled reactively
via HyperliquidErrorPolicy, and counted here only to warn as the initial buffer
depletes. Counted for observability; never enforced as a limit we cannot see.
"""

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, TYPE_CHECKING

from cybotrade.hyperliquid import HyperliquidClient

from adrs.oms.rate_limit.error_policy import is_hyperliquid_rate_limit_error
from adrs.oms.rate_limit.exchange_limit_profiles import (
    Endpoints,
    HYPERLIQUID_ADDRESS_ACTION_BUFFER,
    HYPERLIQUID_COSTS,
    HYPERLIQUID_IP_WEIGHT_PER_MINUTE,
    HYPERLIQUID_THROTTLE_COOLDOWN_MS,
    HYPERLIQUID_WEIGHT_WINDOW_SEC,
    HyperliquidLimitProfile,
    HyperliquidRateLimitPool,
)
from adrs.oms.rate_limit.rate_limiter import LocalRateLimitError, RateLimiter

if TYPE_CHECKING:
    from adrs.oms.config import ConfigManager

logger = logging.getLogger(__name__)

# Fraction of the initial address buffer at which to start warning.
_ACTION_WARN_RATIO = 0.8


class HyperliquidRateLimiter(RateLimiter):
    def __init__(self, config: "ConfigManager"):
        if not isinstance(config.exchange, HyperliquidClient):
            raise Exception("Exchange mismatch with rate limiter")
        super().__init__(config)

        # Weight is per IP, so a shard behind one NAT address must divide it;
        # the address allowance is account-scoped and is not divided. This is
        # the distinction the RateLimiter base class documents.
        ceiling = int(
            HYPERLIQUID_IP_WEIGHT_PER_MINUTE
            * self.soft_limit_percentage
            / self.tenants_per_egress_ip
        )
        self.limit_profile = HyperliquidLimitProfile(
            request_weight_limit_per_minute=ceiling,
            address_action_buffer=HYPERLIQUID_ADDRESS_ACTION_BUFFER,
        )
        # (epoch_ms, weight) charged, trimmed to the trailing window
        self.weight_window: deque[tuple[int, int]] = deque()
        self.address_actions = 0
        self._warned_actions = False

        logger.info(
            f"[HYPERLIQUID_LIMITS] Weight(1m) {ceiling} (IP-scoped, split "
            f"{self.tenants_per_egress_ip} ways). Address actions counted "
            f"against a {HYPERLIQUID_ADDRESS_ACTION_BUFFER} buffer, not enforced."
        )

    # ---- clock ---------------------------------------------------------

    async def init(self):
        """
        Nothing to initialise.

        Hyperliquid publishes no server-time endpoint and nonces are
        client-generated, so there is no clock to sync against.
        """
        return None

    def get_synced_time_ms(self) -> int:
        """Local clock; see init() for why there is nothing to sync to."""
        return int(time.time() * 1000)

    async def on_resync_time(self):
        """No-op, for the same reason as init()."""
        return None

    # ---- window --------------------------------------------------------

    def current_weight(self) -> int:
        """Weight charged inside the trailing window, after trimming."""
        self.reset_limits()
        return sum(weight for _, weight in self.weight_window)

    def _cost(self, endpoint: Endpoints) -> dict[str, int]:
        # KeyError rather than a zero default: an unpriced endpoint would be
        # charged nothing, and undercounting is what gets an IP banned.
        return HYPERLIQUID_COSTS[endpoint]

    def reset_limits(self):
        """Drop entries that have aged out of the trailing window."""
        cutoff = self.get_synced_time_ms() - HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000
        while self.weight_window and self.weight_window[0][0] <= cutoff:
            self.weight_window.popleft()

    def _pool_key(self, endpoint: Endpoints) -> Any:
        # One aggregated budget, so every call contends for the same pool.
        return HyperliquidRateLimitPool.IP_WEIGHT

    def _has_capacity(self, endpoint: Endpoints, **kwargs) -> bool:
        self.reset_limits()
        projected = sum(w for _, w in self.weight_window) + self._cost(endpoint)["weight"]
        ceiling = self.limit_profile.request_weight_limit_per_minute
        if projected > ceiling:
            logger.warning(
                f"[CHECK_LIMITS] Hyperliquid weight window is full: "
                f"{projected} > {ceiling}"
            )
            return False
        return True

    def _next_free_delay(self, endpoint: Endpoints) -> float:
        """Seconds until the oldest entry ages out and frees weight."""
        self.reset_limits()
        if self._has_capacity(endpoint):
            return 0.0
        if not self.weight_window:
            return 0.0
        oldest_ts = self.weight_window[0][0]
        expires_at = oldest_ts + HYPERLIQUID_WEIGHT_WINDOW_SEC * 1000
        return max(0.0, (expires_at - self.get_synced_time_ms()) / 1000.0)

    def check_limits(self, endpoint: Endpoints, **kwargs) -> bool:
        if self.retry_after > self.get_synced_time_ms():
            logger.warning(f"[CHECK_LIMITS] cooling down until {self.retry_after}")
            return False
        return self._has_capacity(endpoint, **kwargs)

    def record_usage(self, endpoint: Endpoints, **kwargs):
        cost = self._cost(endpoint)
        self.weight_window.append((self.get_synced_time_ms(), cost["weight"]))
        if cost["actions"]:
            self.address_actions += cost["actions"]
            self._maybe_warn_actions()

    def _maybe_warn_actions(self) -> None:
        buffer = self.limit_profile.address_action_buffer
        if self._warned_actions or self.address_actions < buffer * _ACTION_WARN_RATIO:
            return
        self._warned_actions = True
        logger.warning(
            f"[HYPERLIQUID_LIMITS] {self.address_actions} exchange actions sent "
            f"against an initial address buffer of {buffer}. The real allowance "
            f"grows with traded volume and is not observable here; if the "
            f"exchange starts throttling, that is why."
        )

    # ---- guard ---------------------------------------------------------

    @asynccontextmanager
    async def guard(self, endpoint: Endpoints) -> AsyncGenerator[None, None]:
        if not self.check_limits(endpoint=endpoint):
            raise LocalRateLimitError(f"Failed due to rate limits {self}")

        self.record_usage(endpoint=endpoint)
        try:
            async with asyncio.timeout(self.call_timeout_sec):
                yield
        except BaseException as e:
            self._handle_call_error(e, endpoint)
            raise e
        else:
            self._on_call_success(endpoint)

    # ---- error handling: completed in the next task ---------------------

    def _handle_call_error(
        self, e: BaseException, endpoint: Endpoints | None = None
    ) -> None:
        return None

    def local_cache_error(self, headers: dict[str, Any], **kwargs: Any) -> None:
        return None

    # ---- repr ----------------------------------------------------------

    def __repr__(self) -> str:
        retry = ""
        if self.retry_after > self.get_synced_time_ms():
            retry = f" [RETRYING_AFTER: {self.retry_after}]"
        return (
            f"HyperliquidRateLimiter(weight: {self.current_weight()}"
            f"/{self.limit_profile.request_weight_limit_per_minute}, "
            f"actions: {self.address_actions}"
            f"/{self.limit_profile.address_action_buffer}){retry}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hyperliquid_rate_limiter.py -v`
Expected: PASS. If `HyperliquidClient(...)` performs IO on construction the fixture will hang — it does not, but if that changes, replace the client with a `SimpleNamespace` and relax the `isinstance` guard test accordingly.

- [ ] **Step 5: Commit**

```bash
git add adrs/oms/rate_limit/hyperliquid_limiter.py tests/test_hyperliquid_rate_limiter.py
git commit -m "feat(hyperliquid): add rate limiter weight accounting"
```

---

### Task 4: `HyperliquidRateLimiter` — throttle detection and cooldown

**Files:**
- Modify: `adrs/oms/rate_limit/hyperliquid_limiter.py` (replace the two placeholder bodies from Task 3)
- Test: `tests/test_hyperliquid_rate_limiter.py` (append)

**Interfaces:**
- Consumes: `is_hyperliquid_rate_limit_error` (Task 2), `HYPERLIQUID_THROTTLE_COOLDOWN_MS` (Task 1).
- Produces: working `_handle_call_error(e, endpoint)` and `local_cache_error(headers, **kwargs)`; `retry_after` armed on a throttle signal.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hyperliquid_rate_limiter.py`:

```python
from cybotrade.hyperliquid import HyperliquidError


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hyperliquid_rate_limiter.py -k cooldown -v`
Expected: FAIL — `assert 0 == 1700000000000 + 10000`, because `_handle_call_error` is still the Task 3 placeholder.

- [ ] **Step 3: Write the implementation**

Replace the two placeholder methods in `adrs/oms/rate_limit/hyperliquid_limiter.py`:

```python
    def _handle_call_error(
        self, e: BaseException, endpoint: Endpoints | None = None
    ) -> None:
        """
        Arm the local cooldown when Hyperliquid is throttling us.

        Only a Hyperliquid throttle signal counts. A timeout or an unrelated
        bug must not stall every call for ten seconds, which is why the check
        goes through is_hyperliquid_rate_limit_error rather than matching text
        on any exception.

        Unlike the Bybit limiter there is no optimistic pre-call decrement to
        refund: the weight window records what was actually sent, and a request
        that failed still cost its weight at the exchange.
        """
        if isinstance(e, Exception) and is_hyperliquid_rate_limit_error(e):
            self._arm_throttle_cooldown()

    def local_cache_error(self, headers: dict[str, Any], **kwargs: Any) -> None:
        """
        Fold a rate-limit failure into local state.

        Hyperliquid returns no rate-limit headers at all -- it reports failure
        in the body of an HTTP 200 -- so `headers` is always empty here and the
        signal comes from the message instead. Callers pass it as
        `message=...`.
        """
        message = str(kwargs.get("message") or "")
        if "rate limit" in message.lower() or "too many requests" in message.lower():
            self._arm_throttle_cooldown()

    def _arm_throttle_cooldown(self) -> None:
        deadline = self.get_synced_time_ms() + HYPERLIQUID_THROTTLE_COOLDOWN_MS
        self.retry_after = max(self.retry_after, deadline)
        logger.warning(
            f"[HYPERLIQUID_LIMITS] throttled by the exchange; holding calls "
            f"until {self.retry_after}. The address allowance is cumulative and "
            f"grows with traded volume, so this clears as volume accrues."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hyperliquid_rate_limiter.py -v`
Expected: PASS (Task 3 and Task 4 tests together)

- [ ] **Step 5: Commit**

```bash
git add adrs/oms/rate_limit/hyperliquid_limiter.py tests/test_hyperliquid_rate_limiter.py
git commit -m "feat(hyperliquid): detect exchange throttling and arm a cooldown"
```

---

### Task 5: `Credentials` dispatch

**Files:**
- Modify: `adrs/oms/config.py` (imports near line 16; the five `match` methods between lines 44 and 155)
- Test: `tests/test_hyperliquid_config.py`

**Interfaces:**
- Consumes: `HyperliquidErrorPolicy` (Task 2); `HyperliquidClient`, `HyperliquidPrivateWS`, `HyperliquidPublicWS` from `cybotrade.hyperliquid`.
- Produces: `Credentials` supporting `Exchange.HYPERLIQUID` across `to_exchange_client`, `to_exchange_event`, `to_public_exchange_event`, `to_exchange_topic`, `to_error_policy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hyperliquid_config.py
import pytest
from cybotrade.hyperliquid import (
    HyperliquidClient,
    HyperliquidPrivateWS,
    HyperliquidPublicWS,
)
from cybotrade.models import Exchange

from adrs.oms.config import Credentials
from adrs.oms.rate_limit.error_policy import HyperliquidErrorPolicy

KEY = "0x0123456789012345678901234567890123456789012345678901234567890123"
ADDRESS = "0x" + "42" * 20
VAULT = "0x" + "cd" * 20


def _creds(**overrides) -> Credentials:
    kwargs = dict(
        exchange=Exchange.HYPERLIQUID,
        api_key=ADDRESS,  # master account address
        api_secret=KEY,  # signing key
    )
    kwargs.update(overrides)
    return Credentials(**kwargs)  # type: ignore[arg-type]


def test_exchange_client():
    client = _creds().to_exchange_client()
    assert isinstance(client, HyperliquidClient)
    assert client.address == ADDRESS
    assert client.url == "https://api.hyperliquid.xyz"


def test_exchange_client_testnet():
    client = _creds(testnet=True).to_exchange_client()
    assert client.url == "https://api.hyperliquid-testnet.xyz"


def test_vault_address_comes_from_the_passphrase_field():
    client = _creds(api_passphrase=VAULT).to_exchange_client()
    assert client.vault_address == VAULT


def test_no_vault_by_default():
    assert _creds().to_exchange_client().vault_address is None


@pytest.mark.parametrize("bad", ["", None])
def test_missing_account_address_is_rejected(bad):
    """
    api_key must be the MASTER account address. cybotrade can resolve it from
    userRole, but that call is awaited and to_exchange_event() is synchronous,
    so there is nowhere to await it -- and putting the agent wallet's own
    address here makes every state read return empty while orders still place.
    """
    creds = _creds(api_key=bad or "")
    with pytest.raises(ValueError, match="master account address"):
        creds.to_exchange_client()
    with pytest.raises(ValueError, match="master account address"):
        creds.to_exchange_event()


def test_private_stream_uses_the_account_address():
    stream = _creds().to_exchange_event()
    assert isinstance(stream, HyperliquidPrivateWS)
    assert stream.address == ADDRESS
    assert "testnet" not in stream.url


def test_private_stream_testnet():
    assert "testnet" in _creds(testnet=True).to_exchange_event().url


def test_public_feed_is_wired():
    """
    Unlike Kucoin and EdgeX, Hyperliquid has a real top-of-book stream, so this
    must not return None -- polling REST would spend the per-IP weight budget
    on prices.
    """
    feed = _creds().to_public_exchange_event(["BTCUSDC", "ETHUSDC"])
    assert isinstance(feed, HyperliquidPublicWS)
    assert [str(s) for s in feed.symbols] == ["BTCUSDC", "ETHUSDC"]


def test_topic():
    assert _creds().to_exchange_topic() == "hyperliquid"


def test_error_policy():
    assert isinstance(_creds().to_error_policy(), HyperliquidErrorPolicy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hyperliquid_config.py -v`
Expected: FAIL — `Exception: Unsupported exchange Exchange.HYPERLIQUID`

- [ ] **Step 3: Add the imports**

In `adrs/oms/config.py`, after the `from cybotrade.edgex import ...` line:

```python
from cybotrade.hyperliquid import (
    HyperliquidClient,
    HyperliquidPrivateWS,
    HyperliquidPublicWS,
)
```

and add `HyperliquidErrorPolicy` to the existing `from adrs.oms.rate_limit.error_policy import (...)` block.

- [ ] **Step 4: Add the five dispatch cases**

Each goes immediately before the `case _:` of its method.

`to_exchange_client`:

```python
            case Exchange.HYPERLIQUID:
                # api_key is the MASTER account address, not a key. cybotrade
                # can resolve it from userRole, but that is an awaited call and
                # to_exchange_event() below is synchronous, so adrs requires it
                # up front -- as Kucoin requires api_passphrase. Putting the
                # agent wallet's own address here makes every state read return
                # empty while orders still place, so it is validated, not
                # defaulted.
                if not self.api_key:
                    raise ValueError(
                        "'api_key' must be the Hyperliquid master account address"
                    )
                return HyperliquidClient(
                    private_key=self.api_secret,
                    account_address=self.api_key,
                    vault_address=self.api_passphrase,
                    testnet=self.testnet,
                )
```

`to_exchange_event`:

```python
            case Exchange.HYPERLIQUID:
                # Takes an address, not credentials: Hyperliquid keys user
                # subscriptions on the address and that data is public.
                if not self.api_key:
                    raise ValueError(
                        "'api_key' must be the Hyperliquid master account address"
                    )
                return HyperliquidPrivateWS(
                    address=self.api_key, testnet=self.testnet
                )
```

`to_public_exchange_event`:

```python
            case Exchange.HYPERLIQUID:
                return HyperliquidPublicWS(
                    symbols=[Symbol(s) for s in symbols], testnet=self.testnet
                )
```

`to_exchange_topic`:

```python
            case Exchange.HYPERLIQUID:
                return "hyperliquid"
```

`to_error_policy`:

```python
            case Exchange.HYPERLIQUID:
                return HyperliquidErrorPolicy()
```

Check how the Bybit and Binance branches of `to_public_exchange_event` pass `symbols` before writing that case: if they hand the list straight through as `list[str]`, drop the `Symbol(...)` wrapping and pass `symbols` unchanged. The test asserts on `str(s)` so it passes either way, but the two branches must agree.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_hyperliquid_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add adrs/oms/config.py tests/test_hyperliquid_config.py
git commit -m "feat(hyperliquid): wire Hyperliquid into the credentials dispatch"
```

---

### Task 6: Example wiring and full-suite verification

**Files:**
- Modify: `examples/run_oms.py` (the limiter selection, around line 67)
- Test: the whole suite

**Interfaces:**
- Consumes: everything above.
- Produces: no new production API.

- [ ] **Step 1: Add the limiter option to the example**

`examples/run_oms.py` currently selects a limiter by commenting one out:

```python
    # rate_limiter = BinanceRateLimiter(config=config)
    rate_limiter = BybitRateLimiter(config=config)
```

Add the Hyperliquid option in the same style, with the import:

```python
from adrs.oms.rate_limit.hyperliquid_limiter import HyperliquidRateLimiter
```

```python
    # rate_limiter = BinanceRateLimiter(config=config)
    # rate_limiter = HyperliquidRateLimiter(config=config)
    rate_limiter = BybitRateLimiter(config=config)
```

A factory is deliberately out of scope; the limiter must match the configured exchange or its constructor raises `Exchange mismatch with rate limiter`.

- [ ] **Step 2: Verify the example still imports**

Run: `uv run python -c "import examples.run_oms"`
Expected: no output, exit 0. If the module executes on import rather than under `if __name__ == "__main__":`, skip this check and rely on Step 3.

- [ ] **Step 3: Run the Hyperliquid tests together**

Run: `uv run pytest tests/ -v -k hyperliquid`
Expected: all PASS across the four new test files.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest tests/ -q`
Expected: no new failures against the pre-change baseline. Capture the baseline first if unsure:

```bash
git stash list   # confirm nothing stashed
uv run pytest tests/ -q 2>&1 | tail -3
```

The cybotrade bump from 2.3.1 to 2.4.0 is additive (a new module plus one enum member), so existing tests should be unaffected. If any fail, check whether they assert on the membership of `Exchange` — a new enum member can break an exhaustiveness assertion.

- [ ] **Step 5: Run the linters the pre-commit hooks enforce**

Run: `uv run ruff check adrs tests && uv run ruff format --check adrs tests`
Expected: clean. The repo's pre-commit hooks run `ruff check`, `ruff format` and `uv-lock`, so a commit fails otherwise.

- [ ] **Step 6: Commit**

```bash
git add examples/run_oms.py
git commit -m "docs(hyperliquid): show the Hyperliquid limiter in the OMS example"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Dependency `cybotrade>=2.4.0` + relock | 1 |
| Cost table, weight constants, `1 + floor(batch/40)` | 1 |
| `HyperliquidLimitProfile`, single pool | 1 |
| Error policy, substring whitelist, inert default | 2 |
| `is_hyperliquid_rate_limit_error` helper | 2 |
| Budget derivation: soft limit, tenant split, address budget undivided | 3 |
| Rolling 60s weight window, capacity, `_next_free_delay` | 3 |
| Thirteen ABC methods | 3 (eleven) + 4 (two) |
| Local action counter and warning | 3 |
| Throttle detection, 10s cooldown, `local_cache_error` without headers | 4 |
| Five dispatch cases | 5 |
| `api_key` required and validated | 5 |
| Public bbo feed wired rather than `None` | 5 |
| Limiter in its own module | 3 |
| Testing list from the spec | 1–5 |
| Out-of-scope items (factory, Kucoin/EdgeX limiters, WS connection counting, address enforcement) | no tasks, as intended |

One spec deviation, deliberate: the spec classified `"was never placed, already canceled, or filled"` as "non-retryable". Reading `ErrorAction` showed the codebase already has `TERMINAL_SUCCESS` for exactly this case (Bybit's `110001`, "order not exists or too late to cancel"), so Task 2 uses that instead. `FATAL` would drop and log an order the OMS should treat as done.

**Type consistency:** `HYPERLIQUID_COSTS` uses keys `"weight"` and `"actions"` in Tasks 1, 3 and their tests. `exchange_request_weight()` is defined in Task 1 and used there in the cost table. `HyperliquidLimitProfile` fields `request_weight_limit_per_minute` / `address_action_buffer` are consistent across Tasks 1, 3, 4. `current_weight()` and `address_actions` are introduced in Task 3 and asserted in Tasks 3 and 4. `is_hyperliquid_rate_limit_error` is defined in Task 2 and consumed in Task 4. `_arm_throttle_cooldown` is private to Task 4 and used by both methods it adds. `HyperliquidErrorPolicy` is defined in Task 2 and dispatched in Task 5.

**Known interface risks, flagged where they occur rather than assumed away:**

- `Credentials` field types: `api_key: str` is non-optional in the model, so the `None` case in Task 5's parametrised test is coerced to `""` by the helper. If pydantic rejects `api_key=""` at construction rather than at dispatch, move the assertion to construction.
- `to_public_exchange_event` may hand `symbols` through as `list[str]` rather than `list[Symbol]`; Task 5 Step 4 says to match the neighbouring branches.
- `ConfigManager` is imported under `TYPE_CHECKING` in Task 3 to avoid a circular import with `adrs.oms.config`, which imports the error policy. If a cycle appears anyway, the limiter only needs `config.exchange` and `config.config`, so the annotation can be loosened to `Any`.

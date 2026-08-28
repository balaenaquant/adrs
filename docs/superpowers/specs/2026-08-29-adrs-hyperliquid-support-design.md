# Hyperliquid Support in adrs — Design

Date: 2026-08-29
Status: Approved for planning

## Goal

Let the adrs OMS trade Hyperliquid perpetuals, at production parity with the
Bybit and Binance integrations: exchange client, private order/position stream,
public top-of-book feed, rate limiting and error policy.

cybotrade 2.4.0 ships the adapter (`cybotrade.hyperliquid`); this work is the
consumer side.

## Context

adrs already dispatches per exchange from one place. `Credentials` in
`adrs/oms/config.py` has five `match self.exchange` methods:

| Method | Purpose | Fallback for an unknown exchange |
|---|---|---|
| `to_exchange_client()` | REST client | raises |
| `to_exchange_event()` | private WS | raises |
| `to_public_exchange_event(symbols)` | public top-of-book WS | returns `None` |
| `to_exchange_topic()` | topic string | raises |
| `to_error_policy()` | error policy | `DefaultErrorPolicy()` |

EdgeX is the precedent: another DEX already threaded through this seam.

What is *not* already solved: `RateLimiter` (`adrs/oms/rate_limit/rate_limiter.py`)
is an ABC with thirteen abstract methods and exactly two implementations,
`BinanceRateLimiter` and `BybitRateLimiter`. There is no no-op implementation and
no factory — the caller picks one by hand (`examples/run_oms.py` comments one out
to select the other). Kucoin and EdgeX are configurable but have **no** limiter,
so running them today means charging some other exchange's pools. Hyperliquid
will not inherit that.

## Decisions

| Decision | Choice |
|---|---|
| Scope | Full: dispatch + rate limiter + error policy |
| Credentials | Reuse existing fields, EdgeX-style (no model change); `api_key` carries the master account address and is required |
| Public price feed | Wire the real `HyperliquidPublicWS` |
| Rate limiting | Model the IP weight budget exactly; handle the address budget reactively (approach B) |
| Limiter location | New module, not an addition to the 1211-line `rate_limiter.py` |

## Hyperliquid's two budgets

Hyperliquid rate-limits on two independent axes.

**1. IP weight — 1200 per minute, aggregated.**

| Call | Weight |
|---|---|
| `l2Book`, `allMids`, `clearinghouseState`, `orderStatus`, `spotClearinghouseState`, `exchangeStatus` | 2 |
| every other documented info call (incl. `metaAndAssetCtxs`, `frontendOpenOrders`, `historicalOrders`) | 20 |
| `userRole` | 60 |
| `POST /exchange` | `1 + floor(batch_length / 40)` |

Some info calls add weight per page returned (per 20 items; `candleSnapshot` per
60). The OMS does not use those paginated endpoints, so the static table is
exact for its call set.

**2. Address budget — 1 request per 1 USDC traded cumulatively, with an initial
buffer of 10,000 requests.** Exhausted, the address is throttled to one request
every 10 seconds. Cancels get a larger allowance: `min(limit + 100000, limit * 2)`.

These are different in kind. The IP budget is a rolling one-minute window. The
address budget is a *cumulative lifetime allowance* that grows with traded volume
— not a rate, and not expressible in the pool model the existing limiters use.

### Why the address budget is handled reactively

Modelling it proactively would mean polling cumulative traded volume, paying
weight for the privilege, and acting on an estimate that is wrong in one of two
directions: throttle when there was headroom, or fail to protect when there was
not.

It is also enormous in practice — $1M of volume earns 1M requests — and its
failure mode is clean and detectable: the exchange starts refusing, and
`HyperliquidErrorPolicy` can back off on the signal. The initial 10,000-request
buffer is the only regime where exhaustion is plausible, so the limiter keeps a
local count of `/exchange` actions and logs a warning as that count approaches
the buffer. Counted for observability, not enforced as a limit we cannot see.

This is the honest boundary: model exactly what is known exactly, react to the
rest, and say which is which.

## Cost table

Added to `adrs/oms/rate_limit/exchange_limit_profiles.py` as
`HYPERLIQUID_COSTS`, in the shape the existing tables use
(`{"weight": int, "orders": int}`).

| `Endpoints` | Hyperliquid call | weight | address actions |
|---|---|---|---|
| `GET_SYMBOL_INFO` | `metaAndAssetCtxs` | 20 | 0 |
| `GET_ORDERBOOK_SNAPSHOT` | `l2Book` | 2 | 0 |
| `PLACE_ORDER` | `POST /exchange` `order` | 1 | 1 |
| `CANCEL_ORDER` | `POST /exchange` `cancel` | 1 | 1 |
| `GET_ORDER_DETAILS` | `orderStatus` | 2 | 0 |
| `GET_WALLET_BALANCE` | `clearinghouseState` | 2 | 0 |
| `GET_POSITION` | `clearinghouseState` | 2 | 0 |
| `GET_OPEN_ORDERS` | `frontendOpenOrders` | 20 | 0 |
| `GET_OPEN_ORDERS_ALL` | `frontendOpenOrders` | 20 | 0 |
| `GET_SERVER_TIME` | unused | 0 | 0 |

`GET_SYMBOL_INFO` and `GET_OPEN_ORDERS` are the expensive reads at weight 20.
`update_symbol_info()` takes a guard per symbol, so a 20-symbol refresh costs 400
weight — a third of the minute budget in one sweep. The plan must confirm the
per-symbol guard behaviour against this table rather than assume the Binance
comment still applies.

**Scoping.** The 1200/min is per IP, so it is divided by
`config.tenants_per_egress_ip`, exactly as `BinanceRateLimiter` divides its
weight budget. The address budget is account-scoped and is **not** divided. This
is the distinction the `RateLimiter` base class already documents.
`config.soft_limit_percent` applies to the weight ceiling as elsewhere.

## `HyperliquidRateLimiter`

New file `adrs/oms/rate_limit/hyperliquid_limiter.py`. `rate_limiter.py` is
already 1211 lines carrying two implementations; a third belongs beside it, not
inside it. The existing two are left alone — moving them is unrelated churn.

Implementation of the thirteen abstract methods:

| Method | Implementation |
|---|---|
| `init()` | no-op; there is no exchange-time sync to perform |
| `get_synced_time_ms()` | local `time.time() * 1000`. Hyperliquid nonces are client-generated and the reference SDK uses the local clock; there is no server-time endpoint to sync against |
| `guard(endpoint)` | inherited pattern: pre-check via `check_limits`, run the call, `record_usage` on success, `_handle_call_error` on failure |
| `_pool_key(endpoint)` | a single constant — every call contends for the one IP weight window, so there is one pool |
| `_has_capacity(endpoint)` | projected weight in the trailing 60s window + this call's weight ≤ ceiling |
| `_next_free_delay(endpoint)` | seconds until the oldest entry in the 60s window expires and frees enough weight |
| `_handle_call_error(e, endpoint)` | detect the address-throttle and HTTP 429 signals, arm `retry_after` |
| `on_resync_time()` | no-op, for the same reason as `init()` |
| `reset_limits()` | clear the weight window and the local action count |
| `check_limits(endpoint)` | `_has_capacity` plus the inherited `retry_after` and reserve-yield rules |
| `record_usage(endpoint)` | append `(timestamp, weight)` to the window; increment the action count for `/exchange` endpoints |
| `local_cache_error(headers, **kwargs)` | Hyperliquid sends no rate-limit headers, so this folds the error body's signal in instead — documented explicitly, since a silent no-op here would look like an oversight |
| `__repr__()` | current weight usage against the ceiling, and the action count against the 10,000 buffer |

State is a deque of `(epoch_ms, weight)` trimmed to the trailing 60 seconds —
the same rolling-window shape `BinanceRateLimiter` uses, with one pool instead of
several.

## Error policy

`HyperliquidErrorPolicy` in `adrs/oms/rate_limit/error_policy.py`, alongside
`BybitErrorPolicy` and `BinanceErrorPolicy`, returned from
`Credentials.to_error_policy()`.

Hyperliquid reports failure in the **body of an HTTP 200**, and a batch can fail
per order, so the policy classifies on `HyperliquidError.message` rather than on a
status code or a numeric error code. Unlike Bybit's `retCode` and Binance's
`code`, there is no stable numeric identifier — matching is by substring, which
is why the table below is a whitelist with an inert default rather than an
exhaustive mapping.

`HYPERLIQUID_ERROR_ACTIONS: tuple[tuple[str, ErrorAction], ...]`, matched
case-insensitively in order:

`ErrorAction` already has the four states needed — `TERMINAL_SUCCESS`, `RETRY`,
`RATE_LIMITED`, `FATAL` — so no new action is required.

| Substring | Action | Provenance |
|---|---|---|
| `"was never placed, already canceled, or filled"` | `TERMINAL_SUCCESS` | **verified live** — returned when cancelling an unknown order id |
| `"rate limit"` / `"too many requests"` | `RATE_LIMITED` | the address-budget throttle |
| `"insufficient margin"` | `FATAL` | margin rejection |
| `"too far"` | `FATAL` | oracle-band rejection |
| `"reduce only"` | `FATAL` | reduce-only violation |

The first row is `TERMINAL_SUCCESS`, not `FATAL`: the order is gone, which is
what the caller asked for. This is the same reading the codebase already applies
to Bybit's `110001` ("order not exists or too late to cancel"). Classifying it
`FATAL` would drop and log an order the OMS should count as done.

Anything unmatched falls through to the policy's default action — the current
retry-everything behaviour, unchanged. Nothing is guessed, and the same
whitelist-with-inert-default shape as `ORDER_STATUS_MAP` in the cybotrade adapter
and commit `6b4e986`.

Only the first row is confirmed against the live exchange. The rest are the
categories the OMS must not retry blindly; the plan verifies each string as it is
implemented and drops any that cannot be confirmed, rather than shipping a match
that silently never fires.

## Credentials mapping

No change to the `Credentials` model. Following EdgeX's precedent:

| `Credentials` field | Hyperliquid meaning |
|---|---|
| `api_secret` | the signing private key (agent/API wallet, or master key) |
| `api_key` | the **master account address** — required, not optional |
| `api_passphrase` | optional `vault_address` |
| `testnet` | selects the testnet endpoints |

```python
case Exchange.HYPERLIQUID:
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

**`api_key` is required, and that is forced by the interface rather than
preference.** cybotrade can resolve the master account itself when
`account_address` is omitted, but it does so with an `await`ed `userRole` call —
and `to_exchange_event()`, which needs the same address to construct
`HyperliquidPrivateWS`, is a synchronous method. There is nowhere to await. So
adrs requires the address up front and validates it, exactly as
`Exchange.KUCOIN_LINEAR` validates `api_passphrase`.

Two things follow from that, both good: the client and the private stream are
guaranteed to agree on which account they are talking about, and the weight-60
`userRole` lookup never fires from adrs at all.

**This mapping is a documented footgun and the docstring must say so.** The field
named `api_key` is not a key, and putting the *agent wallet's own* address there
rather than the master account's makes every state read return empty — no
balance, no positions, no open orders — while orders still place successfully. A
strategy would believe it was flat while holding a position. cybotrade 2.4.0
detects and corrects this case with a warning, but adrs should not rely on that
rescue.

## Dispatch cases

```python
# to_exchange_client:       HyperliquidClient(...)  -- see Credentials mapping
# to_exchange_event:        HyperliquidPrivateWS(address=self.api_key, testnet=self.testnet)
# to_public_exchange_event: HyperliquidPublicWS(symbols=symbols, testnet=self.testnet)
# to_exchange_topic:        "hyperliquid"
# to_error_policy:          HyperliquidErrorPolicy()
```

`HyperliquidPrivateWS` takes an **address, not credentials** — Hyperliquid keys
user subscriptions on the address and the data is public. It receives
`self.api_key`, the validated master account address, which is why that field is
mandatory (see Credentials mapping above).

`HyperliquidPublicWS` takes symbols verbatim and has no metadata to resolve
against, so a symbol must carry Hyperliquid's own casing — `kPEPEUSDC`, not
`KPEPEUSDC`. Seven assets are spelled with a lowercase k.

## The `userRole` blind spot, and why it stays closed

cybotrade's agent-address resolution calls `userRole`, which at **weight 60** is
the most expensive info call Hyperliquid offers. It fires inside cybotrade, so it
never passes through adrs's `guard()` and would be invisible to this limiter.

Requiring `api_key` closes it: with `account_address` always supplied, cybotrade
skips the lookup entirely and adrs's weight accounting stays complete. Recorded
here so that a future change relaxing `api_key` to optional is understood to
reopen an unaccounted 60-weight call, not merely to add a convenience.

## WebSocket budget

Hyperliquid allows 10 connections, 30 new connections per minute, 1000
subscriptions, 10 unique users across user-specific subscriptions, 2000 messages
per minute, and 100 simultaneous inflight post messages.

One OMS instance opens two connections (private + public) and a handful of
subscriptions, so there is generous headroom. It is recorded because the
10-connection and 10-unique-user ceilings are per IP and per account: a shard
running many OMS processes behind one egress IP can reach them, and nothing in
adrs currently counts connections.

Hyperliquid also caps open orders at 1000 by default (rising with volume, to
5000). The OMS ladders multiple price levels per symbol, so a wide portfolio can
approach this. Out of scope to enforce, in scope to document.

## Testing

Unit tests, no network, following the existing `tests/test_rate_limiter_backoff.py`
patterns:

- weight accounting per endpoint against the cost table, including the weight-20
  reads and the `1 + floor(batch/40)` exchange formula
- the rolling 60s window: capacity exhaustion, `_next_free_delay`, expiry
- ceiling divided by `tenants_per_egress_ip`, and `soft_limit_percent` applied
- `retry_after` armed from an address-throttle error, and cleared on expiry
- action counter increments only for `/exchange` endpoints, and warns near 10,000
- `HyperliquidErrorPolicy` classification for each enumerated message, and the
  inert default for an unknown one
- `Credentials` dispatch: all five methods for `Exchange.HYPERLIQUID`, including
  `api_key=""` producing `account_address=None`

An end-to-end live test is out of scope for this spec: cybotrade's own
`tests/test_hyperliquid.py` already covers the adapter against the exchange, and
the adapter's order lifecycle was validated on mainnet during that work.

## Dependency

`adrs/pyproject.toml`: `cybotrade>=2.4.0` (currently `>=2.3.1`, with
`uv.lock` pinning 2.3.1, so `cybotrade.hyperliquid` does not resolve yet).
`uv lock` must run after the pin bump; adrs's `uv-lock` pre-commit hook rewrites
the lockfile on any dependency change, so a stale lock fails the commit.

## Out of scope

- Enforcing the address budget or the open-order cap
- Counting websocket connections against Hyperliquid's per-IP ceiling
- Hyperliquid spot, trigger orders (TP/SL), vault-specific order routing beyond
  passing `vault_address` through
- A `RateLimiter` factory, or retrofitting limiters for Kucoin and EdgeX
- Refactoring `rate_limiter.py`

# Websocket Price Feed — Design

Date: 2026-08-11
Status: approved, not yet implemented

## Purpose

Serve top-of-book prices from a websocket feed instead of REST, so the OMS stops
spending per-IP request weight on prices that a stream gives away for free.

`/fapi/v1/depth` is the OMS's single largest consumer of Binance request weight.
It costs weight 10 per call and is read from four places, two of which sit on a
2-second cron:

| call site | trigger | cadence |
| --- | --- | --- |
| `OrderExecutor.get_current_price` | placement tick, signal recompute | 30s / on weight change |
| `place_multi_limit_order` | one shared read per placement batch | 30s |
| `place_single_limit_order` | **one read per backlog retry** | 2s |
| `on_order_expiry_check` | one read per symbol per firing tick | 2s |

Measured against `examples/example_config.json` (`order_placement_interval: 30`,
`expiry_check: 2`) with two symbols, a quiet OMS spends ~24 weight/min while one
in active churn runs ~300-900 weight/min. Depth reads are essentially the whole
difference. The retry path's ceiling on its own is 900 weight/min — 3 concurrent
retries (`MAX_CONCURRENT_BACKLOG_RETRIES`) x weight 10 x 30 ticks/min — reached
when the backlog has at least 3 due items on every tick, which a stream of
post-only rejections in a moving market produces.

That matters because REQUEST_WEIGHT is metered **per source IP** and a shard is
one NAT gateway shared by up to 14 tenants (`prime` `shards.ts`). At a 14-way
split each tenant's budget is ~137 weight/min, which an OMS in churn exceeds by
almost 7x on depth reads alone.

Removing them takes active churn to roughly ~50 weight/min, leaving only
open-orders, position and wallet polling.

## Scope

In scope, Binance USDⓈ-M only:

- A public market-data websocket adapter in `cybotrade`.
- A `PriceFeed` cache and staleness policy in `adrs`.
- All four read sites above, via the two functions they share.

Out of scope, deliberately:

- Bybit. Follows as its own change; the component boundaries below are chosen so
  it is an adapter addition rather than a redesign. See *Bybit readiness*.
- Position/account data over websocket. Separate work, tracked already.
- Anything about order placement or cancellation.

## Decisions

Settled during design, with the reasoning that is easy to lose later:

**Stale prices fall back to REST rather than skipping the tick.** A stale read
takes today's REST path, so the worst case is exactly the cost we already pay
and never a halted OMS. The fallback keeps its existing `reserve()` wrapper, so
it self-throttles under budget pressure instead of stampeding — no new
cross-process coordination.

**Freshness is per-symbol quote age, not connection state.** A socket can stay
open while data stops flowing, and a reconnect leaves a plausible-looking price
in memory. Age catches both; connection state catches neither.

**The feed subscribes to a stream that pushes on a timer, not on change.** This
is the crux. `@bookTicker` is event-driven, so silence is ambiguous: either the
book is quiet or the feed is dead. Resolving that ambiguity pessimistically
means an illiquid pair falls back to REST every tick and re-fetches a price that
has not changed — spending weight 10 to learn nothing, and losing the entire
saving on exactly the pairs where nothing is happening.

A timer-driven stream makes silence unambiguous, so a quiet book stays servable
and a dead feed is still caught within `max_age`. Binance's Partial Book Depth
(`<symbol>@depth5@100ms`, documented speeds 100ms/250ms/500ms) is per-symbol and
carries top-of-book directly, so it replaces `@bookTicker` rather than
supplementing it. **Whether it truly ticks while idle is unverified — see Open
questions.**

Websocket cadence costs no request weight: market streams are metered by
connection count, not weight, so a 100ms push and a 10s push cost the same zero.

**Ping/pong cannot serve as the liveness signal.** Binance sends a ping every 3
minutes with a 10-minute pong timeout. Three-minute resolution is useless as a
freshness guard.

**`max_age` is per-feed, not global.** Binance ~1s (10 missed 100ms ticks).
Bybit ~4s, because `orderbook.1` only *guarantees* a push every 3 seconds when a
book is quiet, so 1s there would fall back constantly for no safety gain.

**Sequence/ordering logic lives in the adapter, not in `PriceFeed`.** An earlier
draft had `PriceFeed` reject any update whose id was not greater than the stored
one. That is wrong for Bybit, whose `u` resets to 1 on service restart — the rule
would reject every message afterwards, causing permanent staleness. Bybit also
repeats the previous `u` on its 3-second no-change snapshot, which the rule would
also reject. Ordering is protocol semantics and belongs with the protocol.

**No config flag; this is the default path.** Every failure mode degrades to
REST, so the design is fail-safe by construction and the flag would only guard
against higher weight — which is where we already are. Rollback is revert and
redeploy; staging is by deployment scope (one tenant first, then widen) rather
than by config.

## Architecture

Three units, each with one responsibility.

```
Binance ──ws──> BinancePublicWS ──event──> PriceFeed ──quote──> OrderUtils.get_order_book
              (cybotrade: wire format)   (adrs: policy)        OrderExecutor.get_current_price
                                                                      │ None
                                                                      └──> REST via reserve()
```

### Components

**`BinancePublicWS(ExchangeEvent)` — cybotrade**

Mirrors the existing `BinancePrivateWS`: builds a `Request` for a combined
per-symbol stream, heartbeats with `Message.Ping()`, and `start()` iterates
`persist_conn`. Unlike the private WS it needs no listenKey, so it makes no REST
call at all — notably avoiding that class's blocking sync `requests.put` on every
heartbeat.

Its only job is wire format to typed event. It holds no cache and makes no
policy decisions. Malformed messages are logged and dropped, never raised, and
never invalidate good state.

Needs two additions to cybotrade's shared vocabulary:

- `EventType.BookTicker`
- a `BookTicker` model: `symbol, bid, ask, bid_qty, ask_qty, update_id, event_time`

The URL form needs confirming against the live endpoint: docs show both
`/ws/<stream>` and `/public/ws/<stream>` (and the `stream?streams=` combined
equivalents), while `BinancePrivateWS` uses the un-prefixed `/ws/` form.

**`PriceFeed` — adrs**

The policy unit. Holds `dict[Symbol, Quote]`, `Quote = (bid, ask, received_at)`.

- `apply(book_ticker)` — store, after rejecting a crossed or zero book
  (`bid >= ask`, or either side 0). A corrupt message must never become a quote.
- `get(symbol, max_age_sec) -> Quote | None` — `None` when unseen or older than
  `max_age_sec`, measured on `time.monotonic()`.
- `clear()` — drop everything. Fires on reconnect.
- `invalidate(symbol)` — drop one symbol. Never fires on Binance; exists so
  Bybit's per-symbol resync is an addition rather than a redesign.
- counters: quotes served, fallbacks taken, feed-down duration, event-time
  divergence.

It never calls the exchange and never decides what to do about a miss, which is
what makes it testable with nothing but an injected clock.

**Wiring — adrs**

All four read sites already funnel through two functions, so this is a
two-function change:

```python
# OrderUtils.get_order_book
quote = price_feed.get(pair)
if quote is not None:
    return [quote.bid, quote.ask]
async with rate_limiter.reserve(endpoint=Endpoints.GET_ORDERBOOK_SNAPSHOT):
    ...unchanged REST path...

# OrderExecutor.get_current_price
quote = price_feed.get(symbol)
if quote is not None:
    return (quote.bid + quote.ask) / Decimal("2.0")
...unchanged REST path...
```

The mid-price formula matches `cybotrade`'s `ExchangeClient.get_current_price`
exactly — mid of best bid and best ask — so the feed and REST paths cannot quote
differently for the same book.

## Data flow

Feed task starts alongside `exchange_events_task` in `run()`, cancelled on
shutdown. It starts cold: an empty cache and a stale cache take the same path, so
there is no startup special case.

`clear()` is triggered by reconnect. `persist_conn` re-runs `on_connected` on
every reconnect, so the adapter emits its subscribe frame and a `Subscribed`
event; the OMS handler treats that as "feed restarted" and clears. This single
rule is what makes the guard sound: after any gap every symbol falls back to REST
for its first quote, then rejoins the feed.

A single Binance connection is only valid for 24 hours, so this path runs daily
in normal operation, not only under failure.

Symbol-set changes: the stream URL encodes the symbol list, so
`on_refresh_config` detecting a `base_asset_to_symbol_table` change must tear
down the feed task, rebuild the adapter, restart and `clear()`. Same shape as the
existing credentials-change handling for `exchange_events_task`.

## Error handling

Every failure degrades to REST. None can produce a stale quote.

| condition | behaviour |
| --- | --- |
| Cold start, still connecting | cache empty, REST |
| Hard disconnect | `persist_conn` reconnects with backoff; `on_connected` clears; REST until first message |
| 24h forced disconnect | as above, daily |
| Silent stall (socket open, no data) | per-symbol age exceeds `max_age`, REST |
| One symbol's stream dies | that symbol REST, others unaffected |
| Reconnect storm | stays on REST; `reserve()` throttles; cost converges to today's |
| Malformed message | logged, dropped, good state untouched |
| Crossed or zero book | rejected in `apply()`, never stored |

Quote age is stamped at parse time on a monotonic clock. If the event loop
stalls, a message can sit unprocessed and then be stamped "now", making an older
price look fresh — not hypothetical, since the `listenKey` keepalive does a
blocking sync `requests.put` every 30s. A few hundred ms does not threaten a 1s
threshold, so the guard stays monotonic-only and the exchange `E` field is
recorded alongside it, emitting a metric when `synced_now - E` diverges
materially from the monotonic age. That surfaces both loop stalls and clock skew
without complicating the guard, reusing the `exchange_time_offset` the rate
limiter already maintains.

## Testing

`PriceFeed` — pure, clock injected, no mocks: fresh served; stale `None`; unseen
`None`; `clear()` empties; `invalidate(symbol)` affects only that symbol; crossed
book rejected; zero book rejected.

Equivalence — for the same book, `get_current_price` from the feed equals
`get_current_price` from REST. This is the test that guarantees changing the
source does not silently move our quotes.

Adapter (cybotrade) — payload fixtures, no network, following
`py-cybotrade/tests/test_exceptions.py`: a real `depth5` payload parses to the
typed event; a malformed one is dropped, not raised.

Wiring (adrs) — a feed hit must not touch the exchange at all (assert the client
mock was never awaited); a stale read must fall back *through* `reserve()`, since
that is the property bounding the worst case.

Reconnect — `on_connected` fires, cache cleared, next read is REST.

## Open questions

**Does Partial Book Depth push while the book is idle?** The whole
illiquid-symbol argument rests on it, and the Binance docs pages for it kept
redirecting during design, so it is unverified rather than assumed.

Settle it first, before any OMS wiring, with a
`scripts/verify_binance_price_feed.py` following the existing
`scripts/verify_bybit_rate_limiter.py` convention: subscribe to a thin pair and
log inter-message gaps while the book is motionless.

If it does not tick when idle, keep `@bookTicker` as the price source and add
`<symbol>@markPrice@1s` purely as a per-symbol liveness beacon. Everything
downstream is unchanged; only the adapter's subscription list differs.

## Bybit readiness

Recorded so the follow-up does not relitigate it. `orderbook.1.{symbol}` is
10ms, typed `snapshot`/`delta`, with `u` and `seq`; `u == 1` means service
restart and requires a local reset; a no-change book re-pushes a snapshot after 3
seconds reusing the previous `u`. (`tickers.{symbol}` also carries
`bid1Price`/`ask1Price` at 100ms, but for linear it sends deltas where an absent
field means unchanged, so it needs partial-state merging. `orderbook.1` is the
cleaner top-of-book source.)

`PriceFeed` needs no change: snapshot/delta merging and gap detection live in
`BybitPublicWS`, which normalises to the same event. It uses `invalidate(symbol)`
on `u == 1` or a sequence gap, and its own `max_age` of ~4s.

Both exchanges then share one mechanism — a guaranteed periodic push that makes
silence meaningful.

## Out-of-scope follow-ups

- Route `start_user_data_stream` / `keepalive_user_data_stream` through the async
  cybotrade http client. Today they are blocking sync `requests` calls that stall
  the event loop and are invisible to the rate limiter.
- Bybit public feed, per *Bybit readiness*.
- Position/account websocket stream, removing REST `GET_POSITION` from the hot
  path.

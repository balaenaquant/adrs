# Websocket Price Feed — Design

Date: 2026-08-11
Status: approved, not yet implemented. The gating experiment has been run and
refuted the first approach; the design below reflects the result. See *Experiment*
and *Environment caveat*.

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

**Liveness is a property of the connection; freshness is a property of the
quote.** This is the crux, and it took an experiment to get right.

`@bookTicker` is event-driven: it pushes on every top-of-book change. So for a
quiet book, silence is *positive information* — nothing changed, and an old quote
is still the current top of book. Treating age alone as the guard answers the
wrong question and is actively harmful: an illiquid pair would fall back to REST
every tick to re-fetch a price that had not moved, spending weight 10 to learn
nothing and losing the entire saving on exactly the pairs where nothing happens.

The right question is **"would I have been told if it moved?"** — which is
answered by whether the socket is healthy, not by how old the quote is.

Two candidate ways to prove the socket is healthy were rejected on evidence:

- **Ping/pong cannot do it.** Binance sends a ping every 3 minutes with a
  10-minute pong timeout. Three-minute resolution is useless as a freshness guard.
- **A timer-driven market stream would do it, but Binance has none that suits.**
  Partial Book Depth (`<symbol>@depth5@100ms`) looked ideal — per-symbol, carries
  top-of-book, documented at 100ms. It is change-driven. See *Experiment*.

**So liveness comes from a high-frequency stream on a liquid symbol.** The feed
always subscribes to `btcusdt@bookTicker` in addition to the traded symbols
(deduplicated when BTCUSDT is itself traded). BTCUSDT's book never stops moving,
so that subscription delivers continuously whenever the socket is healthy —
measured at 58-212 messages/second with a **515ms maximum gap over 60 seconds**.

`PriceFeed` therefore tracks two things:

- `last_message_at` — any message on the connection, from any stream. This is the
  liveness signal, and `heartbeat_max_age` (2s, ~4x the observed worst gap) is
  what gates every quote.
- per-symbol `received_at` — used only as a **backstop** with a deliberately
  generous cap (60s), covering the one case liveness cannot: a single symbol's
  subscription dying while the connection stays healthy. Without the backstop
  that symbol would be quoted from a dead subscription indefinitely.

The heartbeat gates the quote, not the reverse: a quiet symbol on a healthy
socket is served at any age below the backstop, which is what makes an illiquid
pair cost nothing.

Websocket cadence costs no request weight either way: market streams are metered
by connection count, not request weight.

**Bybit will not need the heartbeat.** `orderbook.1` re-pushes a snapshot after 3
seconds of no change, so it carries its own per-symbol liveness and can use
per-symbol age directly with `max_age` ~4s. The asymmetry is real and worth
keeping rather than forcing one mechanism onto both.

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

## Experiment

Run with `scripts/verify_binance_price_feed.py` on 2026-08-11. BITOUSDT was
chosen by querying `/fapi/v1/ticker/24hr` for the lowest 24h quote volume of 678
USDT perps (~1.2k trades/24h), so "idle" is a measured condition rather than a
guess, with BTCUSDT as a control on the same connection.

| stream | msgs / 90s | median gap | max gap |
| --- | --- | --- | --- |
| `bitousdt@depth5@100ms` | 6 | 8,046ms | **50,592ms** |
| `bitousdt@bookTicker` | 2 | 38,167ms | 58,660ms |
| `btcusdt@depth5@100ms` | 807 | 102ms | 517ms |

**Partial Book Depth is change-driven.** A 100ms timer over 90s would deliver
~900 messages; the idle book delivered 6, with a 50-second silence. The control
delivered 807 at a 102ms median on the same connection, which both validates the
harness and shows the trap: on a liquid symbol the stream looks perfectly
timer-like, because its book never stops moving. Testing only BTCUSDT would have
"confirmed" the original design.

The measurement had to be fixed first. Recording only the gaps *between* messages
made an early burst followed by a minute of silence report a 173ms maximum gap and
look healthy. The trailing gap from the last message to the end of the run must be
counted.

Separately, `btcusdt@bookTicker` was measured at 12,726 messages/60s (max gap
515ms) and 1,038/18s, which is what the heartbeat design relies on.

## Environment caveat

In the environment where the experiment ran, **only order-book streams are
delivered**. `@aggTrade`, `@markPrice@1s` and `@ticker` are accepted by
`SUBSCRIBE` (`{"result": null}`) and confirmed by `LIST_SUBSCRIPTIONS`, yet
deliver nothing — verified both combined and each alone on its own connection,
with an independent `websockets` client as well as cybotrade's. Zero `@aggTrade`
on BTCUSDT over 18s is impossible on a live market (~26 trades/sec), so this is an
egress filter, proxy or regional restriction locally, not Binance behaviour.

Consequences:

- Whether `@markPrice@1s` is usable as a per-symbol beacon is **unresolved**, and
  nothing here should be read as evidence that it is unavailable in production.
- The chosen design does not depend on it, which is a reason to prefer the
  heartbeat over a beacon rather than merely a convenience.
- The Partial Book Depth conclusion is unaffected: both arms of that comparison
  used `@depth5`, a stream that does deliver here.
- **Re-run the script from the shard before implementing.** If the shard also
  cannot receive `@markPrice`/`@aggTrade`, that is a genuine production
  constraint worth knowing early. If it can, per-symbol beacons become an option
  again — but the heartbeat design still stands and needs no revisiting.

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
stream, heartbeats with `Message.Ping()`, and `start()` iterates `persist_conn`.
Unlike the private WS it needs no listenKey, so it makes no REST call at all —
notably avoiding that class's blocking sync `requests.put` on every heartbeat.

Subscribes to `<symbol>@bookTicker` for every traded symbol, **plus
`btcusdt@bookTicker` as the liveness heartbeat**, deduplicated when BTCUSDT is
itself traded. The heartbeat subscription is not optional: without it, a portfolio
of only illiquid symbols has nothing proving the socket is alive.

Its only job is wire format to typed event. It holds no cache and makes no
policy decisions. Malformed messages are logged and dropped, never raised, and
never invalidate good state.

Needs two additions to cybotrade's shared vocabulary:

- `EventType.BookTicker`
- a `BookTicker` model: `symbol, bid, ask, bid_qty, ask_qty, update_id, event_time`

URL form is confirmed working: `wss://fstream.binance.com/stream?streams=a/b/c`
(the un-prefixed form, matching `BinancePrivateWS`). The `/public/` variant that
also appears in the docs was not needed. Combined-stream payloads arrive wrapped
as `{"stream": name, "data": {...}}`; the `/ws` endpoint with an explicit
`SUBSCRIBE` frame delivers them unwrapped, and is the better fit if the symbol set
must change without reconnecting — see *Data flow*.

**`PriceFeed` — adrs**

The policy unit. Holds `dict[Symbol, Quote]` where `Quote = (bid, ask,
received_at)`, plus a single `last_message_at` for connection liveness.

- `apply(book_ticker)` — stamp `last_message_at`, then store the quote after
  rejecting a crossed or zero book (`bid >= ask`, or either side 0). A corrupt
  message must never become a quote — but it still counts as liveness, since it
  proves the socket is delivering.
- `get(symbol) -> Quote | None` — `None` when any of: the connection is not proven
  live (`now - last_message_at > heartbeat_max_age`, 2s), the symbol is unseen, or
  the quote exceeds the backstop cap (60s). Otherwise the quote, at any age.
- `clear()` — drop all quotes and reset liveness. Fires on reconnect.
- `invalidate(symbol)` — drop one symbol. Never fires on Binance; exists so
  Bybit's per-symbol resync is an addition rather than a redesign.
- counters: quotes served, fallbacks taken by reason (no-liveness / unseen /
  backstop), feed-down duration, event-time divergence.

All ages are measured on `time.monotonic()`.

Note that `last_message_at` is stamped from *any* stream, including the BTCUSDT
heartbeat, so it stays fresh regardless of which symbols are quiet. That is the
whole mechanism: liveness comes from the busiest book on the connection, and
individual quotes are then trusted for as long as the connection is proven alive.

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

Symbol-set changes: with the combined-URL form the stream list is encoded in the
URL, so `on_refresh_config` detecting a `base_asset_to_symbol_table` change must
tear down the feed task, rebuild the adapter, restart and `clear()` — the same
shape as the existing credentials-change handling for `exchange_events_task`.

Using the `/ws` endpoint with `SUBSCRIBE`/`UNSUBSCRIBE` frames instead would allow
changing symbols without dropping the connection, and was confirmed working
during the experiment (`{"result": null}` ack, `LIST_SUBSCRIPTIONS` reflects
state). Prefer it if symbol churn turns out to be frequent; the reconnect approach
is simpler and adequate if it is rare. Either way `clear()` on reconnect is
unchanged.

## Error handling

Every failure degrades to REST. None can produce a stale quote.

| condition | behaviour |
| --- | --- |
| Cold start, still connecting | no liveness yet, REST |
| Hard disconnect | `persist_conn` reconnects with backoff; `on_connected` clears; REST until first message |
| 24h forced disconnect | as above, daily |
| Silent stall (socket open, no data) | `last_message_at` ages past `heartbeat_max_age` (2s) → **every** symbol REST |
| Heartbeat symbol stops but others flow | still live: any message refreshes liveness |
| One symbol's subscription dies, connection healthy | liveness holds, so that symbol is served until the 60s backstop, then REST |
| Reconnect storm | stays on REST; `reserve()` throttles; cost converges to today's |
| Malformed message | logged, dropped; counts as liveness, does not become a quote |
| Crossed or zero book | rejected in `apply()`, never stored |

The row worth dwelling on is the second-to-last: it is the one case the heartbeat
cannot detect, and the 60s backstop is what bounds it. A tighter backstop would
reintroduce the illiquid-pair cost this design exists to remove, so the trade is
deliberate — up to 60s of quoting from a dead single subscription, versus REST on
every tick for every quiet symbol.

Quote age is stamped at parse time on a monotonic clock. If the event loop
stalls, a message can sit unprocessed and then be stamped "now", making an older
price look fresh — not hypothetical, since the `listenKey` keepalive does a
blocking sync `requests.put` every 30s. A few hundred ms does not threaten the 2s
heartbeat threshold, so the guard stays monotonic-only and the exchange `E` field is
recorded alongside it, emitting a metric when `synced_now - E` diverges
materially from the monotonic age. That surfaces both loop stalls and clock skew
without complicating the guard, reusing the `exchange_time_offset` the rate
limiter already maintains.

## Testing

`PriceFeed` — pure, clock injected, no mocks. The liveness interaction is the part
worth covering thoroughly, because it is where the design's value and its risk both
live:

- a quiet symbol is served at 30s old **while liveness is fresh** — this is the
  illiquid-pair case the design exists for, and it must not regress to REST
- the same quote returns `None` once `last_message_at` ages past 2s
- a message for symbol A refreshes liveness for symbol B (the heartbeat mechanism)
- a malformed message refreshes liveness but does not become a quote
- the 60s backstop returns `None` even while liveness is fresh
- unseen symbol `None`; `clear()` empties and resets liveness;
  `invalidate(symbol)` affects only that symbol; crossed and zero books rejected

Equivalence — for the same book, `get_current_price` from the feed equals
`get_current_price` from REST. This is the test that guarantees changing the
source does not silently move our quotes.

Adapter (cybotrade) — payload fixtures, no network, following
`py-cybotrade/tests/test_exceptions.py`: a real wrapped combined-stream
`bookTicker` payload parses to the typed event; a malformed one is dropped, not
raised; the subscription list includes the heartbeat symbol and deduplicates it
when BTCUSDT is traded.

Wiring (adrs) — a feed hit must not touch the exchange at all (assert the client
mock was never awaited); a stale read must fall back *through* `reserve()`, since
that is the property bounding the worst case.

Reconnect — `on_connected` fires, cache cleared, next read is REST.

## Open questions

None blocking. The question that gated this design — whether Partial Book Depth
ticks on an idle book — was answered no; see *Experiment*.

Two things to do at implementation time rather than decide now:

- **Re-run `scripts/verify_binance_price_feed.py` from the shard**, per
  *Environment caveat*. It does not change the design, but if the shard shares the
  local stream filtering that is worth knowing before deploying.
- **Choose the combined-URL or `/ws` + `SUBSCRIBE` form** based on how often the
  symbol set actually changes; see *Data flow*.

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
on `u == 1` or a sequence gap.

The two exchanges deliberately use *different* liveness mechanisms, and forcing
them together would be a mistake. Bybit's 3-second repeat-snapshot is per-symbol
liveness, so Bybit can gate on per-symbol age alone with `max_age` ~4s and needs
no heartbeat subscription. Binance has no equivalent, which is why it needs the
liquid-symbol heartbeat. `PriceFeed` supports both by exposing the two knobs
(`heartbeat_max_age`, backstop cap) rather than one policy: Binance sets both,
Bybit disables the heartbeat requirement and sets a tight per-symbol age.

## Out-of-scope follow-ups

- Route `start_user_data_stream` / `keepalive_user_data_stream` through the async
  cybotrade http client. Today they are blocking sync `requests` calls that stall
  the event loop and are invisible to the rate limiter.
- Bybit public feed, per *Bybit readiness*.
- Position/account websocket stream, removing REST `GET_POSITION` from the hot
  path.

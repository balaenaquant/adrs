# Bybit Price Feed and Position Stream — Design

Date: 2026-08-14
Status: approved, not yet implemented
Follows: `2026-08-11-ws-price-feed-design.md` (shipped as adrs 1.8.0/1.8.1) and
`2026-08-13-ws-position-stream-design.md` (shipped as adrs 1.9.0, corrected in 1.9.1)

## Purpose

Bring Bybit to parity with Binance: serve top-of-book from a websocket instead of
polling, and maintain `PositionManager.exchange` from Bybit's `position` topic
instead of a REST read on every placement tick.

This is **parity work, not the fix for Bybit's rate-limit complaint.** That
distinction is load-bearing, so it is recorded here rather than left implied. The
Bybit rate-limit report is a burst of order *cancellations* exhausting the cancel
pool:

```
0 remaining of 8 (X-Bapi-Limit-Status)
Failed to cancel order <id> because Failed due to rate limits
  <RateLimitState IP_GLOBAL: Usage: 0, UID_PLACE: Remaining: 9/10,
   UID_CANCEL: Remaining: 0/None, UID_POSITION: Remaining: 49/50,
   UID_WALLET: Remaining: 49/50, UID_OPEN_ORDERS: Remaining: 49/50>
[CHECK_LIMITS] UID_CANCEL reached its limit
```

Dozens of distinct order ids, all inside one millisecond, with every read pool
idle. A price feed and a position stream reduce **reads**; this is a write pool.
Neither change in this spec will improve it. The cancel-burst problem is tracked
separately and is likely the higher-value work.

What this spec does buy: Bybit stops spending `UID_POSITION` and price-read quota
on every tick, and the drift-prone incremental fill accounting is retired on the
Bybit side too — the same correctness win that motivated the Binance position
stream, independent of any rate-limit relief.

## What is already reusable

Most of the machinery exists, because the Binance work was built without
exchange-specific assumptions in the consuming layers:

| piece | state |
| --- | --- |
| `EventType.BookTicker` | exists |
| `EventType.PositionUpdate` | exists |
| `PriceFeed` (`adrs/oms/price_feed.py`) | exchange-agnostic, unchanged |
| `PositionManager.apply_stream_positions` | takes `list[Position]`, unchanged |
| `POSITION_ANCHOR_MAX_AGE_SEC`, the 60s REST reconcile, `invalidate_exchange_anchor` | unchanged |
| `case EventType.PositionUpdate:` in `on_exchange_event` | already source-agnostic |
| `BybitPrivateWS` | exists, topic-dispatched, handles `order` |

So **no new event type and no change to position *handling* in adrs** — the
consumer is already source-agnostic.

**One adrs change is needed for the position half, though, and it is easy to miss.**
Unlike Binance, where a listenKey stream delivers every event type unbidden, Bybit
requires explicit topic subscription — and adrs owns that list.
`config.py:to_exchange_event()` currently constructs `BybitPrivateWS(...,
topics=["order"])`, so `position` frames would never arrive no matter what the
cybotrade parser does. That list must become `["order", "position"]`.

An earlier draft of this spec claimed the position half needed no adrs change at
all. That was wrong, and it is recorded here because the mistake is instructive:
the two exchanges differ in *who* decides what the private socket carries.

## Gate experiments (both run before this spec was written)

The Binance work established that assumptions about payload shape must be
measured, not read from documentation — the order-book depth assumption there was
wrong and only an experiment caught it. Both Bybit unknowns were settled the same
way, against a Bybit **demo** account and from a production shard.

### Is Bybit's position payload absolute?

Bybit sends `size` (unsigned) plus a separate `side`, not one signed number like
Binance's `pa`, so a consumer must derive the sign. Opened and closed 0.001
BTCUSDT while subscribed to `position`:

| | after open | after close |
| --- | --- | --- |
| `size` | `'0.001'` | `'0'` |
| `side` | `'Buy'` | `''` |
| `entryPrice` | `'63354.1'` | `'0'` |

**`size` is absolute and unsigned.** The overwrite-never-accumulate design holds.

**On close, `side` is an empty string** — not `"None"`, not absent, not `"Sell"`.
This is the finding that would most likely have been guessed wrong: a parser
requiring `side in {"Buy", "Sell"}` would raise or mis-sign on every close, and a
closed position that fails to parse is exactly the frame a consumer must not
miss.

**Frames are partial** — one symbol per frame, the one that changed. So
`apply_stream_positions` overwrites per symbol; replacing the whole mapping would
erase positions a frame does not mention.

`positionIdx: 0` on both frames confirms one-way mode. Per-position
`unrealisedPnl` is present, as `up` is on Binance.

**Difference from Binance worth recording:** Bybit's REST `/v5/position/list`
*includes* flat symbols as `size='0', side=''`, whereas Binance's
`/fapi/v3/positionRisk` *omits* them. The Binance verification script had a bug
caused precisely by that omission. Any Bybit verifier must not inherit that
workaround — for Bybit, absent means genuinely unknown, not flat.

### Which public topic delivers, and in what shape?

On Binance, only order-book streams reached the production egress IP —
`markPrice`, `aggTrade` and `ticker` delivered nothing there. So topic choice was
measured from inside a Bybit tenant pod on the production shard, not assumed:

| topic | rate | max gap | carries bid/ask |
| --- | --- | --- | --- |
| `orderbook.1.BTCUSDT` | 21.2/s | 590ms | yes — `b: [["63357.4","1.827"]]`, `a: [["63357.5","10.36"]]` |
| `tickers.BTCUSDT` | 8.4/s | 601ms | fat payload; deltas carry only changed fields |

`orderbook.1` wins: it always carries both sides, and `tickers` deltas may omit
bid/ask entirely, which would leave the feed guessing.

A second probe characterised the frame type over 30s across two symbols — 3,683
frames, **100% `type: "snapshot"`, 100% with both sides present**:

```json
{"topic":"orderbook.1.BTCUSDT","ts":1786639170499,"type":"snapshot",
 "data":{"s":"BTCUSDT","b":[["63040.8","3.323"]],"a":[["63040.9","4.686"]],
         "u":31071061,"seq":763537012912},"cts":1786639170498}
```

So at depth 1 every frame is a complete top-of-book and **no delta merging or
per-symbol state is required**. That was not assumed: the original design
question was how to merge one-sided deltas, and the measurement removed the need
for the machinery entirely.

Because 30s is not proof about all future behaviour, the adapter still handles the
other case defensively — see *Error handling*.

Note the rate: 21-61/s per symbol against Binance `bookTicker`'s ~265/s, so the
event-loop cost that made the Binance heartbeat expensive is roughly an order of
magnitude smaller here.

## Architecture

```
Bybit public WS   orderbook.1.<SYM> ──► BookTicker ──► PriceFeed  (existing, unchanged)
                  (snapshot-only, b[0]/a[0])

Bybit private WS  topic "position"  ──► list[Position] ──► PositionManager
                  (size + side → signed Decimal)           .apply_stream_positions
                                                           (existing, unchanged)

REST GET_POSITION      every 60s ────► the correction anchor and the liveness signal
REST wallet balance    every call ───► create_equity (always fresh, by rule)
```

### cybotrade

**`BybitPrivateWS` gains a `case "position"`** beside the existing `case "order"`,
in the same `match msg["topic"]` block. It maps each `data[]` entry onto the
existing `Position` model:

- `size` → `quantity`, as a **signed `Decimal`**: negative when `side == "Sell"`,
  positive when `side == "Buy"`.
- `side == ""` is the close frame. With `size == "0"` the sign is irrelevant and
  the result is `Decimal("0")`. It must not raise and must not be filtered out —
  dropping it would leave a consumer holding the old size.
- `entryPrice` → `entry_price`.
- `symbol` → `Symbol`, the same raw exchange string the REST path uses, so stream
  and REST land on the same dict keys.
- `positionIdx` is not read. Hedge mode is out of scope.

Emits `Event(event_type=EventType.PositionUpdate, data=list[Position])` — the same
event the Binance adapter emits, so adrs needs no new handling.

Parsing is defensive per entry, as the Binance parser is: an unparseable entry is
logged and skipped, a frame with no usable entries emits nothing. This socket also
carries order fills, so one bad frame must not kill it.

**`BybitPublicWS` is new**, mirroring `BinancePublicWS`. It subscribes
`orderbook.1.<SYMBOL>` for each configured symbol, emits `EventType.BookTicker`
built from `b[0][0]` (bid) and `a[0][0]` (ask), and carries the same
heartbeat-based liveness: a frame arriving recently is what proves the connection
is alive, since the quote itself is change-driven.

### adrs

Two changes, both small.

**1. Subscribe to the `position` topic.** `config.py:to_exchange_event()` builds
`BybitPrivateWS(..., topics=["order"])`. Add `"position"`. Without this the
cybotrade parser is dead code, because Bybit only sends topics you ask for.

**2. Select the public feed by exchange.**
`adrs/oms/oms.py` imports `BinancePublicWS` concretely and types
`price_feed_ws: BinancePublicWS | None`. That becomes exchange-selected, following
the pattern `adrs/oms/config.py` already uses to choose `BybitLinearClient` vs
`BinanceLinearClient` and `BybitPrivateWS` vs `BinancePrivateWS`. The selection
belongs in `config.py` with its siblings, not as a branch inside `oms.py`.

**`config.py` supports four exchanges, not two** — Bybit, Binance, **Kucoin and
EdgeX** (`config.py:80,88,97,104`). Only Binance and Bybit get a public feed
adapter, so the other two must keep running with REST prices.

The good news, confirmed by reading it: **that fallback already exists and is
already exercised in production.** `_start_price_feed` (`oms.py:459-463`) opens
with

```python
if self.config.config.credentials.exchange is not Exchange.BINANCE_LINEAR:
    logger.info(
        "[PRICE_FEED] No public feed for "
        f"{self.config.config.credentials.exchange}, using REST prices"
    )
    return
```

so today Bybit, Kucoin and EdgeX all take that branch and run exactly as they did
before 1.8.0. The change is therefore to *widen* a guard that already works, not to
invent `None` handling: admit Bybit, pick the adapter class, and leave Kucoin and
EdgeX taking the same early return with the same log line.

This is still the main regression risk in the change — a rewrite that assumes a
class, or drops the early return, would break two exchanges that have nothing to do
with Bybit — but the safe path is already there to preserve rather than build.

## Data flow

The Bybit private websocket already runs — `BybitPrivateWS`, started as
`exchange_events_task`, supervised since 1.8.0. `position` frames arrive on it
today and are discarded by the `case _: pass` at the end of the topic match. That
discard is the Bybit equivalent of the `[ON_EVENT] 'EventType.Unknown'` bursts
that disappearing confirmed the Binance stream was live, and it is the cheapest
positive signal that this change took effect.

The public feed is a new connection with its own lifecycle, reusing the
supervision and start/stop paths added for Binance in 1.8.0.

## Error handling

| condition | behaviour |
| --- | --- |
| Steady state, anchor fresh | stream serves sizing and pricing, no REST reads |
| Position anchor older than 90s | `delta_calculation` forces a read, as today |
| Private socket dies, account quiet | indistinguishable from quiet; the 60s anchor keeps correcting, so exposure is bounded at 60s |
| Missed `position` frame | the next frame is absolute, or the anchor corrects; no accumulating error |
| Private socket reconnects | anchor invalidated via `invalidate_exchange_anchor()`, forcing one REST read — the reconnect gap loses order updates too, so nothing else would notice |
| Price feed frame is `type: "delta"` | **dropped, not merged.** Measured behaviour is snapshot-only; a delta means an assumption changed, and dropping fails safe because the next snapshot lands within ~20ms |
| Price feed frame missing `b` or `a` | dropped, same reasoning |
| Price feed quote crossed or zero | already rejected by `PriceFeed.apply`, which pops the symbol |
| Price feed reconnects | `PriceFeed.clear()`, because a quote goes stale even though a position does not |
| Price feed silent past the heartbeat threshold | quotes treated as not live; read paths fall back to REST |

The asymmetry between the two reconnect rows is deliberate and is the same rule
the Binance design set: **liveness belongs to the connection, freshness belongs to
the quote.** A position is a fact about the account that survives a dropped
socket; a quote is not.

## Balances

`create_equity` keeps reading the wallet balance from REST on **every** call. No
streamed balance substitutes for it under any freshness condition, and no
anchor logic is extended to cover it. This is the same correctness boundary the
Binance design set, restated because it is the rule most likely to be "optimised"
by a later reader. No `EventType.BalanceUpdate`, and no balance value crosses from
the Bybit stream into position or equity state.

## Rate-limit accounting

Bybit meters per-UID request counts, not a per-IP weight budget, so the saving is
counted in requests against pools rather than weight:

- `UID_POSITION` — currently spent on every placement tick by
  `delta_calculation`; the stream removes that, leaving the 60s anchor read.
- Price reads — removed from the tick.
- `UID_CANCEL`, `UID_PLACE` — **unaffected.** These are the pools that actually
  break, per the complaint above.

Note the ceiling arithmetic, because it misled once during this design: the
"8" in `0 remaining of 8` is *ours*, not Bybit's.
`BybitLimitProfile.with_buffer(buffer_pct = 1.0 - soft_limit_percent)` reduces the
documented bootstrap of 10 by the 20% safety buffer. We deliberately under-admit.
Likewise `UID_CANCEL: Remaining: 0/None` is not a dropped limit — `None` means no
response header for that pool has been observed yet, and `_uid_pool_snapshot`
documents the fallback to the buffered profile limit.

## Testing

**cybotrade, position parsing** — payload-fixture tests from the captured frames,
no network:

- the open frame (`size='0.001'`, `side='Buy'`) → `quantity == Decimal("0.001")`.
- the close frame (`size='0'`, `side=''`) → `quantity == Decimal("0")`, present in
  the emitted list rather than filtered out.
- a `Sell` frame → negative `quantity`.
- a frame with several `data[]` entries → one `Position` each, order preserved.
- malformed entries → skipped, the rest of the frame still emitted; a frame with
  no usable entries emits no event.
- no balance plumbing exists (assert no `EventType.BalanceUpdate`).

**cybotrade, public feed** — fixture tests over the captured snapshot frame:

- a snapshot → `BookTicker` with bid `b[0][0]` and ask `a[0][0]`.
- a `type: "delta"` frame → dropped, no event.
- a frame with an empty `b` or `a` → dropped, no event.
- subscription message shape for several symbols.

**adrs** — the price-feed selection:

- a Bybit-configured OMS constructs `BybitPublicWS`; a Binance-configured one
  still constructs `BinancePublicWS`; neither import is hard-coded in `oms.py`.
- **a Kucoin- and an EdgeX-configured OMS get `None` and still start**, with no
  feed task and no error — the regression this spec is most likely to cause.
- with the selector returning `None`, a price read falls back to REST, i.e.
  pre-1.8.0 behaviour.

The existing position-stream tests already cover the consuming side and need no
extension, which is the point of the event being source-agnostic.

Quantities are `Decimal` throughout, never float. Ages use `time.monotonic()`.

## Rollout

Same shape as the Binance rollout, and the same constraint applies: the position
stream and the code that retires the per-tick read must ship together. Do not ship
one without the other, and do not revert one without the other.

Validation on a Bybit tenant, in order:

1. The `case _: pass` discard of `position` frames stops happening — the cheapest
   proof the stream is feeding.
2. `UID_POSITION` remaining stays near its ceiling between anchor reads.
3. Force a socket drop and confirm the anchor invalidation fires one REST read.
4. Prices served from the feed, with REST fallback exercised by pausing the feed.

## Out of scope

- **The cancel-burst problem.** The actual rate-limit complaint. Separate work,
  probably more valuable than this spec.
- Hedge mode. The REST path ignores `positionIdx` and lets the last position per
  symbol win; the stream mirrors that exactly. Documented limitation.
- Bybit balance or equity from the stream. See *Balances*.
- Retiring the 60s REST anchor. It is the liveness mechanism, not redundancy.
- Learning the account's true `UID_CANCEL` tier when Bybit omits the ceiling
  header. Noted during the gate runs — order placement responses do return
  `x-bapi-limit: 10`, but the cancel path was deliberately not probed, since
  hammering the pool that is already failing for users would be reckless.

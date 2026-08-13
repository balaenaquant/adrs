# Websocket Position and Balance Stream — Design

Date: 2026-08-13
Status: approved, not yet implemented
Follows: `2026-08-11-ws-price-feed-design.md` (shipped as adrs 1.8.0/1.8.1)

## Purpose

Maintain OMS positions from Binance's `ACCOUNT_UPDATE` user-data stream instead of
polling `GET_POSITION`, and retire the drift-prone incremental fill accounting
while doing it.

Position and balance reads are what remain of the OMS's steady-state request
weight now that the price feed serves top-of-book. Measured against
`examples/example_config.json` (`order_placement_interval: 30`, aegis 60s) with
two symbols:

| read | trigger | weight/min |
| --- | --- | --- |
| `GET_POSITION` forced by `delta_calculation` | placement tick, 2x/min | 10 |
| `GET_POSITION` on the aegis tick | 1x/min | 5 |
| `GET_WALLET_BALANCE` on the aegis tick | 1x/min | 5 |
| open orders | placement tick | 4 |

Steady state is ~24 weight/min, of which **20 is positions and balance**. This
design removes the 10 spent by the placement tick, taking steady state to ~14
(a 42% cut). See *Balances* for why the remaining 10 stays.

The correctness motive matters as much as the weight. `position.exchange` has
three writers today:

| writer | semantics |
| --- | --- |
| `position.py:121` (`update_exchange`, REST) | **absolute**, authoritative |
| `order_placement_manager.py:168` (`update_positions`, from `OrderUpdate`) | **incremental**, `+= asset_filled` |
| `oms.py:194` (`init`) | seeds zeros for unknown symbols |

The incremental writer drifts by construction — its own comment records a bug
where fills over-count from the third update onward — and REST polling is what
silently rescues it. Replacing it with absolute values from `ACCOUNT_UPDATE`
retires a bug class rather than adding a fast path beside it.

## Scope

In scope, Binance USDⓈ-M only:

- `EventType.PositionUpdate` and `ACCOUNT_UPDATE` parsing in `cybotrade`.
- `PositionManager.apply_stream_positions`, and `delta_calculation` trusting it
  while the REST anchor is fresh.
- Deleting the `exchange += asset_filled` line from `update_positions`.
- Parsing and emitting the frame's balance array (`a.B`), which arrives in the
  same frame for free. See *Balances* for what it does and does not replace.

Out of scope:

- Bybit. Follows as its own change; see *Bybit readiness*.
- Hedge mode. Today's REST path ignores `positionSide` and lets the last position
  per symbol win; the stream mirrors that exactly. Documented limitation, not
  fixed here.
- Retiring the REST equity poll. See *Balances*.

## Decisions

**`ACCOUNT_UPDATE` replaces the increments; REST stays as a slow anchor.** Two
writers, not three: absolute values from the stream between polls, and a REST
read every 60s that re-establishes truth. The incremental writer goes away.

**The order-sizing path trusts the stream while the anchor is fresh.** This is
where the saving is: `delta_calculation` currently forces `max_age_sec=0` on every
placement tick, and that read is 10 of the 20 weight. It now passes
`POSITION_ANCHOR_MAX_AGE_SEC` (90s) instead, so a fresh anchor means no read.

**The 60s REST reconcile is the liveness mechanism, not merely a correctness
floor.** This is the load-bearing decision and it differs from the price feed.

`ACCOUNT_UPDATE` is event-driven: it fires when the account changes, so a quiet
account emits nothing for hours and silence is ambiguous exactly as it was for
`@bookTicker`. But the price feed's answer — subscribe to a busy stream as a
heartbeat — has no equivalent here. The user-data socket carries only account
events; Binance's socket-level ping is every 3 minutes and the listenKey keepalive
every 30, both useless as a freshness signal.

So "the position feed is live" is defined as *the last successful REST reconcile
was within 90 seconds*. If REST starts failing, the anchor ages out and the sizing
path forces a read, exactly as today. This needs no new trust model and does not
require solving "is the user-data socket alive?", for which no cheap signal
exists.

**It is a separate socket from the price feed**, so `PriceFeed`'s liveness must
not be reused for positions — the two fail independently.

**No new class and no new cron job.** `PositionManager` already owns
`self.exchange` and the REST path; a second owner would create two sources of
truth. `_positions_fresh_within()` and `_exchange_refreshed_at` (added when
position reads were coalesced) are already the anchor-age check this needs, and
`on_aegis_update` already calls `update_exchange()` every 60s — the anchor exists
today.

**`update_positions` loses one line, not two.** It does
`pending -= asset_filled` and `exchange += asset_filled`. Only the second is
removed. `pending` keeps its incremental accounting, corrected each placement tick
by the open-orders snapshot; that machinery is untouched.

## Architecture

```
ACCOUNT_UPDATE ─┬─ a.P ──► Position[] ──► PositionManager.exchange   (absolute, free)
  (user-data    └─ a.B ──► Balance   ──► cached for reporting
   socket)
REST GET_POSITION every 60s ──────────► PositionManager.exchange   (anchor + liveness)
```

### cybotrade

`EventType.PositionUpdate`, and `binance/ws.py` parses `ACCOUNT_UPDATE`'s `a.P`
array into the **existing** `Position` model (`symbol`, `quantity`, `entry_price`,
`updated_time`, `orig`): `pa` → signed `quantity`, `ep` → `entry_price`, mirroring
how `get_positions` maps `positionAmt`/`entryPrice` and ignores `positionSide`.
Event data is `list[Position]`, since one frame carries several. No new position
model is needed.

Parsing is defensive in the same way as the `BookTicker` adapter: an unparseable
or partial frame is logged and dropped, never raised. The user-data socket also
carries order fills, so a bad frame must not kill it.

### adrs

1. `PositionManager.apply_stream_positions(list[Position])` — absolute overwrite
   per symbol. No arithmetic, so nothing can drift.
2. `delta_calculation` passes `max_age_sec=POSITION_ANCHOR_MAX_AGE_SEC` instead of
   `0`.
3. `update_positions()` loses its `exchange += asset_filled` line.
4. `OrderPlacementManager.on_exchange_event` gains a `PositionUpdate` case
   calling `apply_stream_positions`.

## Data flow

The user-data stream already runs — `BinancePrivateWS`, started in `run()` as
`exchange_events_task`, supervised since 1.8.0. `ACCOUNT_UPDATE` frames arrive on
it today and are discarded as `EventType.Unknown`; that discard is the source of
the `[ON_EVENT] 'EventType.Unknown': None` bursts in production logs. No new
connection, no new task, no new lifecycle.

On reconnect the listenKey is rebuilt per attempt (the cybotrade 2.1.0 fix), and
the next 60s anchor re-establishes truth. There is deliberately no cache-clear on
reconnect, unlike the price feed: a position is a fact about the account rather
than a quote that goes stale, and the anchor corrects it within 60s regardless.

## Error handling

| condition | behaviour |
| --- | --- |
| Steady state, anchor fresh | stream serves sizing, 0 weight |
| Anchor older than 90s (REST failing) | `delta_calculation` forces a read, as today |
| User-data socket dies, account quiet | indistinguishable from quiet, but the 60s anchor keeps correcting: exposure bounded at 60s |
| Missed `ACCOUNT_UPDATE` frame | next frame (absolute) or next anchor corrects; **no accumulating error**, unlike today |
| Malformed frame | logged, dropped; order fills on the same socket unaffected |
| Reconnect | listenKey rebuilt per attempt; next anchor re-establishes truth |
| Hedge mode | last position per symbol wins — identical to today's REST behaviour |

**Worst case, stated plainly.** A missed `ACCOUNT_UPDATE` leaves a position wrong
for up to 60 seconds, and a placement tick landing in that window sizes an order
off a stale position. Today's incremental path has the same failure *plus*
systematic drift, so this is a reduction in risk — but it is not zero, and that is
the trade being made.

## Balances

`a.B` arrives in the same frame, so parsing it is free. It carries `a` (asset),
`wb` (wallet balance), `cw` (cross wallet balance) and `bc` (balance delta).

**It cannot replace the REST equity poll, and this design does not retire it.**
The equity metric reports `Balance.margin_balance`, which includes unrealised PnL
and therefore moves with mark price. `ACCOUNT_UPDATE` fires on *wallet* changes —
fills, funding, settlement — not on mark-price movement, so a stream-sourced
equity figure would sit still while the real one moved. Retiring the poll would
save 5 weight/min and silently degrade the metric.

What the balance array is worth here:

- Emitted as `EventType.BalanceUpdate` carrying the existing `Balance` model, with
  the fields the frame actually supplies populated and the derived ones
  (`margin_balance`, `equity`, `unrealised_pnl`) left `None` rather than guessed.
- A realised-balance change is visible immediately rather than up to 60s later,
  which is useful for detecting funding and fills.
- It is the input a future accurate-equity derivation would need: streamed wallet
  balance plus streamed positions valued at the price feed's mark would yield
  equity locally, at zero weight. That is a real option but it introduces a figure
  that can disagree with the exchange's own, so it belongs in its own change with
  its own reconciliation check — not here.

Consumers of `Balance` from this stream must treat `margin_balance is None` as
"not available from the stream", never as zero.

## The gating experiment

**`pa` being absolute rather than a delta is unverified and load-bearing.** If it
were a delta, absolute overwrite would be catastrophically wrong — it would
replace a true position with the size of the last change.

Binance's documentation would not serve the `ACCOUNT_UPDATE` payload: three
fetches of the user-data-stream pages returned the docs homepage. The same thing
happened with the order-book depth question in the price feed design, where the
assumption turned out to be wrong and only an experiment caught it.

So, before any wiring: `scripts/verify_binance_account_update.py`, following the
`verify_binance_price_feed.py` convention. It opens a user-data stream, prints raw
`ACCOUNT_UPDATE` frames, and compares the position derived from `a.P` against a
REST `GET_POSITION` taken at the same moment. It must confirm:

- `pa` is the absolute position amount, signed, matching `positionAmt` from REST.
- Whether a frame lists **every** position or only the changed ones. This decides
  whether `apply_stream_positions` overwrites per symbol (correct if partial) or
  may replace the whole dict (only safe if every frame is complete). The design
  assumes **partial** and overwrites per symbol, which is safe either way.
- What `a.B` actually contains for this account type.

It needs Binance credentials, so it runs in the tenant pod or wherever a key
exists; this repo's `.envrc` carries Bybit keys only.

## Testing

`PositionManager.apply_stream_positions` — absolute overwrite; a second frame with
a different quantity replaces rather than accumulates; a symbol absent from a
frame keeps its previous value.

The retired increment — a test that an `OrderUpdate` fill no longer moves
`position.exchange`, and still moves `position.pending`. This is the one that pins
the bug-class removal.

Anchor freshness — `delta_calculation` performs no REST read while the anchor is
fresh, and forces one when it has aged past 90s. Assert on the exchange mock not
being awaited, so a regression that quietly reintroduces the per-tick read fails.

Adapter (cybotrade) — payload fixtures captured from the verification script, not
hand-written, following `test_binance_public_ws.py`. A real frame parses to
`list[Position]`; a malformed one is dropped, not raised; a frame with no `a.P`
yields nothing rather than an empty overwrite.

Balance — `margin_balance` is `None` from a streamed frame, and the equity poll
still runs.

## Bybit readiness

Recorded so the follow-up does not relitigate it. Bybit's equivalent is the
`position` private topic, pushed on change like Binance's. The same anchor
reasoning applies, because it is equally event-driven. `PositionManager` needs no
change: the adapter normalises to `list[Position]` and the anchor logic is
exchange-agnostic.

Bybit's position payload reports `size` as unsigned with a separate `side` field,
unlike Binance's signed `positionAmt`, so the adapter must apply the sign. Getting
that wrong inverts a position, which is why the Bybit change needs its own
verification against a REST read rather than assuming symmetry with Binance.

## Out-of-scope follow-ups

- Accurate equity derived from streamed balance plus streamed positions valued at
  the price feed's mark, retiring the last balance poll.
- Hedge-mode support across both the REST and stream paths.
- The `ACCOUNT_CONFIG_UPDATE` event (leverage changes), which the OMS currently
  learns about only indirectly.

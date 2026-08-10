# Websocket Price Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve top-of-book prices to the OMS from a Binance websocket feed instead of `/fapi/v1/depth`, falling back to REST whenever the feed is not proven live.

**Architecture:** Three units. `BinancePublicWS` (cybotrade) turns `@bookTicker` wire format into a typed event and holds no state. `PriceFeed` (adrs) caches quotes and owns the staleness policy, gating every quote on *connection* liveness rather than per-symbol age — a quiet book on a healthy socket is still the current book. The two functions all four OMS read sites share consult the feed and fall through to today's REST path on a miss.

**Tech Stack:** Python (adrs 3.14, cybotrade 3.12), `cybotrade.websocket` (Rust/pyo3 `persist_conn`, auto-reconnects), `pytest`, `ruff`, `Decimal` for all prices.

Design spec: `docs/superpowers/specs/2026-08-11-ws-price-feed-design.md`. Read the *Decisions*, *Experiment* and *Environment caveat* sections before starting — the first design was refuted by measurement and the reasoning matters.

## Global Constraints

- **Prices are always `Decimal`.** Never float. Mid-price must be `(bid + ask) / Decimal("2.0")` to match `cybotrade.io.exchange.ExchangeClient.get_current_price` exactly.
- **All ages measured on `time.monotonic()`**, never wall clock and never exchange event time.
- **`heartbeat_max_age` = 2.0s**, `quote_max_age` (backstop) = 60.0s, heartbeat symbol = `BTCUSDT`.
- **No config flag.** This is the default path; there is no `price_feed_enabled` setting.
- **A feed miss must fall through to the existing REST path unchanged**, keeping its `rate_limiter.reserve()` wrapper. That wrapper is what bounds the worst case to today's cost.
- **cybotrade release gates Task 5.** adrs pins `cybotrade>=2.0.18`; `EventType.BookTicker` arrives in 2.0.19. Tasks 3 and 4 must not import it — that is why `PriceFeed` takes primitives, not a cybotrade event type.
- **adrs commands** run from the adrs worktree with the parent venv:
  `~/bq/research/adrs/.venv/bin/pytest <args> -q -p no:cacheprovider --no-cov`
  (`--no-cov` because `pyproject.toml` sets `--cov-fail-under=40`, which a single-file run cannot meet.)
- **cybotrade commands** need the compiled extension present in the worktree once:
  `cp ~/bq/execution/cybotrade/py-cybotrade/cybotrade/cybotrade.cpython-312-darwin.so <worktree>/py-cybotrade/cybotrade/`
  then from `~/bq/execution/cybotrade`:
  `PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/<file> -q`
  (The `.so` is gitignored, so it will not be committed.)
- **`ruff check` and `ruff format` must pass** before every adrs commit; pre-commit hooks enforce both.
- Both repos already have branch `fix-rate-limit-ip-scope`; continue on it.

## File Structure

**cybotrade** (`py-cybotrade/cybotrade/`)

| file | responsibility |
| --- | --- |
| `models.py` | add `BookTicker` dataclass — the normalised top-of-book event payload |
| `io/event.py` | add `EventType.BookTicker` to the shared event vocabulary |
| `binance/ws.py` | add `BinancePublicWS` — wire format to typed event, no state, no policy |
| `binance/__init__.py` | export `BinancePublicWS` |
| `tests/test_binance_public_ws.py` | payload-fixture parsing tests, no network |

**adrs** (`adrs/oms/`)

| file | responsibility |
| --- | --- |
| `price_feed.py` | **new.** `Quote`, `PriceFeed` — cache plus staleness policy. Pure: no exchange calls, no cybotrade event types, clock injected. |
| `ops/order_utils.py` | `get_order_book` consults the feed before REST |
| `ops/order_executer.py` | `get_current_price` consults the feed; carries `price_feed` for its two `get_order_book` calls |
| `ops/order_placement_manager.py` | carries `price_feed` for the expiry-check `get_order_book` call |
| `oms.py` | owns the `PriceFeed` and the feed task lifecycle; translates events into the feed |
| `tests/test_price_feed.py` | **new.** policy tests, injected clock, no mocks |
| `tests/test_price_feed_wiring.py` | **new.** feed-hit-avoids-REST, miss-falls-back-through-reserve, equivalence |

`price_feed.py` sits at `adrs/oms/` top level rather than under `ops/`: it is consumed by `ops/`, `oms.py` and later a Bybit adapter, so it is not owned by the order-placement subsystem.

---

## Task 1: cybotrade — `BookTicker` model and event type

**Files:**
- Modify: `py-cybotrade/cybotrade/models.py` (append after `OrderbookSnapshot`, ~line 173)
- Modify: `py-cybotrade/cybotrade/io/event.py:7-14`
- Test: `py-cybotrade/tests/test_book_ticker_model.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `cybotrade.models.BookTicker(symbol: Symbol, bid: Decimal, ask: Decimal, bid_qty: Decimal, ask_qty: Decimal, update_id: int, event_time: datetime)`; `cybotrade.io.event.EventType.BookTicker` with value `"book_ticker"`.

- [ ] **Step 1: Write the failing test**

Create `py-cybotrade/tests/test_book_ticker_model.py`:

```python
"""BookTicker is the normalised top-of-book payload carried by EventType.BookTicker."""

from datetime import datetime, timezone
from decimal import Decimal

from cybotrade import Symbol
from cybotrade.io.event import EventType
from cybotrade.models import BookTicker


def test_book_ticker_holds_decimal_prices():
    bt = BookTicker(
        symbol=Symbol("BTCUSDT"),
        bid=Decimal("63889.10"),
        ask=Decimal("63889.20"),
        bid_qty=Decimal("2.344"),
        ask_qty=Decimal("22.823"),
        update_id=11257589649068,
        event_time=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )
    assert bt.bid == Decimal("63889.10")
    assert bt.ask == Decimal("63889.20")
    assert bt.update_id == 11257589649068


def test_book_ticker_event_type_exists():
    assert EventType.BookTicker.value == "book_ticker"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cp ~/bq/execution/cybotrade/py-cybotrade/cybotrade/cybotrade.cpython-312-darwin.so \
   .claude/worktrees/fix-rate-limit-ip-scope/py-cybotrade/cybotrade/
cd ~/bq/execution/cybotrade && PYTHONPATH=.claude/worktrees/fix-rate-limit-ip-scope/py-cybotrade \
  .venv/bin/pytest .claude/worktrees/fix-rate-limit-ip-scope/py-cybotrade/tests/test_book_ticker_model.py -q
```
Expected: FAIL with `ImportError: cannot import name 'BookTicker' from 'cybotrade.models'`

- [ ] **Step 3: Add the model**

In `models.py`, after the `OrderbookSnapshot` dataclass:

```python
@dataclass(slots=True)
class BookTicker:
    """
    Best bid/ask for one symbol, as pushed by an exchange's top-of-book stream.

    `update_id` is the exchange's own sequence number for the book. Consumers
    must not use it to order or reject updates — that is protocol semantics and
    belongs in the exchange adapter, because the meaning differs per exchange
    (Bybit resets it to 1 on service restart).
    """

    symbol: Symbol
    bid: Decimal
    ask: Decimal
    bid_qty: Decimal
    ask_qty: Decimal
    update_id: int
    event_time: datetime
```

In `io/event.py`, add to `EventType` after `OrderUpdate`:

```python
    BookTicker = "book_ticker"
```

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd .claude/worktrees/fix-rate-limit-ip-scope
git add py-cybotrade/cybotrade/models.py py-cybotrade/cybotrade/io/event.py py-cybotrade/tests/test_book_ticker_model.py
git commit -m "feat: add BookTicker model and event type

The normalised top-of-book payload for public market-data streams. update_id is
carried for adapters and metrics but deliberately not for consumer-side ordering:
its semantics differ per exchange, and Bybit resets it to 1 on service restart."
```

---

## Task 2: cybotrade — `BinancePublicWS` adapter

**Files:**
- Modify: `py-cybotrade/cybotrade/binance/ws.py` (append after `BinancePrivateWS`)
- Modify: `py-cybotrade/cybotrade/binance/__init__.py`
- Test: `py-cybotrade/tests/test_binance_public_ws.py` (create)

**Interfaces:**
- Consumes: `BookTicker`, `EventType.BookTicker` from Task 1.
- Produces: `BinancePublicWS(symbols: list[str], testnet: bool = False, heartbeat_symbol: str = "BTCUSDT")` with attributes `.streams: list[str]`, `.request`, an overridable `.on_event`, and methods `parse_message(payload: str) -> BookTicker | None` and `async start()`.

- [ ] **Step 1: Write the failing test**

Create `py-cybotrade/tests/test_binance_public_ws.py`:

```python
"""
BinancePublicWS: wire format to typed event, nothing else.

Payloads are real combined-stream frames captured from
wss://fstream.binance.com/stream?streams=... on 2026-08-11.
"""

from decimal import Decimal

import pytest

from cybotrade.binance import BinancePublicWS
from cybotrade.models import BookTicker

# Verbatim frame, wrapped as combined streams deliver it
WRAPPED = (
    '{"stream":"btcusdt@bookTicker","data":{"e":"bookTicker",'
    '"u":11257589649068,"s":"BTCUSDT","ps":"BTCUSDT","b":"63889.10",'
    '"B":"2.344","a":"63889.20","A":"22.823","T":1786386890546,'
    '"E":1786386890550}}'
)


def test_subscribes_to_each_symbol_plus_the_heartbeat():
    ws = BinancePublicWS(symbols=["BITOUSDT"])
    assert ws.streams == ["bitousdt@bookTicker", "btcusdt@bookTicker"]


def test_heartbeat_symbol_is_not_duplicated_when_traded():
    ws = BinancePublicWS(symbols=["BTCUSDT", "ETHUSDT"])
    assert ws.streams == ["btcusdt@bookTicker", "ethusdt@bookTicker"]


def test_url_uses_the_verified_combined_stream_form():
    ws = BinancePublicWS(symbols=["BTCUSDT"])
    assert ws.request.url == (
        "wss://fstream.binance.com/stream?streams=btcusdt@bookTicker"
    )


def test_parses_a_wrapped_book_ticker_frame():
    ws = BinancePublicWS(symbols=["BTCUSDT"])
    bt = ws.parse_message(WRAPPED)
    assert isinstance(bt, BookTicker)
    assert str(bt.symbol) == "BTCUSDT"
    assert bt.bid == Decimal("63889.10")
    assert bt.ask == Decimal("63889.20")
    assert bt.bid_qty == Decimal("2.344")
    assert bt.update_id == 11257589649068


def test_parses_an_unwrapped_frame_too():
    """The /ws endpoint with an explicit SUBSCRIBE delivers frames unwrapped."""
    ws = BinancePublicWS(symbols=["BTCUSDT"])
    unwrapped = (
        '{"e":"bookTicker","u":1,"s":"BTCUSDT","b":"1.5","B":"2",'
        '"a":"1.6","A":"3","T":1786386890546,"E":1786386890550}'
    )
    bt = ws.parse_message(unwrapped)
    assert bt is not None and bt.ask == Decimal("1.6")


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        '{"result":null,"id":1}',  # SUBSCRIBE ack
        '{"stream":"x","data":{"e":"aggTrade","s":"BTCUSDT"}}',  # wrong event
        '{"stream":"x","data":{"e":"bookTicker","s":"BTCUSDT"}}',  # missing prices
    ],
)
def test_unparseable_or_irrelevant_frames_return_none_and_do_not_raise(payload):
    ws = BinancePublicWS(symbols=["BTCUSDT"])
    assert ws.parse_message(payload) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd ~/bq/execution/cybotrade && PYTHONPATH=.claude/worktrees/fix-rate-limit-ip-scope/py-cybotrade \
  .venv/bin/pytest .claude/worktrees/fix-rate-limit-ip-scope/py-cybotrade/tests/test_binance_public_ws.py -q
```
Expected: FAIL with `ImportError: cannot import name 'BinancePublicWS'`

- [ ] **Step 3: Implement the adapter**

Append to `binance/ws.py` (it already imports `logging`, `json`, `timedelta`, `Message`, `Request`, `Event`, `EventType`; add `Decimal`, `datetime`, `timezone`, `Symbol`, `BookTicker` as needed):

```python
class BinancePublicWS(ExchangeEvent):
    """
    Public top-of-book stream. Turns `<symbol>@bookTicker` frames into
    BookTicker events and does nothing else: no cache, no staleness policy.

    Unlike BinancePrivateWS this needs no listenKey, so it makes no REST call at
    all — in particular it does not repeat that class's blocking sync
    requests.put on every heartbeat.

    A heartbeat symbol is always subscribed alongside the traded ones. Consumers
    use "any message on this connection" as proof the feed is alive, and
    @bookTicker only pushes on change — so a portfolio of purely illiquid symbols
    would otherwise have nothing proving the socket still works. BTCUSDT's book
    never stops moving (measured 58-212 msgs/sec, 515ms worst gap over 60s).
    """

    def __init__(
        self,
        symbols: list[str],
        testnet: bool = False,
        heartbeat_symbol: str = "BTCUSDT",
    ):
        wanted = [s.lower() for s in symbols]
        if heartbeat_symbol.lower() not in wanted:
            wanted.append(heartbeat_symbol.lower())
        self.symbols = symbols
        self.heartbeat_symbol = heartbeat_symbol
        self.streams = [f"{s}@bookTicker" for s in wanted]
        base = (
            "wss://fstream.binance.com"
            if not testnet
            else "wss://stream.binancefuture.com"
        )
        self.request = Request(url=f"{base}/stream?streams={'/'.join(self.streams)}")
        self.testnet = testnet
        self.set_heartbeat_interval(timedelta(seconds=30))

    async def on_heartbeat(self, sender):
        # Public streams need no keepalive REST call; Binance pings every 3
        # minutes and accepts unsolicited pongs.
        await sender.send(Message.Pong())

    async def on_connected(self, sender) -> None:
        # persist_conn re-runs this on every reconnect. Consumers treat the
        # Subscribed event as "feed restarted" and drop their cache, so no quote
        # can survive a gap in delivery.
        await self.on_event(
            Event(
                event_type=EventType.Subscribed,
                orig={"streams": self.streams},
                data=self.streams,
            )
        )

    async def on_event(self, event) -> None:
        """User-defined; assign to receive events."""
        pass

    def parse_message(self, payload: str) -> "BookTicker | None":
        """
        BookTicker for a top-of-book frame, None for anything else.

        Returns None rather than raising for junk, subscription acks and other
        event types: a bad frame must never take down the feed task.
        """
        try:
            body = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(body, dict):
            return None
        data = body.get("data", body)
        if not isinstance(data, dict) or data.get("e") != "bookTicker":
            return None
        try:
            return BookTicker(
                symbol=Symbol(data["s"]),
                bid=Decimal(data["b"]),
                ask=Decimal(data["a"]),
                bid_qty=Decimal(data["B"]),
                ask_qty=Decimal(data["A"]),
                update_id=int(data["u"]),
                event_time=datetime.fromtimestamp(
                    int(data["E"]) / 1000, tz=timezone.utc
                ),
            )
        except (KeyError, TypeError, ValueError, ArithmeticError) as e:
            logging.warning(f"Binance public WS: unparseable bookTicker frame: {e}")
            return None

    async def on_message(self, message: Message) -> None:
        if not isinstance(message, Message.Text):
            return
        book_ticker = self.parse_message(message.payload)
        if book_ticker is None:
            # Still surfaced: consumers count any frame as liveness.
            await self.on_event(
                Event(event_type=EventType.Unknown, orig=message.payload, data=None)
            )
            return
        await self.on_event(
            Event(
                event_type=EventType.BookTicker,
                orig=message.payload,
                data=book_ticker,
            )
        )

    async def start(self):
        async for item in self._stream():
            try:
                await self.on_message(item)
            except Exception as e:
                logging.warning(f"Binance public WS encountered an Exception: {e}")
                continue
```

`_stream()` is inherited from `ExchangeEvent` (`io/exchange.py:250`) and defaults to
`persist_conn`, which reconnects with growing backoff and re-runs `on_connected` on
each attempt. That is why resubscription needs no code here — and why
`on_connected` emitting `Subscribed` is what tells the consumer to drop its cache.

In `binance/__init__.py`:

```python
from .ws import BinancePrivateWS, BinancePublicWS

__all__ = [
    "BinanceLinearClient",
    "BinanceError",
    "BinancePrivateWS",
    "BinancePublicWS",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd .claude/worktrees/fix-rate-limit-ip-scope
git add py-cybotrade/cybotrade/binance/ws.py py-cybotrade/cybotrade/binance/__init__.py py-cybotrade/tests/test_binance_public_ws.py
git commit -m "feat: add BinancePublicWS top-of-book stream adapter

Subscribes <symbol>@bookTicker for the requested symbols plus a heartbeat symbol
(BTCUSDT, deduplicated). Consumers prove the feed is alive from 'any frame on
this connection', and @bookTicker only pushes on change, so a portfolio of purely
illiquid symbols would otherwise have nothing to prove the socket still works.

Parsing returns None for junk, acks and other event types rather than raising,
and unparseable frames are still surfaced as Unknown so they count as liveness.
Needs no listenKey, so unlike BinancePrivateWS it makes no REST call on heartbeat."
```

---

## Task 3: adrs — `PriceFeed`

**Files:**
- Create: `adrs/oms/price_feed.py`
- Test: `tests/test_price_feed.py` (create)

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (deliberately — this must land before the cybotrade release).
- Produces:
  - `Quote(bid: Decimal, ask: Decimal, received_at: float)` with property `mid -> Decimal`
  - `PriceFeed(heartbeat_max_age_sec: float | None = 2.0, quote_max_age_sec: float = 60.0, clock: Callable[[], float] = time.monotonic)`
  - `PriceFeed.apply(symbol: Symbol, bid: Decimal, ask: Decimal) -> bool`
  - `PriceFeed.note_message() -> None`
  - `PriceFeed.get(symbol: Symbol) -> Quote | None`
  - `PriceFeed.clear() -> None`
  - `PriceFeed.invalidate(symbol: Symbol) -> None`
  - `PriceFeed.stats() -> dict[str, int | float | None]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_feed.py`:

```python
"""
PriceFeed staleness policy.

The load-bearing behaviour is that liveness belongs to the *connection* while
freshness belongs to the quote. @bookTicker pushes on every top-of-book change,
so silence on a healthy socket means the book has not moved and an old quote is
still current. Gating on per-symbol age instead would send every quiet symbol
back to REST each tick to re-fetch a price that had not changed, which is the
entire cost this feed exists to remove.

Clock is injected so none of this needs sleeps.
"""

from decimal import Decimal

from cybotrade import Symbol

from adrs.oms.price_feed import PriceFeed, Quote

BTC = Symbol("BTCUSDT")
THIN = Symbol("BITOUSDT")


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _feed(clock: FakeClock, **kw) -> PriceFeed:
    return PriceFeed(clock=clock, **kw)


def test_serves_a_fresh_quote():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    quote = feed.get(BTC)
    assert quote is not None
    assert (quote.bid, quote.ask) == (Decimal("100"), Decimal("102"))


def test_mid_matches_the_cybotrade_formula():
    """Feed and REST paths must not quote differently for the same book."""
    quote = Quote(bid=Decimal("63889.10"), ask=Decimal("63889.20"), received_at=0.0)
    assert quote.mid == (Decimal("63889.10") + Decimal("63889.20")) / Decimal("2.0")


def test_quiet_symbol_is_served_while_liveness_is_fresh():
    """The reason this design exists: an idle book costs no REST weight."""
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(30.0)  # thin book has not ticked for 30s...
    feed.apply(BTC, Decimal("100"), Decimal("102"))  # ...but BTC keeps arriving
    assert feed.get(THIN) is not None


def test_quote_is_withheld_once_liveness_goes_stale():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    clock.advance(2.5)  # past heartbeat_max_age of 2.0s
    assert feed.get(BTC) is None


def test_any_symbol_refreshes_liveness_for_every_symbol():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(1.5)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    clock.advance(1.5)  # 3s since THIN's quote, 1.5s since any message
    assert feed.get(THIN) is not None


def test_note_message_refreshes_liveness_without_storing_a_quote():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(1.5)
    feed.note_message()  # e.g. an unparseable frame: proves the socket delivers
    clock.advance(1.5)
    assert feed.get(THIN) is not None
    assert feed.get(BTC) is None  # never had a quote


def test_backstop_withholds_a_quote_even_while_liveness_is_fresh():
    """Covers the one failure liveness cannot see: a dead single subscription."""
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    for _ in range(70):  # BTC keeps the socket provably alive for 70s
        clock.advance(1.0)
        feed.apply(BTC, Decimal("100"), Decimal("102"))
    assert feed.get(BTC) is not None
    assert feed.get(THIN) is None  # past the 60s backstop


def test_unseen_symbol_returns_none():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    assert feed.get(THIN) is None


def test_clear_drops_quotes_and_resets_liveness():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    feed.clear()
    assert feed.get(BTC) is None
    # Even a fresh quote for another symbol must not resurrect the cleared one
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    assert feed.get(BTC) is None


def test_invalidate_affects_only_that_symbol():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    feed.invalidate(THIN)
    assert feed.get(THIN) is None
    assert feed.get(BTC) is not None


def test_crossed_book_is_rejected_but_still_counts_as_liveness():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    clock.advance(1.5)
    assert feed.apply(BTC, Decimal("103"), Decimal("102")) is False  # bid > ask
    quote = feed.get(BTC)
    assert quote is not None
    assert quote.ask == Decimal("102")  # the good quote survived
    assert quote.bid == Decimal("100")


def test_equal_bid_ask_and_zero_sides_are_rejected():
    clock = FakeClock()
    feed = _feed(clock)
    assert feed.apply(BTC, Decimal("100"), Decimal("100")) is False
    assert feed.apply(BTC, Decimal("0"), Decimal("102")) is False
    assert feed.apply(BTC, Decimal("100"), Decimal("0")) is False
    assert feed.get(BTC) is None


def test_heartbeat_requirement_can_be_disabled_for_bybit():
    """
    Bybit's orderbook.1 re-pushes a snapshot after 3s of no change, so it carries
    per-symbol liveness and needs no heartbeat subscription.
    """
    clock = FakeClock()
    feed = _feed(clock, heartbeat_max_age_sec=None, quote_max_age_sec=4.0)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(3.0)
    assert feed.get(THIN) is not None
    clock.advance(2.0)  # 5s > 4s per-symbol cap
    assert feed.get(THIN) is None


def test_stats_count_fallbacks_by_reason():
    clock = FakeClock()
    feed = _feed(clock)
    feed.get(BTC)  # no liveness at all yet
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    feed.get(THIN)  # unseen
    feed.get(BTC)  # served
    stats = feed.stats()
    assert stats["served"] == 1
    assert stats["fallback_no_liveness"] == 1
    assert stats["fallback_unseen"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd ~/bq/research/adrs/.claude/worktrees/fix-rate-limit-ip-scope
~/bq/research/adrs/.venv/bin/pytest tests/test_price_feed.py -q -p no:cacheprovider --no-cov
```
Expected: FAIL with `ModuleNotFoundError: No module named 'adrs.oms.price_feed'`

- [ ] **Step 3: Implement `PriceFeed`**

Create `adrs/oms/price_feed.py`:

```python
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from cybotrade import Symbol

logger = logging.getLogger(__name__)

# How long since *any* frame on the connection before every quote is withheld.
#
# This is the liveness signal, not a per-quote freshness limit. @bookTicker
# pushes on every top-of-book change, so silence on a healthy socket means the
# book has not moved and the last quote is still the current one. The heartbeat
# subscription (BTCUSDT) was measured at 58-212 frames/sec with a 515ms worst
# gap over 60s, so 2s is ~4x the observed worst case: tight enough to catch a
# wedged socket inside one placement tick, loose enough to survive a scheduling
# hiccup.
DEFAULT_HEARTBEAT_MAX_AGE_SEC = 2.0

# Backstop for the single failure liveness cannot see: one symbol's subscription
# dying while the connection keeps delivering others. Deliberately generous — a
# tight value would reintroduce the REST-per-tick cost on quiet symbols that this
# feed exists to remove. The trade is up to this long quoting from a dead
# subscription, versus REST on every tick for every quiet symbol.
DEFAULT_QUOTE_MAX_AGE_SEC = 60.0


@dataclass(frozen=True, slots=True)
class Quote:
    bid: Decimal
    ask: Decimal
    received_at: float  # time.monotonic() at parse time

    @property
    def mid(self) -> Decimal:
        # Must match cybotrade's ExchangeClient.get_current_price exactly, or the
        # feed and REST paths would quote differently for the same book.
        return (self.bid + self.ask) / Decimal("2.0")


class PriceFeed:
    """
    Top-of-book cache with a staleness policy. Pure: it makes no exchange calls,
    knows nothing about websockets, and decides nothing about what to do on a
    miss — callers fall back to REST.

    Set `heartbeat_max_age_sec=None` for a feed whose stream re-pushes an
    unchanged book on a timer (Bybit's orderbook.1 does, after 3s), in which case
    per-symbol age alone is a sufficient guard and `quote_max_age_sec` should be
    tightened accordingly.
    """

    def __init__(
        self,
        heartbeat_max_age_sec: float | None = DEFAULT_HEARTBEAT_MAX_AGE_SEC,
        quote_max_age_sec: float = DEFAULT_QUOTE_MAX_AGE_SEC,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._heartbeat_max_age_sec = heartbeat_max_age_sec
        self._quote_max_age_sec = quote_max_age_sec
        self._clock = clock
        self._quotes: dict[Symbol, Quote] = {}
        self._last_message_at: float | None = None
        self._served = 0
        self._fallback_no_liveness = 0
        self._fallback_unseen = 0
        self._fallback_backstop = 0
        self._rejected = 0

    def note_message(self) -> None:
        """
        Record that the connection delivered something.

        Called for every frame, including ones that do not yield a quote: a
        malformed frame still proves the socket is alive.
        """
        self._last_message_at = self._clock()

    def apply(self, symbol: Symbol, bid: Decimal, ask: Decimal) -> bool:
        """Store a quote. False (and no store) if the book is crossed or zero."""
        self.note_message()
        if bid <= 0 or ask <= 0 or bid >= ask:
            self._rejected += 1
            logger.warning(
                f"[PRICE_FEED] Rejected implausible book for {symbol}: "
                f"bid={bid} ask={ask}"
            )
            return False
        self._quotes[symbol] = Quote(bid=bid, ask=ask, received_at=self._clock())
        return True

    def get(self, symbol: Symbol) -> Quote | None:
        """The current quote, or None when the caller must fall back to REST."""
        now = self._clock()
        if self._heartbeat_max_age_sec is not None:
            if (
                self._last_message_at is None
                or now - self._last_message_at > self._heartbeat_max_age_sec
            ):
                self._fallback_no_liveness += 1
                return None
        quote = self._quotes.get(symbol)
        if quote is None:
            self._fallback_unseen += 1
            return None
        if now - quote.received_at > self._quote_max_age_sec:
            self._fallback_backstop += 1
            return None
        self._served += 1
        return quote

    def clear(self) -> None:
        """
        Drop every quote and reset liveness. Called on reconnect: after a gap in
        delivery no cached quote can be trusted, so each symbol takes one REST
        read before rejoining the feed.
        """
        self._quotes.clear()
        self._last_message_at = None

    def invalidate(self, symbol: Symbol) -> None:
        """Drop one symbol. Unused on Binance; Bybit needs it on a sequence gap."""
        self._quotes.pop(symbol, None)

    def stats(self) -> dict[str, int | float | None]:
        now = self._clock()
        return {
            "served": self._served,
            "fallback_no_liveness": self._fallback_no_liveness,
            "fallback_unseen": self._fallback_unseen,
            "fallback_backstop": self._fallback_backstop,
            "rejected": self._rejected,
            "tracked_symbols": len(self._quotes),
            "liveness_age_sec": (
                None
                if self._last_message_at is None
                else now - self._last_message_at
            ),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_price_feed.py -q -p no:cacheprovider --no-cov
```
Expected: 14 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check adrs/oms/price_feed.py tests/test_price_feed.py
ruff format adrs/oms/price_feed.py tests/test_price_feed.py
git add adrs/oms/price_feed.py tests/test_price_feed.py
git commit -m "feat(oms): add PriceFeed top-of-book cache and staleness policy

Liveness belongs to the connection, freshness to the quote. @bookTicker pushes on
every top-of-book change, so silence on a healthy socket means the book has not
moved and the last quote is still current; gating on per-symbol age instead would
send every quiet symbol back to REST each tick to re-fetch an unchanged price,
which is the whole cost this exists to remove.

So a quote is served whenever the connection is proven live, with per-symbol age
kept only as a generous 60s backstop for the one case liveness cannot see: a
single subscription dying while the connection keeps delivering others.

Pure and clock-injected: no exchange calls, no websocket knowledge, no cybotrade
event types -- so it lands before the cybotrade release that adds them."
```

---

## Task 4: adrs — wire the read paths to the feed

**Files:**
- Modify: `adrs/oms/ops/order_utils.py:23-41` (`get_order_book`)
- Modify: `adrs/oms/ops/order_executer.py:92-105` (`__init__`), `:373-379`, `:452-464` (`get_current_price`), `:564-570`
- Modify: `adrs/oms/ops/order_placement_manager.py:51-76` (`__init__`), `:606-612`
- Test: `tests/test_price_feed_wiring.py` (create)

**Interfaces:**
- Consumes: `PriceFeed`, `Quote` from Task 3.
- Produces: `OrderUtils.get_order_book(..., price_feed: PriceFeed | None = None)`; `OrderExecutor.__init__(..., price_feed: PriceFeed | None = None)` setting `self.price_feed`; `OrderPlacementManager.__init__(..., price_feed: PriceFeed | None = None)` setting `self.price_feed` and forwarding it to `executor_cls`.

`price_feed` defaults to `None` everywhere, meaning "no feed, go straight to REST". That keeps every existing construction site and test working unchanged, so this task lands without touching them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_feed_wiring.py`:

```python
"""
The read paths must prefer the feed and fall back to REST *through* reserve().

The reserve() detail is not incidental: it is what makes the worst case equal
today's cost instead of a stampede when the feed is down.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.models import Exchange, Level, OrderbookSnapshot

from adrs.oms.ops.order_utils import OrderUtils
from adrs.oms.price_feed import PriceFeed

BTC = Symbol("BTCUSDT")


class SpyRateLimiter:
    def __init__(self):
        self.reserved = []

    @asynccontextmanager
    async def reserve(self, endpoint):
        self.reserved.append(endpoint)
        yield


def _snapshot(bid: str, ask: str) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        symbol=BTC,
        last_update_time=datetime(2026, 8, 11, tzinfo=timezone.utc),
        last_update_id=1,
        bids=[Level(price=Decimal(bid), quantity=Decimal("1"))],
        asks=[Level(price=Decimal(ask), quantity=Decimal("1"))],
        exchange=Exchange.BINANCE_LINEAR,
        orig=None,
    )


def test_feed_hit_never_touches_the_exchange():
    feed = PriceFeed()
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    exchange = MagicMock()
    exchange.get_orderbook_snapshot = AsyncMock()
    limiter = SpyRateLimiter()

    result = asyncio.run(
        OrderUtils.get_order_book(
            exchange=exchange,
            pair=BTC,
            need_log=False,
            rate_limiter=limiter,
            price_feed=feed,
        )
    )
    assert result == [Decimal("100"), Decimal("102")]
    exchange.get_orderbook_snapshot.assert_not_awaited()
    assert limiter.reserved == []  # no weight spent


def test_feed_miss_falls_back_through_reserve():
    feed = PriceFeed()  # empty: no liveness, so every get() misses
    exchange = MagicMock()
    exchange.get_orderbook_snapshot = AsyncMock(return_value=_snapshot("99", "101"))
    limiter = SpyRateLimiter()

    result = asyncio.run(
        OrderUtils.get_order_book(
            exchange=exchange,
            pair=BTC,
            need_log=False,
            rate_limiter=limiter,
            price_feed=feed,
        )
    )
    assert result == [Decimal("99"), Decimal("101")]
    exchange.get_orderbook_snapshot.assert_awaited_once()
    assert len(limiter.reserved) == 1  # went through the rate limiter


def test_no_feed_configured_behaves_exactly_as_before():
    exchange = MagicMock()
    exchange.get_orderbook_snapshot = AsyncMock(return_value=_snapshot("99", "101"))
    limiter = SpyRateLimiter()

    result = asyncio.run(
        OrderUtils.get_order_book(
            exchange=exchange, pair=BTC, need_log=False, rate_limiter=limiter
        )
    )
    assert result == [Decimal("99"), Decimal("101")]
    assert len(limiter.reserved) == 1


def test_feed_and_rest_agree_on_current_price():
    """
    Equivalence: switching the source must not move our quotes. cybotrade's
    get_current_price returns the mid of best bid and best ask.
    """
    feed = PriceFeed()
    feed.apply(BTC, Decimal("63889.10"), Decimal("63889.20"))
    quote = feed.get(BTC)
    assert quote is not None
    rest_equivalent = (Decimal("63889.10") + Decimal("63889.20")) / Decimal("2.0")
    assert quote.mid == rest_equivalent
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_price_feed_wiring.py -q -p no:cacheprovider --no-cov
```
Expected: FAIL with `TypeError: get_order_book() got an unexpected keyword argument 'price_feed'`

- [ ] **Step 3: Add the feed check to `get_order_book`**

In `adrs/oms/ops/order_utils.py`, extend the `TYPE_CHECKING` block and the signature:

```python
if TYPE_CHECKING:
    from adrs.oms.price_feed import PriceFeed
    from adrs.oms.rate_limit.rate_limiter import RateLimiter
```

```python
    @staticmethod
    async def get_order_book(
        exchange: ExchangeClient,
        pair: Symbol,
        need_log: bool,
        rate_limiter: "RateLimiter",
        endpoint: Endpoints = Endpoints.GET_ORDERBOOK_SNAPSHOT,
        price_feed: "PriceFeed | None" = None,
    ) -> list[Decimal]:
        # The websocket feed answers for free when it is proven live. A miss
        # (feed down, symbol unseen, or past the staleness backstop) falls
        # through to REST below, which is exactly the old behaviour.
        if price_feed is not None:
            quote = price_feed.get(pair)
            if quote is not None:
                if need_log:
                    logger.info(f"best_bid: {quote.bid}, best_ask: {quote.ask} (ws)")
                return [quote.bid, quote.ask]

        # reserve() waits for a rate-limit slot rather than failing, so
        # contention no longer needs a retry loop. Genuine fetch errors
        # propagate to the caller (placement backlogs it, expiry skips it).
        async with rate_limiter.reserve(endpoint=endpoint):
            orderbook = await exchange.get_orderbook_snapshot(symbol=pair)
        (best_bid, best_ask) = (
            max(map(lambda level: level.price, orderbook.bids)),
            min(map(lambda level: level.price, orderbook.asks)),
        )
        if need_log:
            logger.info(f"best_bid: {best_bid}, best_ask: {best_ask}")
        return [best_bid, best_ask]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_price_feed_wiring.py -q -p no:cacheprovider --no-cov
```
Expected: 4 passed.

- [ ] **Step 5: Thread `price_feed` through the executor and manager**

In `adrs/oms/ops/order_executer.py`, add the import and constructor parameter:

```python
from adrs.oms.price_feed import PriceFeed
```

```python
    def __init__(
        self,
        config_manager: ConfigManager,
        order_pools: OrderPoolHandler,
        rate_limiter: RateLimiter,
        error_policy: ExchangeErrorPolicy,
        price_feed: PriceFeed | None = None,
    ):
        self.exchange: ExchangeClient = config_manager.exchange
        self.config: Config = config_manager.config
        self.order_pools = order_pools
        self.symbol_infos = config_manager.symbol_infos
        self.rate_limiter = rate_limiter
        self.error_policy = error_policy
        self.price_feed = price_feed
        self.package_id = make_package_id(self.config.portfolio_id)
```

Add `price_feed=self.price_feed` to both `OrderUtils.get_order_book(...)` calls (lines ~373 and ~564).

Replace `get_current_price` (lines ~452-464) with:

```python
    async def get_current_price(self, symbol: Symbol) -> Decimal | None:
        """
        Canonical current-price fetch: reserves a rate-limit slot and returns
        None on any failure. All price reads go through here so behaviour is
        consistent (waits under contention, never raises to the caller).

        Prefers the websocket feed, which costs no request weight. The mid
        formula matches cybotrade's get_current_price so both sources agree.
        """
        if self.price_feed is not None:
            quote = self.price_feed.get(symbol)
            if quote is not None:
                return quote.mid

        endpoint = Endpoints.GET_ORDERBOOK_SNAPSHOT
        try:
            async with self.rate_limiter.reserve(endpoint=endpoint):
                return await self.exchange.get_current_price(symbol=symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch current price due to {e}")
            return None
```

In `adrs/oms/ops/order_placement_manager.py`, add the import, then the constructor parameter and forwarding:

```python
from adrs.oms.price_feed import PriceFeed
```

```python
    def __init__(
        self,
        position: PositionManager,
        config: ConfigManager,
        rate_limiter: RateLimiter,
        error_policy: ExchangeErrorPolicy,
        executor_cls: type[OrderExecutor] = OrderExecutor,
        price_feed: PriceFeed | None = None,
    ) -> None:
        self.position = position
        self.config_manager = config
        self.price_feed = price_feed
```

then pass `price_feed=price_feed` into the `self.executor = executor_cls(...)` call, and add `price_feed=self.price_feed` to the `OrderUtils.get_order_book(...)` call at line ~606.

- [ ] **Step 6: Run the full suite to confirm nothing regressed**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/ -q -p no:cacheprovider --ignore=tests/integration \
  --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py --no-cov
```
Expected: all pass. (`test_datamap.py` and `test_dataloader.py` are excluded because they need an untracked `credentials.json` and a local clickhouse; they fail on any fresh worktree.)

- [ ] **Step 7: Lint and commit**

```bash
ruff check adrs/oms tests/
ruff format adrs/oms tests/test_price_feed_wiring.py
git add adrs/oms/ops/order_utils.py adrs/oms/ops/order_executer.py \
        adrs/oms/ops/order_placement_manager.py tests/test_price_feed_wiring.py
git commit -m "feat(oms): read top-of-book from the price feed before REST

All four depth-read sites funnel through get_order_book and get_current_price, so
this is a two-function change plus plumbing price_feed to their callers. A feed
hit costs no request weight; a miss falls through to the unchanged REST path,
still wrapped in reserve() -- which is what keeps the worst case equal to today's
cost rather than a stampede when the feed is down.

price_feed defaults to None (meaning: go straight to REST) so every existing
construction site and test keeps working untouched."
```

---

## Task 5: adrs — feed task lifecycle in the OMS

**Blocked on:** cybotrade 2.0.19 published with Tasks 1-2, and `pyproject.toml` bumped to `cybotrade>=2.0.19`.

**Files:**
- Modify: `adrs/oms/oms.py` — imports, `__init__` (~line 98-131), `run()` (~line 711-717), `on_refresh_config` (~line 272-300), `_handle_shutdown` (~line 215)
- Modify: `pyproject.toml:21`
- Test: `tests/test_price_feed_lifecycle.py` (create)

**Interfaces:**
- Consumes: `BinancePublicWS`, `EventType.BookTicker`, `BookTicker` (Tasks 1-2); `PriceFeed` (Task 3); `price_feed` parameters (Task 4).
- Produces: `OMS.price_feed: PriceFeed`, `OMS.price_feed_task: asyncio.Task | None`, `OMS.on_price_feed_event(event: Event) -> None`, `OMS._start_price_feed() -> None`, `OMS._stop_price_feed() -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_price_feed_lifecycle.py`:

```python
"""
Event translation and cache invalidation for the OMS-owned price feed.

The reconnect rule is the one that makes the staleness guard sound: persist_conn
re-runs on_connected on every reconnect, and Binance forces a disconnect every 24
hours, so this path runs daily in normal operation rather than only under failure.
"""

from datetime import datetime, timezone
from decimal import Decimal

from cybotrade import Symbol
from cybotrade.io.event import Event, EventType
from cybotrade.models import BookTicker

from adrs.oms.oms import OMS
from adrs.oms.price_feed import PriceFeed

BTC = Symbol("BTCUSDT")


def _oms_with_feed() -> OMS:
    """OMS with __init__ skipped; only the price-feed collaborators are needed."""
    oms = object.__new__(OMS)
    oms.price_feed = PriceFeed()
    return oms


def _book_ticker(bid: str, ask: str) -> BookTicker:
    return BookTicker(
        symbol=BTC,
        bid=Decimal(bid),
        ask=Decimal(ask),
        bid_qty=Decimal("1"),
        ask_qty=Decimal("1"),
        update_id=1,
        event_time=datetime(2026, 8, 11, tzinfo=timezone.utc),
    )


def test_book_ticker_event_becomes_a_quote():
    oms = _oms_with_feed()
    oms.on_price_feed_event(
        Event(event_type=EventType.BookTicker, orig="{}", data=_book_ticker("100", "102"))
    )
    quote = oms.price_feed.get(BTC)
    assert quote is not None and quote.bid == Decimal("100")


def test_subscribed_event_clears_the_cache():
    """A Subscribed event means the socket (re)connected: nothing cached is trustworthy."""
    oms = _oms_with_feed()
    oms.on_price_feed_event(
        Event(event_type=EventType.BookTicker, orig="{}", data=_book_ticker("100", "102"))
    )
    assert oms.price_feed.get(BTC) is not None
    oms.on_price_feed_event(
        Event(event_type=EventType.Subscribed, orig="{}", data=["btcusdt@bookTicker"])
    )
    assert oms.price_feed.get(BTC) is None


def test_unknown_event_refreshes_liveness_without_creating_a_quote():
    oms = _oms_with_feed()
    oms.on_price_feed_event(Event(event_type=EventType.Unknown, orig="junk", data=None))
    assert oms.price_feed.get(BTC) is None
    assert oms.price_feed.stats()["liveness_age_sec"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_price_feed_lifecycle.py -q -p no:cacheprovider --no-cov
```
Expected: FAIL with `AttributeError: 'OMS' object has no attribute 'on_price_feed_event'`

- [ ] **Step 3: Bump the dependency**

In `pyproject.toml` line 21, change `"cybotrade>=2.0.18",` to `"cybotrade>=2.0.19",`, then:

```bash
uv lock
```

- [ ] **Step 4: Implement the lifecycle**

In `adrs/oms/oms.py`, add imports:

```python
from cybotrade.binance import BinancePublicWS
from cybotrade.io.event import Event, EventType
from cybotrade.models import Exchange

from adrs.oms.price_feed import PriceFeed
```

In `__init__`, before `self.opm = self.opm_cls(...)`:

```python
        # Owned here, not by the OPM: the feed serves both the executor's reads
        # and the signal recompute, and a later Bybit adapter will share it.
        self.price_feed = PriceFeed()
```

and add `price_feed=self.price_feed,` to the `self.opm_cls(...)` call. Next to the existing `self.exchange_events_task` initialisation:

```python
        # Initialised in run(); set here so on_refresh_config and shutdown are
        # safe to call before run() starts (e.g. in tests).
        self.price_feed_task: asyncio.Task | None = None
```

Add the handler and lifecycle methods:

```python
    def on_price_feed_event(self, event: Event) -> None:
        """
        Translate public market-data events into the feed.

        Every event refreshes liveness, including ones that carry no quote: an
        unparseable frame still proves the socket is delivering. Subscribed means
        the connection just (re)established, so nothing cached survives it.
        """
        match event.event_type:
            case EventType.BookTicker:
                book_ticker = event.data
                self.price_feed.apply(
                    book_ticker.symbol, book_ticker.bid, book_ticker.ask
                )
            case EventType.Subscribed:
                logger.info(f"[PRICE_FEED] (Re)subscribed to {event.data}, clearing")
                self.price_feed.clear()
                self.price_feed.note_message()
            case _:
                self.price_feed.note_message()

    def _start_price_feed(self) -> None:
        """
        Start the public feed for the configured symbols.

        Only Binance has an adapter today; on any other exchange the OMS runs
        exactly as before, reading prices from REST.
        """
        if self.config.config.credentials.exchange is not Exchange.BINANCE_LINEAR:
            logger.info(
                "[PRICE_FEED] No public feed for "
                f"{self.config.config.credentials.exchange}, using REST prices"
            )
            return
        symbols = list(self.config.config.base_asset_to_symbol_table.values())
        self.price_feed_ws = BinancePublicWS(symbols=symbols)
        self.price_feed_ws.on_event = self.on_price_feed_event
        self.price_feed_task = asyncio.create_task(self.price_feed_ws.start())
        logger.info(f"[PRICE_FEED] Started for {self.price_feed_ws.streams}")

    def _stop_price_feed(self) -> None:
        if self.price_feed_task is not None:
            self.price_feed_task.cancel()
            self.price_feed_task = None
        self.price_feed.clear()
```

In `run()`, next to the existing exchange-events task creation:

```python
            self.exchange_events_task = asyncio.create_task(self.exchange_event.start())
            self._start_price_feed()
            await self.scheduler.start()
```

In `on_refresh_config`, inside the `base_asset_to_symbol_table` change branch, after `await self.init()`:

```python
            # The stream list is encoded in the feed's URL, so a changed symbol
            # table needs a new connection. clear() happens in _stop_price_feed:
            # quotes for symbols we may no longer trade must not linger.
            self._stop_price_feed()
            self._start_price_feed()
```

At the start of `_handle_shutdown`, after the log line:

```python
        self._stop_price_feed()
```

- [ ] **Step 5: Add the event-time divergence warning**

The spec's *Error handling* section calls for this, and it is not cosmetic: quote
age is stamped when the frame is *parsed*, so a stalled event loop can leave a
frame queued and then stamp it "now", making an older price look fresh. The
`listenKey` keepalive does a blocking sync `requests.put` every 30s today
(cybotrade `TODO.md`), so loop stalls are real rather than theoretical.

Computing this in the OMS rather than in `PriceFeed` keeps the feed pure — the
synced clock lives on the rate limiter, which the OMS already holds.

Append to `tests/test_price_feed_lifecycle.py`:

```python
def test_large_event_time_divergence_is_logged(caplog):
    """
    A frame whose exchange timestamp is far older than 'now' means the loop
    stalled or our clock has drifted; either way the quote's age understates it.
    """
    oms = _oms_with_feed()
    oms.rate_limiter = MagicMock()
    # Exchange stamped the frame 5s before our synced clock reads
    oms.rate_limiter.get_synced_time_ms = MagicMock(
        return_value=int(datetime(2026, 8, 11, tzinfo=timezone.utc).timestamp() * 1000)
        + 5_000
    )
    with caplog.at_level(logging.WARNING):
        oms.on_price_feed_event(
            Event(
                event_type=EventType.BookTicker,
                orig="{}",
                data=_book_ticker("100", "102"),
            )
        )
    assert "divergence" in caplog.text.lower()
    # The quote is still stored: this is an observability signal, not a guard
    assert oms.price_feed.get(BTC) is not None
```

Add `import logging` and `from unittest.mock import MagicMock` to that test file's
imports.

In `oms.py`, add the threshold constant near the top:

```python
# A frame this much older than our synced clock means the event loop stalled or
# our clock drifted. Reported rather than enforced: the staleness guard stays on
# the monotonic clock, which is immune to both.
PRICE_FEED_EVENT_TIME_DIVERGENCE_WARN_MS = 2_000
```

and extend the `BookTicker` branch of `on_price_feed_event`:

```python
            case EventType.BookTicker:
                book_ticker = event.data
                self._warn_on_event_time_divergence(book_ticker)
                self.price_feed.apply(
                    book_ticker.symbol, book_ticker.bid, book_ticker.ask
                )
```

with:

```python
    def _warn_on_event_time_divergence(self, book_ticker) -> None:
        try:
            exchange_ms = int(book_ticker.event_time.timestamp() * 1000)
            divergence_ms = self.rate_limiter.get_synced_time_ms() - exchange_ms
        except Exception:  # never let a metric break the feed
            return
        if divergence_ms > PRICE_FEED_EVENT_TIME_DIVERGENCE_WARN_MS:
            logger.warning(
                f"[PRICE_FEED] Event-time divergence {divergence_ms}ms for "
                f"{book_ticker.symbol}: the loop stalled or the clock has drifted, "
                f"so quote age understates real staleness"
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_price_feed_lifecycle.py -q -p no:cacheprovider --no-cov
```
Expected: 4 passed.

- [ ] **Step 7: Run the full suite**

Run:
```bash
~/bq/research/adrs/.venv/bin/pytest tests/ -q -p no:cacheprovider --ignore=tests/integration \
  --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py --no-cov
```
Expected: all pass.

- [ ] **Step 8: Lint and commit**

```bash
ruff check adrs/oms tests/
ruff format adrs/oms tests/test_price_feed_lifecycle.py
git add adrs/oms/oms.py pyproject.toml uv.lock tests/test_price_feed_lifecycle.py
git commit -m "feat(oms): run the Binance public price feed alongside the OMS

Starts BinancePublicWS in run(), cancels it on shutdown, and rebuilds it when the
symbol table changes (the stream list is encoded in the connection URL).

Every feed event refreshes liveness, including ones carrying no quote -- an
unparseable frame still proves the socket delivers. A Subscribed event means the
connection just (re)established, so the cache is dropped: persist_conn re-runs
on_connected on every reconnect and Binance forces a disconnect every 24 hours,
so this path runs daily in normal operation, not only under failure.

Non-Binance exchanges have no adapter yet and keep reading prices from REST."
```

---

## Task 6: verify on the shard and measure

**Files:** none modified. This is the operational gate before widening the rollout.

**Interfaces:** consumes the deployed result of Tasks 1-5.

- [ ] **Step 1: Re-run the stream verification from the shard**

Per the spec's *Environment caveat*, the design was measured from a network where only order-book streams are delivered — `@aggTrade`, `@markPrice` and `@ticker` were accepted by `SUBSCRIBE` yet delivered nothing, which is impossible on a live market and therefore local filtering.

Run on a shard pod:
```bash
python scripts/verify_binance_price_feed.py --seconds 90
```
Confirm `<thin>@bookTicker` and `btcusdt@bookTicker` both deliver. The heartbeat design needs only `@bookTicker`, so this is a confirmation rather than a decision point — but if the shard cannot receive `@bookTicker` either, stop and reassess, because nothing downstream works.

- [ ] **Step 2: Deploy to one tenant and watch the metrics**

There is no config flag, so staging is by deployment scope. On the first tenant, confirm from `PriceFeed.stats()`:

- `fallback_no_liveness` near zero in steady state — a persistently non-zero value means the heartbeat is not arriving and every read is on REST
- `fallback_backstop` at zero — non-zero means a symbol's subscription is dying while the connection stays healthy, the one case liveness cannot see
- `rejected` at zero — non-zero means crossed or zero books are arriving and needs investigating before widening
- `liveness_age_sec` consistently under 2.0

- [ ] **Step 3: Confirm the weight reduction**

Compare `x-mbx-used-weight-1m` before and after on the same tenant. Expected: active-churn weight drops from ~300-900/min toward ~50/min, since the depth reads were essentially the entire gap between a quiet and a busy OMS. If weight does not fall materially, the feed is being bypassed — check `fallback_*` counters first.

- [ ] **Step 4: Widen the rollout**

Once one tenant looks right, widen. Rollback is revert and redeploy.

---

## Notes for the implementer

**Do not add an `update_id` ordering check to `PriceFeed`.** It was in an earlier draft of the design and is wrong: Bybit resets `u` to 1 on service restart, so "reject unless greater" would reject everything afterwards and cause permanent staleness. Ordering is protocol semantics and belongs in the exchange adapter.

**Do not tighten the 60s backstop.** It looks alarmingly loose, and it is deliberate. Tightening it reintroduces the REST-per-tick cost on quiet symbols, which is the entire problem this feed solves. The spec's *Error handling* section states the trade explicitly.

**Do not make the heartbeat subscription optional on Binance.** Without it, a portfolio of only illiquid symbols has nothing proving the socket is alive, and every read falls back to REST.

**If `parse_message` needs a field Binance stopped sending,** return `None` and log rather than raising. A single bad frame must never kill the feed task; the consumer degrades to REST on its own.

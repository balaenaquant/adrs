# Bybit Price Feed and Position Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve Bybit top-of-book from `orderbook.1.<SYMBOL>` and maintain `PositionManager.exchange` from Bybit's `position` topic, reusing the `PriceFeed` and `apply_stream_positions` machinery the Binance work already shipped.

**Architecture:** Two new cybotrade pieces — a `case "position"` in the existing `BybitPrivateWS` topic dispatch, and a new `BybitPublicWS` mirroring `BinancePublicWS`. Both emit event types that already exist (`EventType.PositionUpdate`, `EventType.BookTicker`), so adrs's consumers are untouched. adrs changes only to subscribe the `position` topic and to select the public-feed adapter by exchange.

**Tech Stack:** Python (adrs 3.14, cybotrade 3.12), `cybotrade` websocket adapters, `pytest`, `ruff`, `Decimal` for all quantities.

Spec: `docs/superpowers/specs/2026-08-14-bybit-price-feed-and-position-stream-design.md`. Read *Gate experiments*, *Error handling* and *Balances* before starting.

## Global Constraints

- **Quantities are always `Decimal`, never float.**
- **Bybit's `size` is unsigned and absolute; the sign comes from `side`**: `"Buy"` → positive, `"Sell"` → negative.
- **`side == ""` is the normal close frame** (measured: `size='0'`, `side=''`, `entryPrice='0'`). It must parse to `Decimal("0")`, must NOT raise, and must NOT be filtered out of the emitted list — dropping it leaves a consumer holding the old size.
- **`orderbook.1` is snapshot-only with both sides always present** (measured: 3,683/3,683 frames `type: "snapshot"`, both sides). A `type: "delta"` frame, or a frame missing `b` or `a`, is **dropped** — never merged.
- **Frames are partial**: one changed symbol per frame. Overwrite per symbol; never replace the whole mapping.
- **`positionIdx` is not read.** Hedge mode is out of scope; mirror the REST path's one-way assumption.
- **`create_equity` keeps reading the wallet balance from REST on every call.** No streamed balance substitutes for it under any condition. No `EventType.BalanceUpdate`, and no balance value crosses from the stream into position or equity state.
- **Ages use `time.monotonic()`**, never wall clock, never an exchange timestamp.
- **Kucoin and EdgeX must keep working unchanged**, taking the existing early return in `_start_price_feed` and logging the same "using REST prices" line.
- **cybotrade changes are pure Python** (`bybit/ws.py`) — no Rust, so no `maturin` rebuild. A compiled `.so` must still be present in a worktree to import `cybotrade`. The repo's own checked-in `.so` at `py-cybotrade/cybotrade/cybotrade.cpython-312-darwin.so` is from Jun 17 and **too old** (lacks `persist_conn_with`); copy a current one from `~/bq/execution/cybotrade/.claude/worktrees/fix-rate-limit-ip-scope/py-cybotrade/cybotrade/cybotrade.cpython-312-darwin.so`.
- **cybotrade tests:** from `~/bq/execution/cybotrade`, `PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/<file> -q -p no:cacheprovider`. Offline files only: `test_exceptions.py`, `test_book_ticker_model.py`, `test_binance_public_ws.py`, `test_binance_private_ws_listen_key.py`, `test_binance_account_update.py`. **Baseline 54 passed.** Every other file in that directory needs live credentials and errors on collection — pre-existing.
- **adrs tests:** from inside the worktree, `~/bq/research/adrs/.venv/bin/pytest tests -q -p no:cacheprovider --no-cov --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py`. Do NOT exclude `tests/integration` — it runs offline and is the strongest regression net. **Baseline 390 passed, 1 xfailed.** `tests/integration/test_placement_scenarios.py` holds a `strict=True` xfail that must stay XFAIL; if it ever XPASSes, stop and report.
- `ruff check` and `ruff format` must pass on adrs (`ruff` at `/Users/marcus/.local/bin/ruff`); pre-commit enforces both, and the `uv-lock` hook rewrites `uv.lock` on any dependency change — stage it or the commit is rejected.
- Conventional Commits. cybotrade's `cliff.toml` needs `!` or a `BREAKING CHANGE:` footer for anything breaking; everything here is additive.

## File Structure

**cybotrade** (`py-cybotrade/`)

| file | responsibility |
| --- | --- |
| `cybotrade/bybit/ws.py` | add `parse_position_topic()` + a `case "position"` to the existing `match msg["topic"]`; add the new `BybitPublicWS` class |
| `tests/test_bybit_position_topic.py` | **new.** payload-fixture parsing of the captured frames, no network |
| `tests/test_bybit_public_ws.py` | **new.** snapshot parsing, delta/missing-side rejection, subscription shape |

**adrs**

| file | responsibility |
| --- | --- |
| `adrs/oms/config.py` | subscribe `"position"` for Bybit; add `to_public_exchange_event()` returning the public adapter or `None` |
| `adrs/oms/oms.py` | `_start_price_feed` uses the selector instead of hard-coding `BinancePublicWS` |
| `tests/test_bybit_parity.py` | **new.** topic list, adapter selection per exchange, Kucoin/EdgeX still get no feed |

No new class in adrs, and no change to `PriceFeed`, `PositionManager`, or `on_exchange_event` — the events already exist and the consumers are already source-agnostic.

## Task Sequencing

Task 1 (position parsing) and Task 2 (public adapter) are independent cybotrade changes. Task 3 wires both in adrs and depends on Task 1's function name and Task 2's class name only. Task 4 releases and validates.

---

## Task 1: cybotrade — parse Bybit's `position` topic

**Files:**
- Modify: `py-cybotrade/cybotrade/bybit/ws.py` (the `match msg["topic"]` block, add a case beside `case "order"` at ~line 143; add a method on `BybitPrivateWS`)
- Test: `py-cybotrade/tests/test_bybit_position_topic.py` (create)

**Interfaces:**
- Consumes: `cybotrade.models.Position` (exists: `symbol`, `quantity`, `entry_price`, `updated_time`, `orig`), `EventType.PositionUpdate` (exists, value `"position_update"`).
- Produces: `BybitPrivateWS.parse_position_topic(msg: dict) -> list[Position]`, and an `Event(event_type=EventType.PositionUpdate, orig=msg, data=list[Position])` emitted from the `position` topic case.

- [ ] **Step 1: Write the failing test**

Create `py-cybotrade/tests/test_bybit_position_topic.py`:

```python
"""
Bybit `position` topic -> list[Position].

Payloads captured from a Bybit demo account by opening and closing 0.001 BTCUSDT
while subscribed to `position`. `size` is absolute and unsigned, so the sign has
to come from `side` -- and on close `side` is an EMPTY STRING, which is the shape
a parser is most likely to get wrong. A close frame that raises or gets filtered
out would leave a consumer holding the old size.
"""

from decimal import Decimal

import pytest

from cybotrade.bybit import BybitPrivateWS
from cybotrade.io.event import EventType

OPEN_FRAME = {
    "id": "577525272_position_1786638991175",
    "topic": "position",
    "creationTime": 1786638991175,
    "data": [
        {
            "positionIdx": 0,
            "tradeMode": 0,
            "riskId": 1,
            "symbol": "BTCUSDT",
            "side": "Buy",
            "size": "0.001",
            "entryPrice": "63354.1",
            "leverage": "10",
            "positionValue": "63.36013",
            "unrealisedPnl": "0.00603",
            "positionStatus": "Normal",
            "updatedTime": "1786638991170",
        }
    ],
}

CLOSE_FRAME = {
    "id": "577525272_position_1786638997519",
    "topic": "position",
    "creationTime": 1786638997519,
    "data": [
        {
            "positionIdx": 0,
            "tradeMode": 0,
            "riskId": 1,
            "symbol": "BTCUSDT",
            "side": "",
            "size": "0",
            "entryPrice": "0",
            "leverage": "10",
            "positionValue": "0",
            "unrealisedPnl": "0",
            "positionStatus": "Normal",
            "updatedTime": "1786638997515",
        }
    ],
}


def _ws() -> BybitPrivateWS:
    return BybitPrivateWS(
        api_key="k", api_secret="s", topics=["order", "position"], testnet=True
    )


def test_open_frame_is_a_signed_absolute_quantity():
    (pos,) = _ws().parse_position_topic(OPEN_FRAME)
    assert str(pos.symbol) == "BTCUSDT"
    assert pos.quantity == Decimal("0.001")
    assert pos.entry_price == Decimal("63354.1")


def test_a_short_is_negative():
    frame = {
        "topic": "position",
        "creationTime": 1786638991175,
        "data": [
            {"symbol": "ETHUSDT", "side": "Sell", "size": "2.5", "entryPrice": "3100.5"}
        ],
    }
    (pos,) = _ws().parse_position_topic(frame)
    assert pos.quantity == Decimal("-2.5")


def test_close_frame_has_empty_side_and_is_still_emitted():
    """
    The measured close shape: size='0', side=''. It must parse to zero and be
    RETURNED -- filtering it out would leave a consumer believing it still holds
    the old size, which is the exact bug absolute overwrite exists to prevent.
    """
    positions = _ws().parse_position_topic(CLOSE_FRAME)
    assert len(positions) == 1
    assert positions[0].quantity == Decimal("0")
    assert str(positions[0].symbol) == "BTCUSDT"


def test_every_entry_in_a_multi_symbol_frame_is_returned_in_order():
    frame = {
        "topic": "position",
        "creationTime": 1786638991175,
        "data": [
            {"symbol": "BTCUSDT", "side": "Buy", "size": "0.001", "entryPrice": "63354.1"},
            {"symbol": "ETHUSDT", "side": "Sell", "size": "1.0", "entryPrice": "3100.0"},
        ],
    }
    positions = _ws().parse_position_topic(frame)
    assert [str(p.symbol) for p in positions] == ["BTCUSDT", "ETHUSDT"]
    assert [p.quantity for p in positions] == [Decimal("0.001"), Decimal("-1.0")]


@pytest.mark.parametrize(
    "frame",
    [
        {"topic": "position"},
        {"topic": "position", "data": None},
        {"topic": "position", "data": []},
        {"topic": "position", "data": [{}]},
        {"topic": "position", "data": [{"symbol": "BTCUSDT", "size": "abc", "side": "Buy"}]},
        {"topic": "position", "data": [{"side": "Buy", "size": "1"}]},
    ],
)
def test_unusable_frames_return_empty_without_raising(frame):
    """This socket also carries order fills, so one bad frame must not kill it."""
    assert _ws().parse_position_topic(frame) == []


def test_a_good_entry_survives_a_bad_neighbour():
    frame = {
        "topic": "position",
        "creationTime": 1786638991175,
        "data": [
            {"symbol": "BTCUSDT", "size": "not-a-number", "side": "Buy"},
            {"symbol": "ETHUSDT", "side": "Buy", "size": "1.0", "entryPrice": "3100.0"},
        ],
    }
    positions = _ws().parse_position_topic(frame)
    assert [str(p.symbol) for p in positions] == ["ETHUSDT"]


def test_no_balance_plumbing_exists():
    """Equity stays on REST; an event nothing reads is dead public API."""
    assert not hasattr(EventType, "BalanceUpdate")
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd ~/bq/execution/cybotrade
PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/test_bybit_position_topic.py -q -p no:cacheprovider
```
Expected: FAIL — `AttributeError: 'BybitPrivateWS' object has no attribute 'parse_position_topic'`.

- [ ] **Step 3: Add the parser**

In `py-cybotrade/cybotrade/bybit/ws.py`, add this method to `BybitPrivateWS`. Match the file's existing style; `Position`, `Symbol`, `Decimal`, `datetime`, `timezone` and `logging` may need adding to the imports at the top.

```python
    def parse_position_topic(self, msg: dict) -> list["Position"]:
        """
        Positions from a `position` topic frame.

        `size` is absolute and unsigned, so the sign comes from `side`: "Buy" is
        long, "Sell" is short. Consumers overwrite with these rather than
        accumulating, which is the point -- incremental fill accounting drifts
        and an absolute value cannot.

        A closed position arrives as size "0" with `side` an EMPTY STRING (not
        "None", not absent). It is returned, not filtered: dropping it would
        leave a consumer believing it still holds the old size.

        Returns [] for a frame with no usable entries rather than raising. This
        socket also carries order fills, so one bad frame must not kill it.
        """
        out: list[Position] = []
        entries = msg.get("data") or []
        event_ms = msg.get("creationTime")
        updated = (
            datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc)
            if event_ms is not None
            else datetime.now(timezone.utc)
        )
        for entry in entries:
            try:
                size = Decimal(entry["size"])
                # "" on a closed position, in which case size is 0 and the sign
                # is irrelevant; only an explicit "Sell" flips it.
                quantity = -size if entry.get("side") == "Sell" else size
                out.append(
                    Position(
                        symbol=Symbol(entry["symbol"]),
                        quantity=quantity,
                        entry_price=Decimal(entry.get("entryPrice", "0")),
                        updated_time=updated,
                        orig=entry,
                    )
                )
            except (KeyError, TypeError, ValueError, ArithmeticError) as e:
                logging.warning(f"Bybit position topic: unparseable entry: {e}")
                continue
        return out
```

- [ ] **Step 4: Run the tests and watch them pass**

```bash
cd ~/bq/execution/cybotrade
PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/test_bybit_position_topic.py -q -p no:cacheprovider
```
Expected: PASS, 8 tests.

- [ ] **Step 5: Emit the event from the topic dispatch**

In the same file's `match msg["topic"]` block, add a case **before** the final `case _:` (which is at ~line 194):

```python
                        case "position":
                            positions = self.parse_position_topic(msg)
                            if positions:
                                await self.on_event(
                                    Event(
                                        event_type=EventType.PositionUpdate,
                                        orig=msg,
                                        data=positions,
                                    )
                                )
```

- [ ] **Step 6: Run the whole offline suite**

```bash
cd ~/bq/execution/cybotrade
PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest \
  <worktree>/py-cybotrade/tests/test_exceptions.py \
  <worktree>/py-cybotrade/tests/test_book_ticker_model.py \
  <worktree>/py-cybotrade/tests/test_binance_public_ws.py \
  <worktree>/py-cybotrade/tests/test_binance_private_ws_listen_key.py \
  <worktree>/py-cybotrade/tests/test_binance_account_update.py \
  <worktree>/py-cybotrade/tests/test_bybit_position_topic.py \
  -q -p no:cacheprovider
```
Expected: 62 passed (54 baseline + 8 new).

- [ ] **Step 7: Commit**

```bash
git add py-cybotrade/cybotrade/bybit/ws.py py-cybotrade/tests/test_bybit_position_topic.py
git commit -m "feat(bybit): emit positions from the position topic"
```

---

## Task 2: cybotrade — `BybitPublicWS` over `orderbook.1`

**Files:**
- Modify: `py-cybotrade/cybotrade/bybit/ws.py` (add a second class after `BybitPrivateWS`)
- Test: `py-cybotrade/tests/test_bybit_public_ws.py` (create)

**Interfaces:**
- Consumes: `cybotrade.models.BookTicker` (exists: `symbol`, `bid`, `ask`, `bid_qty`, `ask_qty`, `update_id`, `event_time`), `EventType.BookTicker` (exists).
- Produces: `BybitPublicWS(symbols: list[str], testnet: bool = False, demo: bool = False, heartbeat_symbol: str = "BTCUSDT")` with attributes `symbols`, `heartbeat_symbol`, `topics: list[str]`, `url: str`, and methods `parse_message(payload: str) -> BookTicker | None`, `create_subscription_message() -> Message`, `start()`.

- [ ] **Step 1: Write the failing test**

Create `py-cybotrade/tests/test_bybit_public_ws.py`:

```python
"""
Bybit public top-of-book adapter.

Frames captured from `orderbook.1.BTCUSDT` on the production shard. Measured over
3,683 consecutive frames: 100% arrive as type "snapshot" with BOTH sides present,
so there is deliberately no delta merging and no per-symbol state here. A "delta"
frame means that assumption changed, and dropping it fails safe -- the next
snapshot lands within ~20ms.
"""

import json
from decimal import Decimal

from cybotrade.bybit import BybitPublicWS

SNAPSHOT = json.dumps(
    {
        "topic": "orderbook.1.BTCUSDT",
        "ts": 1786639170499,
        "type": "snapshot",
        "data": {
            "s": "BTCUSDT",
            "b": [["63040.8", "3.323"]],
            "a": [["63040.9", "4.686"]],
            "u": 31071061,
            "seq": 763537012912,
        },
        "cts": 1786639170498,
    }
)


def test_a_snapshot_becomes_a_book_ticker():
    bt = BybitPublicWS(symbols=["BTCUSDT"]).parse_message(SNAPSHOT)
    assert bt is not None
    assert str(bt.symbol) == "BTCUSDT"
    assert bt.bid == Decimal("63040.8")
    assert bt.ask == Decimal("63040.9")
    assert bt.bid_qty == Decimal("3.323")
    assert bt.ask_qty == Decimal("4.686")
    assert bt.update_id == 31071061


def test_a_delta_frame_is_dropped():
    """
    Measured behaviour is snapshot-only. A delta would need book state to merge
    correctly, and a half-merged top-of-book could price an order wrongly, so it
    is refused rather than guessed at.
    """
    frame = json.loads(SNAPSHOT)
    frame["type"] = "delta"
    assert BybitPublicWS(symbols=["BTCUSDT"]).parse_message(json.dumps(frame)) is None


def test_a_frame_missing_a_side_is_dropped():
    for side in ("b", "a"):
        frame = json.loads(SNAPSHOT)
        frame["data"][side] = []
        assert (
            BybitPublicWS(symbols=["BTCUSDT"]).parse_message(json.dumps(frame)) is None
        )


def test_junk_is_dropped_rather_than_raising():
    ws = BybitPublicWS(symbols=["BTCUSDT"])
    for payload in [
        "",
        "not json",
        "[]",
        json.dumps({"op": "subscribe", "success": True}),
        json.dumps({"topic": "tickers.BTCUSDT", "data": {}}),
        json.dumps({"topic": "orderbook.1.BTCUSDT", "type": "snapshot"}),
        json.dumps({"topic": "orderbook.1.BTCUSDT", "type": "snapshot", "data": {}}),
    ]:
        assert ws.parse_message(payload) is None


def test_the_heartbeat_symbol_is_always_subscribed():
    """
    orderbook.1 only pushes on change, so a portfolio of illiquid symbols would
    have nothing proving the socket is alive. BTCUSDT's book never stops moving
    (measured 21/s with a 590ms worst gap from the production shard).
    """
    ws = BybitPublicWS(symbols=["SOLUSDT"])
    assert "orderbook.1.BTCUSDT" in ws.topics
    assert "orderbook.1.SOLUSDT" in ws.topics


def test_the_heartbeat_symbol_is_not_subscribed_twice():
    ws = BybitPublicWS(symbols=["BTCUSDT", "ETHUSDT"])
    assert ws.topics.count("orderbook.1.BTCUSDT") == 1


def test_subscription_message_shape():
    ws = BybitPublicWS(symbols=["BTCUSDT"])
    body = json.loads(ws.create_subscription_message().payload)
    assert body["op"] == "subscribe"
    assert body["args"] == ws.topics


def test_endpoint_matches_the_credential_environment():
    assert "stream.bybit.com" in BybitPublicWS(symbols=["BTCUSDT"]).url
    assert "stream-testnet" in BybitPublicWS(symbols=["BTCUSDT"], testnet=True).url
    # Demo trading has no separate public feed; public data comes from mainnet.
    assert "stream.bybit.com" in BybitPublicWS(symbols=["BTCUSDT"], demo=True).url
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd ~/bq/execution/cybotrade
PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/test_bybit_public_ws.py -q -p no:cacheprovider
```
Expected: FAIL — `ImportError: cannot import name 'BybitPublicWS'`.

- [ ] **Step 3: Add the class**

Append to `py-cybotrade/cybotrade/bybit/ws.py`:

```python
class BybitPublicWS(ExchangeEvent):
    """
    Public top-of-book stream. Turns `orderbook.1.<SYMBOL>` frames into
    BookTicker events and does nothing else: no cache, no staleness policy.

    Unlike BybitPrivateWS this needs no auth, so there is no login step and no
    credentials to hold.

    Measured on the production shard over 3,683 consecutive frames: every frame
    arrives as type "snapshot" with both sides populated, so depth 1 is always a
    complete top-of-book and no delta merging is needed. A "delta" frame is
    dropped rather than merged -- see parse_message.

    A heartbeat symbol is always subscribed alongside the traded ones. Consumers
    use "any message on this connection" as proof the feed is alive, and
    orderbook.1 only pushes on change, so a portfolio of purely illiquid symbols
    would otherwise have nothing proving the socket still works. BTCUSDT's book
    never stops moving (measured 21/s, 590ms worst gap from the shard).
    """

    def __init__(
        self,
        symbols: list[str],
        testnet: bool = False,
        demo: bool = False,
        heartbeat_symbol: str = "BTCUSDT",
    ):
        wanted = list(symbols)
        if heartbeat_symbol not in wanted:
            wanted.append(heartbeat_symbol)
        self.symbols = symbols
        self.heartbeat_symbol = heartbeat_symbol
        self.topics = [f"orderbook.1.{s}" for s in wanted]
        # Demo trading has no separate public endpoint: public market data comes
        # from mainnet, and only the private socket differs.
        self.url = (
            "wss://stream-testnet.bybit.com/v5/public/linear"
            if testnet
            else "wss://stream.bybit.com/v5/public/linear"
        )
        self.request = Request(url=self.url)
        self.testnet = testnet
        self.demo = demo
        self.set_heartbeat_interval(timedelta(seconds=30))

    def create_subscription_message(self) -> Message:
        return Message.Text(json.dumps({"op": "subscribe", "args": self.topics}))

    async def on_heartbeat(self, sender):
        # Bybit expects a JSON ping on this socket rather than a protocol frame.
        await sender.send(Message.Text(json.dumps({"op": "ping"})))

    async def on_connected(self, sender) -> None:
        # persist_conn re-runs this on every reconnect. Consumers treat the
        # Subscribed event as "feed restarted" and drop their cache, so no quote
        # can survive a gap in delivery.
        await sender.send(self.create_subscription_message())
        await self.on_event(
            Event(
                event_type=EventType.Subscribed,
                orig={"topics": self.topics},
                data=self.topics,
            )
        )

    async def on_login(self) -> None:
        # Public streams need no login step. Required only because
        # ExchangeEvent declares it @abstractmethod.
        pass

    async def on_event(self, event) -> None:
        """User-defined; assign to receive events."""
        pass

    def parse_message(self, payload: str) -> "BookTicker | None":
        """
        BookTicker for a top-of-book snapshot, None for anything else.

        None rather than raising for junk, subscription acks and pongs: a bad
        frame must never take down the feed task.

        A `delta` frame is refused. Merging one would need per-symbol book state,
        and a half-merged top-of-book could price an order wrongly; the next
        snapshot arrives within ~20ms, so dropping loses nothing.
        """
        try:
            body = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(body, dict):
            return None
        topic = body.get("topic")
        if not isinstance(topic, str) or not topic.startswith("orderbook.1."):
            return None
        if body.get("type") != "snapshot":
            return None
        data = body.get("data")
        if not isinstance(data, dict):
            return None
        bids, asks = data.get("b") or [], data.get("a") or []
        if not bids or not asks:
            return None
        try:
            return BookTicker(
                symbol=Symbol(data["s"]),
                bid=Decimal(bids[0][0]),
                ask=Decimal(asks[0][0]),
                bid_qty=Decimal(bids[0][1]),
                ask_qty=Decimal(asks[0][1]),
                update_id=int(data["u"]),
                event_time=datetime.fromtimestamp(
                    int(body["ts"]) / 1000, tz=timezone.utc
                ),
            )
        except (KeyError, IndexError, TypeError, ValueError, ArithmeticError) as e:
            logging.warning(f"Bybit public WS: unparseable orderbook frame: {e}")
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
                logging.warning(f"Bybit public WS encountered an Exception: {e}")
                continue
```

Export it. `py-cybotrade/cybotrade/bybit/__init__.py` has both an import and an `__all__`, so **both** need the new name:

```python
from .ws import BybitPrivateWS, BybitPublicWS

__all__ = ["BybitLinearClient", "BybitError", "BybitPrivateWS", "BybitPublicWS"]
```

Missing the `__all__` entry is the failure mode to watch for: `from cybotrade.bybit import BybitPublicWS` would still work, so the tests pass, but the export is incomplete and `import *` consumers would not see it.

- [ ] **Step 4: Run the tests and watch them pass**

```bash
cd ~/bq/execution/cybotrade
PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/test_bybit_public_ws.py -q -p no:cacheprovider
```
Expected: PASS, 9 tests (the missing-side test is parametrised over two sides inside one test function).

- [ ] **Step 5: Run the whole offline suite**

Same command as Task 1 Step 6, plus `test_bybit_public_ws.py`.
Expected: 71 passed (62 + 9).

- [ ] **Step 6: Commit**

```bash
git add py-cybotrade/cybotrade/bybit/ws.py py-cybotrade/cybotrade/bybit/__init__.py py-cybotrade/tests/test_bybit_public_ws.py
git commit -m "feat(bybit): add a public top-of-book adapter over orderbook.1"
```

---

## Task 3: adrs — subscribe the topic and select the feed by exchange

**Files:**
- Modify: `adrs/oms/config.py` (`to_exchange_event()` at ~line 77; add `to_public_exchange_event()` after it)
- Modify: `adrs/oms/oms.py` (imports at ~line 20, `price_feed_ws` type at ~line 167, `_start_price_feed` at ~line 445-475)
- Test: `tests/test_bybit_parity.py` (create)

**Interfaces:**
- Consumes: `BybitPrivateWS.parse_position_topic` (Task 1, reached via the `position` topic subscription), `BybitPublicWS(symbols=..., testnet=..., demo=...)` with a `.topics` attribute (Task 2).
- Produces: `Credentials.to_public_exchange_event() -> ExchangeEvent | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bybit_parity.py`:

```python
"""
Bybit parity wiring.

Two things adrs owns that the cybotrade adapters cannot do for themselves:
subscribing the `position` topic (Bybit only sends topics you ask for, unlike
Binance's listenKey which delivers everything), and choosing which public feed
adapter to construct.
"""

import pytest
from cybotrade.binance import BinancePublicWS
from cybotrade.bybit import BybitPrivateWS, BybitPublicWS

from adrs.oms.config import Credentials, Exchange


def _creds(exchange: Exchange, **kw) -> Credentials:
    return Credentials(
        exchange=exchange,
        api_key="k",
        api_secret="s",
        api_passphrase="p",
        testnet=kw.get("testnet", False),
        demo=kw.get("demo", False),
    )


def test_bybit_subscribes_the_position_topic():
    """
    Without this the cybotrade position parser is dead code: Bybit only pushes
    topics that were explicitly subscribed.
    """
    ws = _creds(Exchange.BYBIT_LINEAR).to_exchange_event()
    assert isinstance(ws, BybitPrivateWS)
    assert "position" in ws.topics
    assert "order" in ws.topics


def test_bybit_gets_the_bybit_public_feed():
    feed = _creds(Exchange.BYBIT_LINEAR).to_public_exchange_event(symbols=["BTCUSDT"])
    assert isinstance(feed, BybitPublicWS)
    assert feed.topics == ["orderbook.1.BTCUSDT"]


def test_binance_still_gets_the_binance_public_feed():
    feed = _creds(Exchange.BINANCE_LINEAR).to_public_exchange_event(symbols=["BTCUSDT"])
    assert isinstance(feed, BinancePublicWS)


@pytest.mark.parametrize(
    "exchange", [Exchange.KUCOIN_LINEAR, Exchange.EDGEX]
)
def test_exchanges_without_an_adapter_get_none_rather_than_an_error(exchange):
    """
    Kucoin and EdgeX have no public adapter and must keep running on REST
    prices, exactly as they did before the price feed existed. Raising here
    would break two exchanges that have nothing to do with this change.
    """
    assert _creds(exchange).to_public_exchange_event(symbols=["BTCUSDT"]) is None


def test_the_public_feed_follows_the_credentials_environment():
    live = _creds(Exchange.BYBIT_LINEAR).to_public_exchange_event(symbols=["BTCUSDT"])
    test = _creds(Exchange.BYBIT_LINEAR, testnet=True).to_public_exchange_event(
        symbols=["BTCUSDT"]
    )
    assert "stream.bybit.com" in live.url
    assert "stream-testnet" in test.url
```

`Exchange.KUCOIN_LINEAR` and `Exchange.EDGEX` are the real member names (verified in `adrs/oms/config.py:59,69`). `Credentials` is a pydantic `BaseModel` with exactly the fields the helper passes.

- [ ] **Step 2: Run it and watch it fail**

```bash
cd <adrs worktree>
~/bq/research/adrs/.venv/bin/pytest tests/test_bybit_parity.py -q -p no:cacheprovider --no-cov
```
Expected: FAIL — `AttributeError: 'Credentials' object has no attribute 'to_public_exchange_event'`, and the topic test fails with `assert 'position' in ['order']`.

- [ ] **Step 3: Subscribe the position topic**

In `adrs/oms/config.py`, in `to_exchange_event()`, change the Bybit branch:

```python
            case Exchange.BYBIT_LINEAR:
                return BybitPrivateWS(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                    demo=self.demo,
                    # Bybit only pushes topics that were subscribed, unlike
                    # Binance's listenKey stream which carries every event type.
                    topics=["order", "position"],
                )
```

- [ ] **Step 4: Add the public-feed selector**

Directly after `to_exchange_event()` in the same file. Add `BybitPublicWS` and `BinancePublicWS` to the imports at the top:

```python
    def to_public_exchange_event(self, symbols: list[str]) -> ExchangeEvent | None:
        """
        The public top-of-book adapter for this exchange, or None where there is
        no adapter.

        None is a supported configuration, not an error: Kucoin and EdgeX have no
        public feed and read prices from REST exactly as every exchange did
        before the feed existed. Raising here would break them.

        testnet must match the credentials — the feed and the REST fallback have
        to price against the same book.
        """
        match self.exchange:
            case Exchange.BYBIT_LINEAR:
                return BybitPublicWS(
                    symbols=symbols, testnet=self.testnet, demo=self.demo
                )
            case Exchange.BINANCE_LINEAR:
                return BinancePublicWS(symbols=symbols, testnet=self.testnet)
            case _:
                return None
```

- [ ] **Step 5: Run the new tests and watch them pass**

```bash
cd <adrs worktree>
~/bq/research/adrs/.venv/bin/pytest tests/test_bybit_parity.py -q -p no:cacheprovider --no-cov
```
Expected: PASS, 6 tests.

- [ ] **Step 6: Use the selector in the OMS**

In `adrs/oms/oms.py`: remove the now-unused `from cybotrade.binance import BinancePublicWS` at ~line 20 (the concrete classes are constructed in `config.py` now), and import the base type instead — note the path is `cybotrade.io`, **not** `cybotrade.io.event`, matching how `config.py:12` already imports it:

```python
from cybotrade.io import ExchangeEvent
```

Then change the field declaration at ~line 167 to `self.price_feed_ws: ExchangeEvent | None = None`.

Then replace the exchange guard and construction in `_start_price_feed` (~lines 459-471) with:

```python
        symbols = list(self.config.config.base_asset_to_symbol_table.values())
        self.price_feed_ws = self.config.config.credentials.to_public_exchange_event(
            symbols=symbols
        )
        if self.price_feed_ws is None:
            logger.info(
                "[PRICE_FEED] No public feed for "
                f"{self.config.config.credentials.exchange}, using REST prices"
            )
            return
        self.price_feed_ws.on_event = self.on_price_feed_event
        self.price_feed_task = asyncio.create_task(self.price_feed_ws.start())
        self._supervise_task(self.price_feed_task, "PRICE_FEED")
        logger.info(f"[PRICE_FEED] Started for {self.price_feed_ws.topics if hasattr(self.price_feed_ws, 'topics') else self.price_feed_ws.streams}")
```

Keep the existing "already running" guard above this unchanged. Note the final log line: `BinancePublicWS` exposes `.streams` and `BybitPublicWS` exposes `.topics`, so it reads whichever exists.

- [ ] **Step 7: Run the full adrs suite**

```bash
cd <adrs worktree>
~/bq/research/adrs/.venv/bin/pytest tests -q -p no:cacheprovider --no-cov \
  --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py
```
Expected: 396 passed, 1 xfailed (390 baseline + 6 new). The `strict=True` xfail must still be XFAIL.

- [ ] **Step 8: Lint and commit**

```bash
cd <adrs worktree>
/Users/marcus/.local/bin/ruff check adrs tests
/Users/marcus/.local/bin/ruff format adrs tests
git add adrs/oms/config.py adrs/oms/oms.py tests/test_bybit_parity.py
git commit -m "feat(oms): subscribe Bybit positions and select the price feed by exchange"
```

---

## Task 4: release and validate on one Bybit tenant

**Files:** none in this task beyond a version bump.

**Interfaces:** none.

- [ ] **Step 1: Release cybotrade**

The adrs change imports `BybitPublicWS`, which only exists in an unreleased cybotrade, so cybotrade ships first. Merge the cybotrade branch, bump `py-cybotrade/Cargo.toml` (the only version source — `pyproject.toml` uses `dynamic = ["version"]`), then use the `/cybotrade-release` skill, which covers the mandatory `dist/` clean, the three build targets and the PyPI upload.

Additive change, so a minor bump: 2.2.0 → 2.3.0.

**adrs resolves cybotrade from public PyPI**, not the GitLab registry — `uv.lock` records `registry = "https://pypi.org/simple"` and `pyproject.toml` declares no custom index. Publishing only to GitLab will not satisfy adrs or CI.

- [ ] **Step 2: Bump the adrs pin**

Only after 2.3.0 is resolvable on PyPI — `uv lock` cannot resolve an unpublished version, and the `uv-lock` pre-commit hook rewrites the lockfile on any dependency change, so bumping earlier makes the branch uncommittable.

```bash
cd <adrs worktree>
# set cybotrade>=2.3.0 in pyproject.toml, then:
uv lock
git add pyproject.toml uv.lock
git commit -m "build: require cybotrade>=2.3.0 for the Bybit position and price streams"
```

- [ ] **Step 3: Release adrs**

Minor bump (new feature): 1.9.1 → 1.10.0. Clean `dist/` first — it accumulates and a stale wheel from a previous release will make `twine upload dist/*` abort the batch.

```bash
cd ~/bq/research/adrs
find dist -name '*.whl' -delete -o -name '*.tar.gz' -delete
uv build
uvx twine check dist/adrs-1.10.0*
uvx twine upload dist/adrs-1.10.0-py3-none-any.whl dist/adrs-1.10.0.tar.gz
```

- [ ] **Step 4: Validate on one Bybit tenant**

Deploy to a single Bybit tenant (`pc-ew7if…` or `pc-wgocc…`) and check, in this order:

1. **The stream is feeding.** `position` frames previously fell through the topic match's `case _: pass`. Confirm that stops, and that `[ON_EVENT] 'EventType.PositionUpdate'` appears at debug level or that positions move without a REST read.
2. **`UID_POSITION` stays near its ceiling** between anchor reads, where it previously dropped on every placement tick.
3. **Prices come from the feed.** `[PRICE_FEED] Started for ['orderbook.1.…']` at startup, then no per-tick price reads.
4. **Force a reconnect** (delete the pod) and confirm the anchor invalidation fires exactly one REST read, and that `PriceFeed.clear()` drops quotes rather than serving stale ones.
5. **Kucoin and EdgeX are untouched.** If a tenant of either exists, confirm it still logs "No public feed for …, using REST prices" and trades normally. This is the regression this change is most likely to cause.

- [ ] **Step 5: Confirm the parity work is delivered, then raise the cancel-burst problem**

Once validation passes, the parity work is done. **Report back that the Bybit rate-limit complaint is still outstanding** — it is a burst of cancels exhausting the cancel pool, which nothing in this plan addresses. See the spec's *Purpose* for the evidence.

---

## Notes for the implementer

- The two cybotrade tasks are independent; either can be done first.
- Do not add hedge-mode handling. `positionIdx` is deliberately unread.
- Do not touch `create_equity`. The wallet balance is read from REST on every call, and that is a correctness boundary, not an optimisation gap.
- Do not add a `PriceFeed` change. It is exchange-agnostic already, and it already rejects crossed or zero books, which is why the adapter can afford to simply drop doubtful frames.
- If the `strict=True` xfail in `tests/integration/test_placement_scenarios.py` ever XPASSes, stop and report rather than editing the marker. It pins an unrelated defect and a strict xfail turning green fails the whole run.

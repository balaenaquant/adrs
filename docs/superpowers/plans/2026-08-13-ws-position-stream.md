# Websocket Position Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maintain OMS positions from Binance's `ACCOUNT_UPDATE` user-data stream instead of polling `GET_POSITION` on every placement tick, and delete the drift-prone incremental fill accounting while doing it.

**Architecture:** `ACCOUNT_UPDATE` carries *absolute* position amounts, so the stream overwrites `PositionManager.exchange` per symbol with no arithmetic. A REST `GET_POSITION` every 60s (already scheduled by `on_aegis_update`) is both the correction anchor and the liveness signal — `ACCOUNT_UPDATE` is event-driven and the user-data socket has no busy stream to use as a heartbeat, so "live" means *the last successful REST reconcile was within 90s*. `delta_calculation` stops forcing its own read, which is the entire weight saving.

**Tech Stack:** Python (adrs 3.14, cybotrade 3.12), `cybotrade` user-data websocket (`BinancePrivateWS`, already running), `pytest`, `ruff`, `Decimal` for all quantities.

Spec: `docs/superpowers/specs/2026-08-13-ws-position-stream-design.md` (PR #44). Read *Decisions*, *Balances* and *The gating experiment* before starting.

## Global Constraints

- **Task 0 is a hard gate.** It verifies that `pa` is the *absolute* position amount, not a delta. If it is a delta, absolute overwrite would replace a true position with the size of the last change — stop and re-open the design rather than adapting in place.
- **Quantities are always `Decimal`, never float.** `pa` is signed: negative means short, mirroring how `get_positions` maps `positionAmt`.
- **Ages on `time.monotonic()`**, never wall clock, never an exchange timestamp.
- `POSITION_ANCHOR_MAX_AGE_SEC = 90.0`.
- **`create_equity` keeps reading the wallet balance from REST on every call.** The streamed balance is never a substitute under any freshness condition. Do not extend the anchor logic to cover it. This is a correctness boundary, not a deferred optimisation — see the spec's *Balances*.
- **No `EventType.BalanceUpdate`, and no balance value crosses into adrs.** Task 0 records what `a.B` contains; that is all.
- **Hedge mode is out of scope.** Today's REST path ignores `positionSide` and lets the last position per symbol win; the stream mirrors that exactly.
- **cybotrade changes are pure Python** (`io/event.py`, `binance/ws.py`) — no Rust, so no `maturin` rebuild. The compiled `.so` must still be present in a worktree to import `cybotrade`: `cp ~/bq/execution/cybotrade/py-cybotrade/cybotrade/cybotrade.cpython-312-darwin.so <worktree>/py-cybotrade/cybotrade/` (gitignored).
- **adrs tests:** from inside the worktree, `~/bq/research/adrs/.venv/bin/pytest <target> -q -p no:cacheprovider --no-cov`. Full suite adds `--ignore=tests/integration --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py`; those two need an untracked `credentials.json` and a local clickhouse and fail on any fresh worktree. **Baseline is 366 passed + 1 xfailed.**
- **cybotrade tests:** from `~/bq/execution/cybotrade`, `PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest <worktree>/py-cybotrade/tests/<file> -q -p no:cacheprovider`. Offline files only: `test_exceptions.py`, `test_book_ticker_model.py`, `test_binance_public_ws.py`, `test_binance_private_ws_listen_key.py`. **Baseline 40 passed.** The rest need live credentials and error out — pre-existing.
- `ruff check` and `ruff format` must pass on adrs (`ruff` at `/Users/marcus/.local/bin/ruff`); pre-commit enforces both, and the `uv-lock` hook rewrites `uv.lock` on any version bump — stage it or the commit is rejected.
- Conventional Commits. cybotrade's `cliff.toml` needs `!` or a `BREAKING CHANGE:` footer for anything breaking.

## File Structure

**cybotrade** (`py-cybotrade/`)

| file | responsibility |
| --- | --- |
| `cybotrade/io/event.py` | add `EventType.PositionUpdate` |
| `cybotrade/binance/ws.py` | add an `ACCOUNT_UPDATE` case to the existing `match msg["e"]` — wire format to `list[Position]`, no state |
| `tests/test_binance_account_update.py` | **new.** payload-fixture parsing, no network |

**adrs**

| file | responsibility |
| --- | --- |
| `scripts/verify_binance_account_update.py` | **new.** Task 0's gate: prove `pa` is absolute, record what `a.B` holds |
| `adrs/oms/position.py` | `apply_stream_positions`, `POSITION_ANCHOR_MAX_AGE_SEC`, `delta_calculation` stops forcing a read |
| `adrs/oms/ops/order_placement_manager.py` | `on_exchange_event` gains a `PositionUpdate` case; `update_positions` loses its `exchange` line |
| `tests/test_position_stream.py` | **new.** absolute-overwrite, anchor freshness, the retired increment |

No new class, task or cron: `PositionManager` already owns the state and the REST path, `_positions_fresh_within()`/`_exchange_refreshed_at` are already the anchor check, and `on_aegis_update` already polls every 60s.

## Task Sequencing

Task 0 gates everything. Task 2 is pure Python over the existing `Position` model, so it lands before any cybotrade release. Task 3 imports `EventType.PositionUpdate` and therefore waits on that release.

---

## Task 0: Prove `pa` is absolute (HARD GATE)

**Files:**
- Create: `scripts/verify_binance_account_update.py` (adrs)

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded finding appended to the spec. No code depends on this task, but every later task depends on its **answer**.

Binance's docs would not serve the `ACCOUNT_UPDATE` payload across three fetches. The same thing happened with the order-book depth question in the price feed design, where the assumption turned out to be **wrong** and only an experiment caught it. Do not skip this.

- [ ] **Step 1: Write the script**

Create `scripts/verify_binance_account_update.py`:

```python
"""
Settle the gating question in
docs/superpowers/specs/2026-08-13-ws-position-stream-design.md: is
ACCOUNT_UPDATE's `pa` the absolute position amount, or a delta?

The whole design rests on it. Absolute means the stream can overwrite
PositionManager.exchange with no arithmetic, which is what retires the
drift-prone incremental accounting. A delta would mean absolute overwrite
replaces a true position with the size of the last change -- catastrophically
wrong on a live account.

Also records what a.B (balances) actually contains, which the spec needs but
does not consume.

Needs BINANCE_API_KEY / BINANCE_API_SECRET in the environment. Run where a key
exists -- the tenant pod. Read-only: it opens a user-data stream and polls
positions; it places no orders.

Usage:
    python scripts/verify_binance_account_update.py [--seconds 900]

To see frames at all, the account must CHANGE during the window (a fill, funding,
or a manual transfer). A quiet account emits nothing -- that is the event-driven
behaviour the design is built around, not a failure of this script.
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request

import websockets

FAPI = "https://fapi.binance.com"


def _key() -> str:
    k = os.environ.get("BINANCE_API_KEY")
    if not k:
        raise SystemExit("BINANCE_API_KEY is not set")
    return k


def _secret() -> str:
    s = os.environ.get("BINANCE_API_SECRET")
    if not s:
        raise SystemExit("BINANCE_API_SECRET is not set")
    return s


def _signed_get(path: str) -> dict:
    params = {"timestamp": str(int(time.time() * 1000)), "recvWindow": "5000"}
    query = urllib.parse.urlencode(sorted(params.items()))
    sig = hmac.new(_secret().encode(), query.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        f"{FAPI}{path}?{query}&signature={sig}",
        headers={"X-MBX-APIKEY": _key()},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def rest_positions() -> dict[str, str]:
    """symbol -> positionAmt, as REST reports it. The value `pa` must match."""
    rows = _signed_get("/fapi/v3/positionRisk")
    return {r["symbol"]: r["positionAmt"] for r in rows}


def listen_key() -> str:
    req = urllib.request.Request(
        f"{FAPI}/fapi/v1/listenKey",
        method="POST",
        headers={"X-MBX-APIKEY": _key()},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["listenKey"]


async def main(seconds: float) -> None:
    before = rest_positions()
    nonzero = {k: v for k, v in before.items() if v not in ("0", "0.0", "0.000")}
    print(f"REST positions now (non-zero): {nonzero or '(none)'}", flush=True)
    print(f"listening {seconds:.0f}s for ACCOUNT_UPDATE...", flush=True)

    url = f"wss://fstream.binance.com/ws/{listen_key()}"
    frames = 0
    async with websockets.connect(url, ping_interval=20) as ws:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=deadline - time.monotonic()
                )
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            msg = json.loads(raw)
            if msg.get("e") != "ACCOUNT_UPDATE":
                continue
            frames += 1
            a = msg.get("a", {})
            print(f"\n--- ACCOUNT_UPDATE #{frames}  m={a.get('m')} ---", flush=True)
            print(f"  a.P: {json.dumps(a.get('P'))}", flush=True)
            print(f"  a.B: {json.dumps(a.get('B'))}", flush=True)

            # The decisive comparison: does pa equal what REST reports right now?
            after = rest_positions()
            for p in a.get("P", []):
                sym, pa = p.get("s"), p.get("pa")
                rest = after.get(sym)
                verdict = (
                    "ABSOLUTE (pa == REST positionAmt)"
                    if rest is not None and _same(pa, rest)
                    else f"MISMATCH (pa={pa} vs REST={rest}) -> pa may be a DELTA"
                )
                print(f"  {sym}: {verdict}", flush=True)

            print(
                f"  frame lists {len(a.get('P') or [])} position(s); "
                f"account holds {len([v for v in after.values() if not _is_zero(v)])} "
                f"non-zero -> {'PARTIAL' if len(a.get('P') or []) < len([v for v in after.values() if not _is_zero(v)]) else 'possibly COMPLETE'}",
                flush=True,
            )

    if frames == 0:
        print(
            "\nNo ACCOUNT_UPDATE in the window. The account did not change, which is "
            "the event-driven behaviour the design assumes -- rerun over a period "
            "with a fill, or trigger one.",
            flush=True,
        )


def _is_zero(v: str) -> bool:
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return True


def _same(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=900.0)
    asyncio.run(main(ap.parse_args().seconds))
```

- [ ] **Step 2: Lint and commit the script**

```bash
ruff check scripts/verify_binance_account_update.py
ruff format scripts/verify_binance_account_update.py
git add scripts/verify_binance_account_update.py
git commit -m "test: verify whether ACCOUNT_UPDATE's pa is absolute or a delta

Gates the position-stream design. Absolute means the stream can overwrite
positions with no arithmetic, which is what retires the drift-prone incremental
accounting; a delta would mean absolute overwrite replaces a true position with
the size of the last change. Binance's docs would not serve the payload across
three fetches, and the same failure on the order-book depth question turned out
to hide a wrong assumption."
```

- [ ] **Step 3: Run it where credentials exist**

In the tenant pod (this repo's `.envrc` carries Bybit keys only):

```bash
python scripts/verify_binance_account_update.py --seconds 900
```

Record in the spec, under *The gating experiment*:
- whether every `pa` matched REST's `positionAmt` (**absolute**) or not (**delta**)
- whether a frame listed every non-zero position or only the changed ones
- the raw `a.B` array for this account type

- [ ] **Step 4: Decide**

- `pa` **absolute** → proceed to Task 1.
- `pa` **a delta**, or frames inconsistent → **STOP.** Report it and re-open the design. Do not adapt the plan in place; the trust model depends on this.
- **No frames observed** → the answer is not established. Do not proceed on the assumption; rerun over a window containing a fill.

---

## Task 1: cybotrade — `EventType.PositionUpdate` and `ACCOUNT_UPDATE` parsing

**Blocked on:** Task 0 confirming `pa` is absolute.

**Files:**
- Modify: `py-cybotrade/cybotrade/io/event.py` (the `EventType` enum, ~line 8-15)
- Modify: `py-cybotrade/cybotrade/binance/ws.py` (the `match msg["e"]` block, ~line 327)
- Test: `py-cybotrade/tests/test_binance_account_update.py` (create)

**Interfaces:**
- Consumes: `cybotrade.models.Position` (already exists: `symbol`, `quantity`, `entry_price`, `updated_time`, `orig`).
- Produces: `EventType.PositionUpdate` with value `"position_update"`; an `Event` whose `data` is `list[Position]`; and `BinancePrivateWS.parse_account_update(msg: dict) -> list[Position]`.

- [ ] **Step 1: Write the failing test**

Create `py-cybotrade/tests/test_binance_account_update.py`:

```python
"""
ACCOUNT_UPDATE -> list[Position]. Payload captured by
adrs/scripts/verify_binance_account_update.py.

`pa` is the absolute position amount, signed, matching REST's positionAmt --
verified before this was written, because a delta would make absolute overwrite
replace a true position with the size of the last change.
"""

from decimal import Decimal

import pytest

from cybotrade.binance import BinancePrivateWS
from cybotrade.io.event import EventType
from cybotrade.models import Position

FRAME = {
    "e": "ACCOUNT_UPDATE",
    "E": 1786600000000,
    "T": 1786599999999,
    "a": {
        "m": "ORDER",
        "B": [{"a": "USDT", "wb": "10000.00", "cw": "10000.00", "bc": "0"}],
        "P": [
            {
                "s": "BTCUSDT",
                "pa": "0.150",
                "ep": "63500.0",
                "cr": "0",
                "up": "12.5",
                "mt": "cross",
                "iw": "0",
                "ps": "BOTH",
            },
            {
                "s": "ETHUSDT",
                "pa": "-2.000",
                "ep": "3100.0",
                "cr": "0",
                "up": "-4.0",
                "mt": "cross",
                "iw": "0",
                "ps": "BOTH",
            },
        ],
    },
}


def _ws() -> BinancePrivateWS:
    ws = object.__new__(BinancePrivateWS)  # no listenKey, no network
    return ws


def test_position_update_event_type_exists():
    assert EventType.PositionUpdate.value == "position_update"


def test_parses_positions_with_signed_decimal_quantities():
    positions = _ws().parse_account_update(FRAME)
    assert len(positions) == 2
    by_symbol = {str(p.symbol): p for p in positions}
    assert isinstance(by_symbol["BTCUSDT"], Position)
    assert by_symbol["BTCUSDT"].quantity == Decimal("0.150")
    assert by_symbol["BTCUSDT"].entry_price == Decimal("63500.0")
    # Short positions keep their sign, exactly as REST's positionAmt does
    assert by_symbol["ETHUSDT"].quantity == Decimal("-2.000")


def test_a_flat_position_is_reported_as_zero_not_dropped():
    """
    A position closing to zero is the single most important frame to apply: drop
    it and the OMS believes it still holds the old size and sizes orders off it.
    """
    frame = {"e": "ACCOUNT_UPDATE", "a": {"P": [
        {"s": "BTCUSDT", "pa": "0", "ep": "0.0", "ps": "BOTH"}
    ]}}
    positions = _ws().parse_account_update(frame)
    assert len(positions) == 1
    assert positions[0].quantity == Decimal("0")


@pytest.mark.parametrize(
    "frame",
    [
        {"e": "ACCOUNT_UPDATE", "a": {}},                       # no P at all
        {"e": "ACCOUNT_UPDATE", "a": {"P": []}},                # empty P
        {"e": "ACCOUNT_UPDATE"},                                # no a
        {"e": "ACCOUNT_UPDATE", "a": {"P": [{"s": "BTCUSDT"}]}},  # no pa
        {"e": "ACCOUNT_UPDATE", "a": {"P": [{"pa": "1"}]}},       # no symbol
    ],
)
def test_partial_or_malformed_frames_yield_no_positions_and_do_not_raise(frame):
    """
    The user-data socket also carries order fills. A bad frame must never kill it.
    """
    assert _ws().parse_account_update(frame) == []


def test_balances_are_not_parsed_into_anything():
    """
    a.B is deliberately not consumed: with equity pinned to REST there is no
    consumer, and an emitted event no handler reads is dead code. The frame's
    balance array must not produce an event or a model.
    """
    assert not hasattr(EventType, "BalanceUpdate")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cp ~/bq/execution/cybotrade/py-cybotrade/cybotrade/cybotrade.cpython-312-darwin.so \
   <worktree>/py-cybotrade/cybotrade/
cd ~/bq/execution/cybotrade && PYTHONPATH=<worktree>/py-cybotrade \
  .venv/bin/pytest <worktree>/py-cybotrade/tests/test_binance_account_update.py -q
```
Expected: FAIL — `AttributeError: PositionUpdate` / `parse_account_update` missing.

- [ ] **Step 3: Add the event type**

In `cybotrade/io/event.py`, after `OrderUpdate`:

```python
    PositionUpdate = "position_update"
```

- [ ] **Step 4: Add the parser and the match case**

In `cybotrade/binance/ws.py`, add a method on `BinancePrivateWS`:

```python
    def parse_account_update(self, msg: dict) -> list[Position]:
        """
        Positions from an ACCOUNT_UPDATE frame's `a.P` array.

        `pa` is the absolute position amount, signed -- negative is short --
        mirroring how get_positions maps REST's positionAmt. Consumers overwrite
        with it rather than accumulating, which is the point: incremental fill
        accounting drifts, and an absolute value cannot.

        A position that has closed arrives as pa == 0 and must be returned, not
        filtered: dropping it leaves a consumer believing it still holds the old
        size.

        Returns [] for a frame with no usable positions rather than raising. This
        socket also carries order fills, so one bad frame must not kill it.
        """
        out: list[Position] = []
        positions = (msg.get("a") or {}).get("P") or []
        event_ms = msg.get("E")
        updated = (
            datetime.fromtimestamp(int(event_ms) / 1000, tz=timezone.utc)
            if event_ms is not None
            else datetime.now(timezone.utc)
        )
        for p in positions:
            try:
                out.append(
                    Position(
                        symbol=Symbol(p["s"]),
                        quantity=Decimal(p["pa"]),
                        entry_price=Decimal(p.get("ep", "0")),
                        updated_time=updated,
                        orig=p,
                    )
                )
            except (KeyError, TypeError, ValueError, ArithmeticError) as e:
                logging.warning(f"Binance ACCOUNT_UPDATE: unparseable position: {e}")
                continue
        return out
```

Then add a case to the existing `match msg["e"]` block, immediately before its
`case _:` fallthrough:

```python
                        case "ACCOUNT_UPDATE":
                            positions = self.parse_account_update(msg)
                            if positions:
                                await self.on_event(
                                    Event(
                                        event_type=EventType.PositionUpdate,
                                        orig=msg,
                                        data=positions,
                                    )
                                )
```

Add `Position` to the existing `from cybotrade.models import (...)` list. `Decimal`,
`datetime`, `timezone`, `Symbol`, `logging`, `Event` and `EventType` are already
imported in this file — do not re-import them.

Note a frame carrying only balances (`a.B` with no `a.P`) yields no positions and
so emits nothing, which is correct: there is no balance consumer.

- [ ] **Step 5: Run tests to verify they pass**

Run the Step 2 command. Expected: 9 passed. Then confirm the pre-existing suite is
untouched:

```bash
cd ~/bq/execution/cybotrade && PYTHONPATH=<worktree>/py-cybotrade .venv/bin/pytest \
  <worktree>/py-cybotrade/tests/test_exceptions.py \
  <worktree>/py-cybotrade/tests/test_book_ticker_model.py \
  <worktree>/py-cybotrade/tests/test_binance_public_ws.py \
  <worktree>/py-cybotrade/tests/test_binance_private_ws_listen_key.py -q
```
Expected: 40 passed.

- [ ] **Step 6: Commit**

```bash
git add py-cybotrade/cybotrade/io/event.py py-cybotrade/cybotrade/binance/ws.py \
        py-cybotrade/tests/test_binance_account_update.py
git commit -m "feat(binance): emit positions from ACCOUNT_UPDATE

Parses the frame's a.P array into the existing Position model and emits it as
EventType.PositionUpdate. pa is the absolute position amount, signed, mirroring
how get_positions maps REST's positionAmt -- verified against a live stream before
this was written, because a delta would make a consumer's absolute overwrite
replace a true position with the size of the last change.

A closed position (pa == 0) is emitted rather than filtered: dropping it would
leave a consumer holding the old size. Unparseable positions are skipped and a
frame with none emits nothing, because this socket also carries order fills and one
bad frame must not kill it.

a.B is deliberately not parsed into anything. With equity pinned to REST there is
no consumer, and an event type no handler reads is dead code -- and a public name
in a published library is the hardest thing to walk back."
```

---

## Task 2: adrs — `apply_stream_positions` and the anchor constant

**Not blocked on the cybotrade release:** this task takes `list[Position]`, and
`Position` exists in cybotrade 2.1.0 already.

**Files:**
- Modify: `adrs/oms/position.py`
- Test: `tests/test_position_stream.py` (create)

**Interfaces:**
- Consumes: `cybotrade.models.Position`.
- Produces: `PositionManager.apply_stream_positions(positions: list[Position]) -> None`; `adrs.oms.position.POSITION_ANCHOR_MAX_AGE_SEC = 90.0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_position_stream.py`:

```python
"""
Positions sourced from the ACCOUNT_UPDATE stream.

The point of this path is that it is *absolute*. The incremental writer it
replaces drifts by construction -- its own comment records fills over-counting
from the third update onward -- so these tests pin the property that makes the
replacement worthwhile: applying the same frame twice must not double a position.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.models import Position

from adrs.oms.position import POSITION_ANCHOR_MAX_AGE_SEC, PositionManager

BTC = Symbol("BTCUSDT")
ETH = Symbol("ETHUSDT")


def _position(symbol: Symbol, qty: str) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        entry_price=Decimal("63500"),
        updated_time=datetime(2026, 8, 13, tzinfo=timezone.utc),
    )


def _pm() -> PositionManager:
    pm = object.__new__(PositionManager)
    pm.exchange = {}
    pm.pending = {}
    pm.desired = {}
    pm.delta_lock = asyncio.Lock()
    pm._refresh_lock = asyncio.Lock()
    pm._exchange_refreshed_at = None
    pm.rate_limiter = MagicMock()
    pm.config = SimpleNamespace(
        config=SimpleNamespace(
            initial_balance=Decimal("10000"),
            leverage=Decimal("1"),
            base_asset_to_symbol_table={"BTC": "BTCUSDT"},
        ),
        exchange=MagicMock(),
    )
    return pm


def test_applying_a_frame_sets_the_position_absolutely():
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    assert pm.exchange[BTC].quantity == Decimal("0.15")


def test_applying_the_same_frame_twice_does_not_double_the_position():
    """The property the incremental writer could not offer."""
    pm = _pm()
    frame = [_position(BTC, "0.15")]
    pm.apply_stream_positions(frame)
    pm.apply_stream_positions(frame)
    assert pm.exchange[BTC].quantity == Decimal("0.15")


def test_a_later_frame_replaces_rather_than_accumulates():
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    pm.apply_stream_positions([_position(BTC, "0.40")])
    assert pm.exchange[BTC].quantity == Decimal("0.40")


def test_a_symbol_absent_from_the_frame_keeps_its_previous_value():
    """
    Frames are partial: they carry the positions that changed. Replacing the whole
    dict would zero every symbol Binance did not mention.
    """
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15"), _position(ETH, "-2")])
    pm.apply_stream_positions([_position(BTC, "0.20")])
    assert pm.exchange[BTC].quantity == Decimal("0.20")
    assert pm.exchange[ETH].quantity == Decimal("-2")


def test_a_flat_position_is_applied_not_ignored():
    """
    Closing to zero is the frame that matters most: ignore it and the OMS sizes
    orders against a position it no longer holds.
    """
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    pm.apply_stream_positions([_position(BTC, "0")])
    assert pm.exchange[BTC].quantity == Decimal("0")


def test_an_empty_frame_changes_nothing():
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    pm.apply_stream_positions([])
    assert pm.exchange[BTC].quantity == Decimal("0.15")


def test_the_stream_does_not_stamp_the_rest_anchor():
    """
    The anchor is what proves liveness, and only a REST read may set it. If the
    stream stamped it, a dead REST path would look healthy forever and the sizing
    path would trust an unanchored stream indefinitely.
    """
    pm = _pm()
    pm.apply_stream_positions([_position(BTC, "0.15")])
    assert pm._exchange_refreshed_at is None


def test_anchor_max_age_is_bounded():
    # Must exceed the 60s aegis cadence that sets the anchor, or the sizing path
    # would force a REST read on most ticks and the saving would vanish.
    assert 60.0 < POSITION_ANCHOR_MAX_AGE_SEC <= 120.0


def test_delta_calculation_does_not_read_rest_while_the_anchor_is_fresh():
    import time

    pm = _pm()
    pm.config.exchange.get_positions = AsyncMock(return_value=[])
    pm._exchange_refreshed_at = time.monotonic()  # fresh anchor
    pm.desired = {BTC: _position(BTC, "1")}
    pm.exchange = {BTC: _position(BTC, "0")}
    pm.pending = {BTC: _position(BTC, "0")}
    pm.update_pending = lambda snapshot: None

    asyncio.run(pm.delta_calculation(SimpleNamespace(orders={})))
    pm.config.exchange.get_positions.assert_not_awaited()


def test_delta_calculation_forces_a_read_once_the_anchor_is_stale():
    import time

    pm = _pm()
    pm.config.exchange.get_positions = AsyncMock(return_value=[])
    pm._exchange_refreshed_at = time.monotonic() - POSITION_ANCHOR_MAX_AGE_SEC - 1
    pm.desired = {BTC: _position(BTC, "1")}
    pm.exchange = {BTC: _position(BTC, "0")}
    pm.pending = {BTC: _position(BTC, "0")}
    pm.update_pending = lambda snapshot: None

    asyncio.run(pm.delta_calculation(SimpleNamespace(orders={})))
    pm.config.exchange.get_positions.assert_awaited_once()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_position_stream.py -q -p no:cacheprovider --no-cov
```
Expected: FAIL — `ImportError: cannot import name 'POSITION_ANCHOR_MAX_AGE_SEC'`.

- [ ] **Step 3: Add the constant and the method**

In `adrs/oms/position.py`, after `POSITION_REFRESH_TTL_SEC`:

```python
# How stale the REST anchor may be before the order-sizing path stops trusting
# the streamed position and forces a read.
#
# ACCOUNT_UPDATE is event-driven, so silence is ambiguous -- a quiet account and a
# dead socket look identical -- and unlike the price feed there is no busy stream
# on the user-data socket to use as a heartbeat: its ping is every 3 minutes and
# the listenKey keepalive every 30. So the 60s REST reconcile is the liveness
# mechanism, and this is the window it is trusted for. Must exceed that 60s
# cadence, or the sizing path forces a read on most ticks and the saving is lost.
POSITION_ANCHOR_MAX_AGE_SEC = 90.0
```

Then add the method to `PositionManager`:

```python
    def apply_stream_positions(self, positions: list[Position]) -> None:
        """
        Overwrite the exchange position for each symbol in a stream frame.

        Absolute, never accumulated: this replaces the incremental fill
        accounting, which drifted by construction. Applying the same frame twice
        is a no-op, which is the property that makes the replacement worth making.

        Frames are partial -- they carry what changed -- so symbols absent from
        `positions` keep their previous value rather than being zeroed. A position
        that has closed arrives as quantity 0 and is applied, not skipped: skipping
        it would leave the OMS sizing orders against a position it no longer holds.

        Deliberately does NOT stamp the REST anchor. Only a real REST read may do
        that, because the anchor is what proves liveness -- if the stream stamped
        it, a dead REST path would look healthy forever.
        """
        for position in positions:
            self.exchange[position.symbol] = position
```

Add `Position` to the existing `from cybotrade.models import ...` line if it is not
already imported.

- [ ] **Step 4: Stop `delta_calculation` forcing a read**

In `delta_calculation` (`position.py:67`), change:

```python
            await self.update_exchange(max_age_sec=0)
```

to:

```python
            # Trust the streamed position while the REST anchor is fresh. The
            # anchor (set only by a real read, refreshed every 60s by
            # on_aegis_update) is the liveness proof; once it ages out this
            # forces a read exactly as it always did.
            await self.update_exchange(max_age_sec=POSITION_ANCHOR_MAX_AGE_SEC)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_position_stream.py -q -p no:cacheprovider --no-cov
```
Expected: 10 passed.

- [ ] **Step 6: Run the full suite**

```bash
~/bq/research/adrs/.venv/bin/pytest tests/ -q -p no:cacheprovider --ignore=tests/integration \
  --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py --no-cov
```
Expected: 376 passed + 1 xfailed (366 baseline + 10 new). If `tests/test_position_manager.py`
fails on a missing attribute, its `_pm()` helper needs the same fields this task's
helper sets — fix the helper, not the production code.

- [ ] **Step 7: Lint and commit**

```bash
ruff check adrs/ tests/
ruff format adrs/oms/position.py tests/test_position_stream.py
git add adrs/oms/position.py tests/test_position_stream.py
git commit -m "feat(oms): apply positions absolutely from a stream frame

apply_stream_positions overwrites per symbol with no arithmetic, so applying the
same frame twice is a no-op -- the property the incremental fill accounting it
replaces could not offer, since that drifted by construction.

delta_calculation stops forcing max_age_sec=0 and trusts the streamed position
while the REST anchor is fresh. That forced read was 10 of the 20 weight/min spent
on positions and balance, and it ran on every placement tick.

The stream deliberately does not stamp the anchor: only a real REST read may,
because the anchor is what proves liveness. If the stream stamped it, a dead REST
path would look healthy forever and the sizing path would trust an unanchored
stream indefinitely.

Frames are partial, so symbols they omit keep their previous value; a position
closing to zero is applied rather than skipped, since skipping it would leave the
OMS sizing orders against a position it no longer holds."
```

---

## Task 3: adrs — wire the event and retire the incremental writer

**Blocked on:** cybotrade published with Task 1, and `pyproject.toml` bumped to that
version.

**Files:**
- Modify: `adrs/oms/ops/order_placement_manager.py` (`on_exchange_event` ~line 105, `update_positions` ~line 157)
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/test_position_stream.py` (extend)

**Interfaces:**
- Consumes: `EventType.PositionUpdate` and `Event.data: list[Position]` (Task 1); `apply_stream_positions` (Task 2).
- Produces: no new public names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_position_stream.py`:

```python
def test_a_position_update_event_reaches_the_position_manager():
    from cybotrade.io.event import Event, EventType

    from adrs.oms.ops.order_placement_manager import OrderPlacementManager

    opm = object.__new__(OrderPlacementManager)
    opm.position = _pm()
    asyncio.run(
        opm.on_exchange_event(
            Event(
                event_type=EventType.PositionUpdate,
                orig="{}",
                data=[_position(BTC, "0.25")],
            )
        )
    )
    assert opm.position.exchange[BTC].quantity == Decimal("0.25")


def test_an_order_fill_no_longer_moves_the_exchange_position():
    """
    The retired writer. update_positions used to do `exchange += asset_filled`,
    which drifted; ACCOUNT_UPDATE now owns that value absolutely. `pending` keeps
    its incremental accounting, corrected each tick by the open-orders snapshot.
    """
    from cybotrade.models import OrderSide

    from adrs.oms.ops.order_placement_manager import OrderPlacementManager

    opm = object.__new__(OrderPlacementManager)
    opm.position = _pm()
    opm.position.exchange = {BTC: _position(BTC, "1.0")}
    opm.position.pending = {BTC: _position(BTC, "1.0")}
    opm.order_pools = SimpleNamespace(order_value_update={})

    update = SimpleNamespace(
        client_order_id="coid",
        symbol=BTC,
        side=OrderSide.BUY,
        filled_size=Decimal("0.3"),
    )
    opm.update_positions(update)

    assert opm.position.exchange[BTC].quantity == Decimal("1.0"), (
        "exchange must be owned by the stream, not moved by fills"
    )
    assert opm.position.pending[BTC].quantity == Decimal("0.7"), (
        "pending keeps its incremental accounting"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_position_stream.py -q -p no:cacheprovider --no-cov
```
Expected: FAIL — `AttributeError: PositionUpdate`, and the fill test fails because
`exchange` still moves.

- [ ] **Step 3: Bump the dependency**

In `pyproject.toml`, raise the `cybotrade>=` pin to the version published with
Task 1, then:

```bash
uv lock
uv pip install --python ~/bq/research/adrs/.venv/bin/python 'cybotrade==<published version>'
```

Confirm before writing code:

```bash
~/bq/research/adrs/.venv/bin/python -c \
  "from cybotrade.io.event import EventType; print(EventType.PositionUpdate)"
```

- [ ] **Step 4: Add the event case**

In `order_placement_manager.py`'s `on_exchange_event`, add before `case EventType.Error:`:

```python
            case EventType.PositionUpdate:
                logger.debug(f"[ON_EVENT] '{event.event_type}': {event.data}")
                self.position.apply_stream_positions(event.data)
```

This also removes these frames from the `case _:` fallthrough, which is what
produced the `[ON_EVENT] 'EventType.Unknown': None` bursts in production logs.

- [ ] **Step 5: Delete the incremental writer**

In `update_positions`, delete this line:

```python
        self.position.exchange[update.symbol].quantity += asset_filled
```

and extend the docstring/comment above it:

```python
    def update_positions(self, update: OrderUpdate):
        # filled_size is cumulative. Apply only the new slice and store the
        # cumulative total (not the slice) as the next baseline, otherwise the
        # baseline drifts and fills over-count from the third update onward.
        #
        # Only `pending` is maintained here now. The exchange position is owned
        # absolutely by ACCOUNT_UPDATE (see PositionManager.apply_stream_positions):
        # this used to also do `exchange += asset_filled`, which drifted for the
        # same reason the baseline does and relied on REST polling to correct it.
        # `pending` keeps its increment because the open-orders snapshot corrects
        # it every placement tick.
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
~/bq/research/adrs/.venv/bin/pytest tests/test_position_stream.py -q -p no:cacheprovider --no-cov
~/bq/research/adrs/.venv/bin/pytest tests/ -q -p no:cacheprovider --ignore=tests/integration \
  --ignore=tests/test_datamap.py --ignore=tests/test_dataloader.py --no-cov
```
Expected: 12 passed in the new file; full suite 378 passed + 1 xfailed.

Any pre-existing test asserting that a fill moves `position.exchange` is now
asserting the retired behaviour — update it to assert the new contract rather than
restoring the increment.

- [ ] **Step 7: Lint and commit**

```bash
ruff check adrs/ tests/
ruff format adrs/oms/ops/order_placement_manager.py tests/test_position_stream.py
git add adrs/oms/ops/order_placement_manager.py pyproject.toml uv.lock tests/test_position_stream.py
git commit -m "feat(oms): source exchange positions from ACCOUNT_UPDATE

on_exchange_event handles EventType.PositionUpdate and applies the frame
absolutely, and update_positions loses its 'exchange += asset_filled' line. That
writer drifted by construction and relied on REST polling to correct it; the stream
now owns the value absolutely, so there is nothing to drift.

pending keeps its increment, because the open-orders snapshot corrects it every
placement tick.

These frames also stop falling through to the Unknown case, which is what produced
the [ON_EVENT] 'EventType.Unknown': None bursts in production logs."
```

---

## Task 4: release and validate on one tenant

**Files:** none modified.

- [ ] **Step 1: Release adrs**

Follow the 1.8.1 procedure: bump `pyproject.toml` (minor — `update_positions`
changes observable behaviour for anything reading `position.exchange` after a
fill), `uv lock`, PR, merge, tag, build in a clean tag worktree, scan the artifacts
for secrets, publish with twine, verify installable.

- [ ] **Step 2: Rebuild the compute image and deploy to one tenant**

- [ ] **Step 3: Confirm the stream is actually feeding positions**

In the tenant's logs:

- `[ON_EVENT] 'EventType.Unknown': None` bursts should be **gone** — those were the
  discarded `ACCOUNT_UPDATE` frames.
- `Weight(1m)` on the `[RATE LIMITER STATS]` line should fall by roughly 10/min
  versus 1.8.1, since the placement tick no longer reads positions.
- After a fill, the position reported to aegis must match the exchange. A
  divergence here means the stream is being applied wrongly and is the signal to
  roll back.

- [ ] **Step 4: Watch for the failure this design accepts**

A missed `ACCOUNT_UPDATE` leaves a position wrong for up to 60s. If reported
positions ever disagree with the exchange for longer than that, the anchor is not
running — check that `on_aegis_update` is still calling `update_exchange()`.

---

## Notes for the implementer

**Do not let the stream stamp the REST anchor.** It is the one thing that would
quietly break the whole trust model: the anchor exists to prove a real read
happened recently, and if stream frames refreshed it, a dead REST path would look
healthy forever while the sizing path trusted an unanchored stream.

**Do not filter zero positions anywhere.** A position closing to flat is the most
important frame in the system. Dropping it leaves the OMS sizing orders against
something it does not hold.

**Do not extend the anchor logic to the wallet balance.** `create_equity` reads
REST every call, by rule. See the spec's *Balances*: a stream-sourced equity figure
would look plausible and be wrong.

**Do not add `EventType.BalanceUpdate`.** Nothing consumes it, and a public name in
a published library is the hardest thing to walk back.

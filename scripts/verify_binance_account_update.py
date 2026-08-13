"""
Settle the gating question in
docs/superpowers/specs/2026-08-13-ws-position-stream-design.md: is
ACCOUNT_UPDATE's `pa` the absolute position amount, or a delta?

The whole design rests on it. Absolute means the stream can overwrite
PositionManager.exchange with no arithmetic, which is what retires the
drift-prone incremental fill accounting. A delta would mean absolute overwrite
replaces a true position with the size of the last change -- catastrophically
wrong on a live account.

Binance's documentation would not serve this payload: three fetches of the
user-data-stream pages returned the docs homepage. The identical failure on the
order-book depth question (see the price feed design) turned out to hide an
assumption that was WRONG, and only an experiment caught it. Hence this script.

Also records what a.B (balances) contains, which the spec documents but
deliberately does not consume.

WHAT IT DOES: opens a user-data stream, records every ACCOUNT_UPDATE frame, and
compares the last `pa` seen per symbol against GET /fapi/v3/positionRisk taken
ONCE after the window closes. Read-only -- it reads positions and a stream. It
places no orders, cancels nothing, and transfers nothing.

It polls REST exactly twice, before and after the window, NOT once per frame.
An earlier version polled per frame and earned an HTTP 418 (IP ban) on a shard
also running a live OMS -- the precise failure mode this whole design exists to
remove. A per-frame poll buys nothing anyway: the final state is what `pa` must
agree with, and frames are printed in full for manual inspection regardless.

CREDENTIALS: from the environment only, never CLI arguments (which leak into shell
history and `ps` output):

    export BINANCE_API_KEY=...
    export BINANCE_API_SECRET=...

Prefer --testnet with testnet keys: the payload shape is identical and no live
position is involved. A read-only mainnet key is the next best option.

USAGE:
    python scripts/verify_binance_account_update.py --seconds 900
    python scripts/verify_binance_account_update.py --testnet --seconds 900

IMPORTANT: ACCOUNT_UPDATE only fires when the account CHANGES -- a fill, funding,
or a transfer. A quiet account emits nothing, which is the event-driven behaviour
the design is built around, not a failure of this script. Run it over a window
containing a fill. No frames means the question is UNANSWERED; it does not mean
the assumption passed.
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
from decimal import Decimal, InvalidOperation

import websockets

MAINNET_REST = "https://fapi.binance.com"
MAINNET_WS = "wss://fstream.binance.com"
TESTNET_REST = "https://testnet.binancefuture.com"
TESTNET_WS = "wss://stream.binancefuture.com"


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Export BINANCE_API_KEY and BINANCE_API_SECRET; "
            "see this file's docstring."
        )
    return value


class Client:
    def __init__(self, testnet: bool):
        self.rest = TESTNET_REST if testnet else MAINNET_REST
        self.ws = TESTNET_WS if testnet else MAINNET_WS
        self._key = _require("BINANCE_API_KEY")
        self._secret = _require("BINANCE_API_SECRET")

    def _signed_get(self, path: str) -> list | dict:
        params = {"timestamp": str(int(time.time() * 1000)), "recvWindow": "5000"}
        query = urllib.parse.urlencode(sorted(params.items()))
        signature = hmac.new(
            self._secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        request = urllib.request.Request(
            f"{self.rest}{path}?{query}&signature={signature}",
            headers={"X-MBX-APIKEY": self._key},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def positions(self) -> dict[str, str]:
        """symbol -> positionAmt, as REST reports it. `pa` must match this.

        /fapi/v3/positionRisk OMITS flat symbols entirely rather than listing
        them at zero, so a missing key means flat. Callers must read this via
        `_rest_amount`, not `.get(symbol)` -- treating absent as "no data"
        reports a false MISMATCH on every closed position.
        """
        rows = self._signed_get("/fapi/v3/positionRisk")
        return {row["symbol"]: row["positionAmt"] for row in rows}

    def listen_key(self) -> str:
        request = urllib.request.Request(
            f"{self.rest}/fapi/v1/listenKey",
            method="POST",
            headers={"X-MBX-APIKEY": self._key},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["listenKey"]


def _is_zero(value: str) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return True


def _same(a: str, b: str) -> bool:
    """Exact comparison -- this project mandates Decimal, and this is the
    script that decides whether `pa` is absolute, so a float tolerance here
    would undermine the very question it is answering."""
    try:
        return Decimal(a) == Decimal(b)
    except (TypeError, ValueError, InvalidOperation):
        return False


def _rest_amount(rest: dict[str, str], symbol: str) -> str:
    """positionAmt for `symbol`, treating omission as flat.

    See Client.positions: REST drops flat symbols, so absent and "0" both mean
    the same thing and must compare equal.
    """
    return rest.get(symbol, "0")


def _record_frame(frame_no: int, account: dict, last_seen: dict[str, str]) -> None:
    """Print one frame verbatim and remember the newest `pa` per symbol.

    No REST call here -- that is the whole point. `last_seen` accumulates the
    final claim the stream makes about each symbol, which is what the
    end-of-window REST snapshot gets checked against.
    """
    positions = account.get("P") or []
    print(f"\n--- ACCOUNT_UPDATE #{frame_no}  reason m={account.get('m')} ---")
    print(f"  a.P = {json.dumps(positions)}")
    print(f"  a.B = {json.dumps(account.get('B'))}")
    print(
        f"  frame lists {len(positions)} position(s): {[p.get('s') for p in positions]}"
    )
    for position in positions:
        symbol, pa = position.get("s"), position.get("pa")
        if symbol is not None:
            last_seen[symbol] = pa


def _compare(last_seen: dict[str, str], rest: dict[str, str]) -> int:
    """Check every streamed `pa` against final REST. Returns mismatch count."""
    mismatches = 0
    for symbol, pa in sorted(last_seen.items()):
        reported = _rest_amount(rest, symbol)
        origin = "omitted by REST = flat" if symbol not in rest else "REST"
        if _same(pa, reported):
            print(f"  {symbol}: pa={pa} == {origin} {reported}  -> ABSOLUTE")
        else:
            mismatches += 1
            print(
                f"  {symbol}: pa={pa} != {origin} {reported}  -> MISMATCH, "
                "pa may be a DELTA"
            )
    return mismatches


async def main(seconds: float, testnet: bool) -> None:
    client = Client(testnet=testnet)
    print(f"endpoint: {client.rest}  ({'TESTNET' if testnet else 'MAINNET'})")

    before = client.positions()
    nonzero = {k: v for k, v in before.items() if not _is_zero(v)}
    print(f"REST positions now (non-zero): {nonzero or '(none)'}")
    print(f"listening {seconds:.0f}s for ACCOUNT_UPDATE...", flush=True)

    url = f"{client.ws}/ws/{client.listen_key()}"
    frames = 0
    last_seen: dict[str, str] = {}
    async with websockets.connect(url, ping_interval=20) as socket:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(
                    socket.recv(), timeout=deadline - time.monotonic()
                )
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                break
            message = json.loads(raw)
            if message.get("e") != "ACCOUNT_UPDATE":
                continue
            frames += 1
            _record_frame(frames, message.get("a") or {}, last_seen)

    if frames == 0:
        print("\n=== verdict ===")
        print(
            "  UNANSWERED. No ACCOUNT_UPDATE arrived, so the account did not change\n"
            "  during the window. That is the event-driven behaviour the design\n"
            "  assumes, not a failure -- but it does NOT confirm the assumption.\n"
            "  Rerun over a period containing a fill."
        )
        return

    # The decisive comparison, and the only REST call besides the opening
    # snapshot. The stream's final claim per symbol must equal the account's
    # final state; a delta would not.
    print(f"\nfinal REST snapshot (1 call) for {len(last_seen)} streamed symbol(s):")
    mismatch = _compare(last_seen, client.positions())

    print("\n=== verdict ===")
    if mismatch == 0:
        print(
            f"  ABSOLUTE. Every streamed `pa` across {frames} frame(s) matched the\n"
            "  account's final positionAmt. Safe to overwrite positions from the\n"
            "  stream. Proceed with Task 1."
        )
    else:
        print(
            f"  MISMATCH on {mismatch} of {len(last_seen)} symbol(s) across {frames} "
            "frame(s).\n"
            "  Do NOT proceed. `pa` may be a delta, or frames may be inconsistent\n"
            "  with REST -- either way the design's trust model needs re-opening.\n"
            "  Rule out a genuine account change between the last frame and the\n"
            "  final snapshot first."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=900.0)
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="use testnet; identical payload shape, no live position at risk",
    )
    args = parser.parse_args()
    asyncio.run(main(args.seconds, args.testnet))

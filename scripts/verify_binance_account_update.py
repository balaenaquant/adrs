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

WHAT IT DOES: opens a user-data stream, and for every ACCOUNT_UPDATE frame
compares each `pa` against what GET /fapi/v3/positionRisk reports at that moment.
Read-only -- it polls positions and reads a stream. It places no orders, cancels
nothing, and transfers nothing.

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
        """symbol -> positionAmt, as REST reports it. `pa` must match this."""
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
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return False


def _report_frame(client: Client, frame_no: int, account: dict) -> tuple[int, int]:
    """Print one frame's verdict. Returns (absolute_hits, mismatches)."""
    print(f"\n--- ACCOUNT_UPDATE #{frame_no}  reason m={account.get('m')} ---")
    print(f"  a.P = {json.dumps(account.get('P'))}")
    print(f"  a.B = {json.dumps(account.get('B'))}")

    # The decisive comparison. Taken immediately after the frame, so REST and the
    # frame should describe the same account state.
    rest = client.positions()
    frame_positions = account.get("P") or []
    absolute, mismatch = 0, 0
    for position in frame_positions:
        symbol, pa = position.get("s"), position.get("pa")
        reported = rest.get(symbol)
        if reported is not None and _same(pa, reported):
            absolute += 1
            print(f"  {symbol}: pa={pa} == REST positionAmt={reported}  -> ABSOLUTE")
        else:
            mismatch += 1
            print(
                f"  {symbol}: pa={pa} != REST positionAmt={reported}  -> MISMATCH, "
                "pa may be a DELTA"
            )

    nonzero_rest = sum(1 for v in rest.values() if not _is_zero(v))
    completeness = (
        "PARTIAL (frame lists fewer than the account holds)"
        if len(frame_positions) < nonzero_rest
        else "possibly COMPLETE for this frame"
    )
    print(
        f"  frame lists {len(frame_positions)} position(s); account holds "
        f"{nonzero_rest} non-zero -> {completeness}"
    )
    return absolute, mismatch


async def main(seconds: float, testnet: bool) -> None:
    client = Client(testnet=testnet)
    print(f"endpoint: {client.rest}  ({'TESTNET' if testnet else 'MAINNET'})")

    before = client.positions()
    nonzero = {k: v for k, v in before.items() if not _is_zero(v)}
    print(f"REST positions now (non-zero): {nonzero or '(none)'}")
    print(f"listening {seconds:.0f}s for ACCOUNT_UPDATE...", flush=True)

    url = f"{client.ws}/ws/{client.listen_key()}"
    frames = absolute = mismatch = 0
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
            a, m = _report_frame(client, frames, message.get("a") or {})
            absolute += a
            mismatch += m

    print("\n=== verdict ===")
    if frames == 0:
        print(
            "  UNANSWERED. No ACCOUNT_UPDATE arrived, so the account did not change\n"
            "  during the window. That is the event-driven behaviour the design\n"
            "  assumes, not a failure -- but it does NOT confirm the assumption.\n"
            "  Rerun over a period containing a fill."
        )
        return
    if mismatch == 0:
        print(
            f"  ABSOLUTE. {absolute}/{absolute} position field(s) across {frames} "
            "frame(s) matched REST's positionAmt.\n"
            "  Safe to overwrite positions from the stream. Proceed with Task 1."
        )
    else:
        print(
            f"  MISMATCH on {mismatch} of {absolute + mismatch} position field(s) "
            f"across {frames} frame(s).\n"
            "  Do NOT proceed. `pa` may be a delta, or frames may be inconsistent\n"
            "  with REST -- either way the design's trust model needs re-opening."
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

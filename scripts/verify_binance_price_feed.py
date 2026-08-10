"""
Settle the open question in docs/superpowers/specs/2026-08-11-ws-price-feed-design.md:
does Binance's Partial Book Depth stream push while a book is idle?

The whole design rests on it. If `<symbol>@depth5@100ms` re-pushes on its timer
regardless of book activity, then silence on the feed unambiguously means the feed
is dead, and an illiquid pair never falls back to REST just for being quiet. If it
only pushes on change, it is no better than `@bookTicker` and the fallback plan
applies: `@bookTicker` for prices plus `<symbol>@markPrice@1s` as a per-symbol
liveness beacon.

Subscribes to all three candidate streams on the thinnest symbol available, plus a
liquid control, and reports inter-message gaps and how often consecutive messages
carry an identical book.

Usage:
    python scripts/verify_binance_price_feed.py [--seconds 90] [--symbol BITOUSDT]

Reads no credentials: these are public market streams.

RESULT (2026-08-11, 90s, BITOUSDT as the thinnest USDT perp, ~1.2k trades/24h):

    stream                        msgs   median gap    max gap
    bitousdt@depth5@100ms            6         8046      50592
    bitousdt@bookTicker              2        38167      58660
    btcusdt@depth5@100ms           807          102        517

Partial Book Depth is CHANGE-DRIVEN, not timer-driven. A 100ms timer over 90s
would deliver ~900 messages; the idle book delivered 6, with a 50-second silent
gap. The BTCUSDT control on the same connection delivered 807 at a 102ms median,
which both proves the harness works and shows why the stream looks timer-like on
a liquid symbol -- its book simply never stops moving.

So `@depth5@100ms` cannot distinguish "quiet book" from "dead feed", and the
spec's primary approach is refuted.

Caveat on the rest: probes of markPrice / ticker / kline / aggTrade returned zero
messages even on BTCUSDT, alongside a pyo3 teardown panic and a "stream timer ran
out" warning. `btcusdt@ticker` returning nothing is not credible, so those zeros
are a harness artifact of ad-hoc probing, NOT evidence that those streams are
unavailable. Anything depending on them needs a clean re-test.
"""

import argparse
import asyncio
import json
import statistics
import time
import urllib.request
from collections import defaultdict
from datetime import timedelta

from cybotrade.websocket import Message, Request, conn

FAPI = "https://fapi.binance.com"
# Both forms appear in Binance's docs; BinancePrivateWS uses the un-prefixed one.
# Whichever connects is the one the adapter should use.
WS_BASES = [
    "wss://fstream.binance.com/stream?streams=",
    "wss://fstream.binance.com/public/stream?streams=",
]


def thinnest_usdt_perp() -> str:
    """Lowest 24h quote volume, so 'idle' is a real condition and not a guess."""
    with urllib.request.urlopen(f"{FAPI}/fapi/v1/ticker/24hr", timeout=20) as r:
        rows = json.load(r)
    usdt = [x for x in rows if x["symbol"].endswith("USDT")]
    usdt.sort(key=lambda x: float(x["quoteVolume"]))
    return usdt[0]["symbol"]


def book_fingerprint(data: dict) -> str:
    """
    Identity of the book itself, ignoring timestamps and sequence ids.

    Consecutive messages with the same fingerprint are the tell for a
    timer-driven stream: it re-sent an unchanged book because the timer fired.
    """
    if "b" in data and "a" in data:  # depth payloads
        return json.dumps([data.get("b"), data.get("a")], sort_keys=True)
    if "b" in data or "a" in data:  # bookTicker: b/B/a/A scalars
        return json.dumps(
            [data.get("b"), data.get("B"), data.get("a"), data.get("A")],
            sort_keys=True,
        )
    if "p" in data:  # markPrice
        return str(data.get("p"))
    return json.dumps(data, sort_keys=True)


async def observe(url: str, streams: list[str], seconds: float) -> dict:
    async def on_connected(sender):
        return None

    async def on_heartbeat(sender):
        # Binance pings every 3 min and allows unsolicited pongs; keep it simple
        await sender.send(Message.Pong())

    result = await conn(
        Request(url=url + "/".join(streams)),
        timedelta(seconds=30),
        on_connected,
        on_heartbeat,
    )
    # conn() returns (sender, stream); tolerate either ordering across versions
    stream = next(x for x in result if hasattr(x, "__anext__"))

    last_at: dict[str, float] = {}
    gaps: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    repeats: dict[str, int] = defaultdict(int)
    last_fp: dict[str, str] = {}
    samples: dict[str, dict] = {}

    started_at = time.monotonic()
    deadline = started_at + seconds
    while time.monotonic() < deadline:
        try:
            msg = await asyncio.wait_for(
                stream.__anext__(), timeout=deadline - time.monotonic()
            )
        except (asyncio.TimeoutError, StopAsyncIteration):
            break
        if not isinstance(msg, Message.Text):
            continue
        try:
            body = json.loads(msg.payload)
        except json.JSONDecodeError:
            continue
        # Combined streams wrap as {"stream": name, "data": {...}}
        name = body.get("stream")
        data = body.get("data", body)
        if name is None:
            name = f"{data.get('s', '?')}@{data.get('e', '?')}".lower()

        now = time.monotonic()
        counts[name] += 1
        samples.setdefault(name, data)
        if name in last_at:
            gaps[name].append((now - last_at[name]) * 1000.0)
        last_at[name] = now

        fp = book_fingerprint(data)
        if last_fp.get(name) == fp:
            repeats[name] += 1
        last_fp[name] = fp

    # A stream that goes silent must not look healthy. Without the trailing gap
    # (last message -> end of observation) a single early burst reads as fine.
    ended_at = time.monotonic()
    for name, at in last_at.items():
        gaps[name].append((ended_at - at) * 1000.0)

    return {
        "counts": dict(counts),
        "gaps": {k: v for k, v in gaps.items()},
        "repeats": dict(repeats),
        "samples": samples,
        "elapsed": ended_at - started_at,
    }


def report(res: dict, seconds: float, thin: str) -> None:
    print(
        f"\n{'stream':<34} {'msgs':>6} {'min':>8} {'med':>8} {'p95':>8} {'max':>9} {'identical':>10}"
    )
    print("-" * 88)
    verdict_rows = {}
    for name in sorted(res["counts"]):
        g = sorted(res["gaps"].get(name, []))
        n = res["counts"][name]
        if g:
            mn, med, mx = g[0], statistics.median(g), g[-1]
            p95 = g[int(len(g) * 0.95) - 1] if len(g) >= 20 else g[-1]
        else:
            mn = med = p95 = mx = float("nan")
        rep = res["repeats"].get(name, 0)
        pct = (rep / max(n - 1, 1)) * 100
        print(
            f"{name:<34} {n:>6} {mn:>8.0f} {med:>8.0f} {p95:>8.0f} {mx:>9.0f} "
            f"{rep:>5} ({pct:>3.0f}%)"
        )
        verdict_rows[name] = (n, mx, pct)

    print("\n(gaps in ms; 'identical' = consecutive msgs carrying an unchanged book)")

    # Binance echoes stream names exactly as subscribed, so match case-insensitively
    def find(suffix: str) -> str | None:
        want = f"{thin}@{suffix}".lower()
        return next((k for k in verdict_rows if k.lower() == want), None)

    thin_depth = find("depth5@100ms")
    thin_bt = find("bookTicker")
    thin_mp = find("markPrice@1s")

    expected = int(res["elapsed"] / 0.1)
    print(
        f"\na 100ms timer over {res['elapsed']:.0f}s would deliver ~{expected} messages"
    )

    print("\n--- verdict ---")
    if thin_depth is not None:
        n, mx, pct = verdict_rows[thin_depth]
        # Timer-driven means it keeps arriving on an idle book: no long silence,
        # and a healthy share of consecutive messages carrying an unchanged book.
        timer_driven = mx < 1000 and n > expected * 0.5
        print(
            f"{thin} depth5@100ms: {n} msgs, max gap {mx:.0f}ms, {pct:.0f}% identical"
        )
        if timer_driven:
            print(
                "  => TIMER-DRIVEN. It re-sends an unchanged book on its own timer,\n"
                "     so silence means a dead feed. Use @depth5@100ms as the feed;\n"
                "     the spec's primary approach stands."
            )
        else:
            print(
                "  => CHANGE-DRIVEN. An idle book goes quiet, so this cannot\n"
                "     distinguish quiet from dead. Use the spec's fallback:\n"
                "     @bookTicker for prices + @markPrice@1s as a liveness beacon."
            )
    else:
        print(f"  no {thin}@depth5@100ms messages at all -- check stream name/URL form")

    for label, key in (("bookTicker", thin_bt), ("markPrice@1s", thin_mp)):
        if key is None:
            print(f"{thin} {label}: NO MESSAGES RECEIVED -- stream name may be wrong")
            continue
        n, mx, pct = verdict_rows[key]
        print(f"{thin} {label}: {n} msgs, max gap {mx:.0f}ms, {pct:.0f}% identical")
        if label == "markPrice@1s":
            ok = mx < 2000 and n > res["elapsed"] * 0.5
            print(f"  => {'usable' if ok else 'NOT usable'} as a 1s liveness beacon")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--symbol", default=None, help="default: thinnest USDT perp")
    args = ap.parse_args()

    thin = args.symbol or thinnest_usdt_perp()
    print(f"thin symbol: {thin}  (control: BTCUSDT)")

    streams = [
        f"{thin.lower()}@depth5@100ms",
        f"{thin.lower()}@bookTicker",
        f"{thin.lower()}@markPrice@1s",
        "btcusdt@depth5@100ms",
    ]

    last_err = None
    for base in WS_BASES:
        print(f"connecting: {base}...")
        try:
            res = await observe(base, streams, args.seconds)
        except Exception as e:  # noqa: BLE001 - probing which URL form works
            print(f"  failed: {e}")
            last_err = e
            continue
        if sum(res["counts"].values()) == 0:
            print("  connected but received nothing; trying next URL form")
            continue
        print(f"  OK via {base}")
        report(res, args.seconds, thin)
        s = res["samples"].get(f"{thin.lower()}@depth5@100ms")
        if s:
            print(f"\nsample depth5 payload:\n{json.dumps(s)[:400]}")
        return
    raise SystemExit(f"no URL form produced messages (last error: {last_err})")


if __name__ == "__main__":
    asyncio.run(main())

"""Find the minimum IP rate limit a single base-OMS needs to converge, as a
function of the number of child orders (K) it must place/cancel/reprice.

Two knobs (both per-run):
  --orders K       target number of child orders the split produces
  --ip-limit L     total IP rate limit (req / 1s window) for this OMS,
                   BEFORE the config soft_limit reduces it

The OMS carries its own limiter. A place draws UID_PLACE (~8/s buffered), a
cancel draws UID_CANCEL (~8/s), each ALSO counting against IP_GLOBAL. So a
reprice storm (cancel ‖ re-place ‖ order-detail reconfirm ‖ orderbook reads)
can peak well above the place-only rate. This harness drives that storm by
flipping the target side, then measures how many placement intervals the book
takes to fully churn. The IP limit is "sufficient" for a K if the book churns
within --intervals.

  effective_ip = ip_limit * soft_limit_percent   (your "divide then reduce")

A too-low IP limit throttles LOCALLY (LocalRateLimitError) and never reaches
the exchange, so sweeping down is ban-safe.

Usage:
    python scripts/oms_loadtest.py -f scratch_loadtest_base.json \
        --symbol BTCUSDT --orders 10 --ip-limit 30 --intervals 4
    # sweep IP downward for a fixed K to find the floor:
    python scripts/oms_loadtest.py -f scratch_loadtest_base.json \
        --symbol BTCUSDT --orders 10 --sweep 120,80,60,40,30,20,15,10
"""

import os
import copy
import json
import time
import asyncio
import logging
import argparse
from types import SimpleNamespace
from collections import deque, Counter
from decimal import Decimal

from adrs.logging import setup_logger, make_colorlog_stream_handler

from cybotrade import Symbol

from adrs.oms.oms import OMS
from adrs.oms.config import Config, FileConfigManager
from adrs.oms.calculation import Calculate
from adrs.oms.ops.order_executer import OrderExecutor, PlacementContext
from adrs.oms.rate_limit.rate_limiter import BybitRateLimiter
from adrs.oms.rate_limit.exchange_limit_profiles import BybitRateLimitPool


class Runaway(Exception):
    """Order pool grew past the Bybit 50-order page limit — pending under-counts
    and the OMS over-places. Abort the trial rather than flood the account."""


# --------------------------------------------------------------------------- #
# Config extended with the two sweep knobs (documented extension point:
# ConfigManager.config_cls). num_orders feeds the splitter, ip_limit the limiter.
# --------------------------------------------------------------------------- #
class SweepConfig(Config):
    num_orders: int = 1
    ip_limit: int = 120
    # Fraction away from the touch to post, so post-only orders REST instead of
    # filling at BBO — exercises the full cancel/reprice worst-case path.
    entry_offset: str = "0"


class SweepConfigManager(FileConfigManager):
    config_cls = SweepConfig


class KSplitExecutor(OrderExecutor):
    """Split a target into exactly num_orders child orders (capped by min_qty)."""

    async def split_order_quantity(
        self, qty: Decimal, ctx: PlacementContext
    ) -> list[Decimal]:
        k = getattr(self.config, "num_orders", 1)
        sizes = Calculate.generate_random_order_size(
            sum_total=qty,
            count=k,
            min_qty=ctx.min_qty,
            precision=ctx.symbol_info.quantity_precision,
            max_qty=ctx.max_qty,
        )
        return [s for s in sizes if s > Decimal("0")]

    async def compute_limit_offsets(self, level, ctx) -> list[Decimal]:
        return [self._offset()] * level

    def _offset(self) -> Decimal:
        return Decimal(str(getattr(self.config, "entry_offset", "0")))

    # Base reprices post at offset 0 (the touch), which on demo fills as a taker
    # and accumulates real position. Keep the resting offset so reprices rest
    # too — the test measures request churn, not fills.
    async def reprice_at_bbo(self, symbol, qty, side, replace_best_bid_ask_time,
                             package_id, initial_price=None, initial_time=None):
        return await self.place_single_limit_order(
            symbol=symbol, offset=self._offset(), qty=qty, side=side,
            symbol_info=self.symbol_infos[symbol],
            replace_best_bid_ask_time=replace_best_bid_ask_time,
            package_id=package_id, initial_price=initial_price,
            initial_time=initial_time,
        )

    async def reprice_at_mid(self, symbol, qty, side, replace_best_bid_ask_time,
                             package_id, initial_price, initial_time=None):
        return await self.place_single_limit_order(
            symbol=symbol, offset=self._offset(), qty=qty, side=side,
            symbol_info=self.symbol_infos[symbol],
            replace_best_bid_ask_time=replace_best_bid_ask_time,
            package_id=package_id, initial_price=initial_price,
            initial_time=initial_time,
        )


class SweepOMS(OMS):
    executor_cls = KSplitExecutor


class NullMetricStream:
    """No-op MetricStream: swallows every publish so the test writes nothing to
    aegis, and accepts subscribes without a NATS connection. Signals are driven
    in-process, so no real subject delivery is needed. Exchange REST reads (which
    the n*K+k model counts) go through the exchange client, not this stream, so
    they are unaffected."""

    async def init(self):
        pass

    async def subscribe(self, subject, callback=None):
        pass

    async def publish(self, subject, payload, **kwargs):
        pass


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
class Recorder:
    """Every reserved slot passes record_usage; count = exact IP requests.
    Tracks a rolling 1s peak (IP window is 1s) plus per-endpoint totals."""

    def __init__(self):
        self.events: deque = deque()  # monotonic ts of each request
        self.per_ep: Counter = Counter()
        self.total = 0
        self.peak_1s = 0
        self.throttles = 0
        self.bans = 0

    def reset(self):
        """Zero counts after the flatten phase so a trial measures only the
        build+churn requests, not the residual-dependent flatten cost."""
        self.events.clear()
        self.per_ep.clear()
        self.total = 0
        self.peak_1s = 0
        self.throttles = 0
        self.bans = 0

    def record(self, endpoint_name: str):
        now = time.monotonic()
        self.events.append(now)
        self.per_ep[endpoint_name] += 1
        self.total += 1
        while self.events and self.events[0] < now - 1.0:
            self.events.popleft()
        self.peak_1s = max(self.peak_1s, len(self.events))


class InstrumentedIPRateLimiter(BybitRateLimiter):
    """BybitRateLimiter with a configurable IP_GLOBAL ceiling and telemetry.
    The IP limit is reduced by the config soft_limit_percent; UID pools keep
    their real defaults so the place/cancel ceilings stay realistic."""

    def __init__(self, config, ip_limit: int, recorder: Recorder):
        super().__init__(config)
        self.limit_profile.limits[BybitRateLimitPool.IP_GLOBAL] = max(
            1, int(Decimal(ip_limit) * self.soft_limit_percentage)
        )
        self._recorder = recorder

    def record_usage(self, endpoint):
        self._recorder.record(endpoint.name)
        return super().record_usage(endpoint)

    def check_limits(self, endpoint, **kwargs) -> bool:
        ok = super().check_limits(endpoint, **kwargs)
        if not ok:
            self._recorder.throttles += 1
        return ok

    def local_cache_error(self, headers):
        self._recorder.bans += 1
        return super().local_cache_error(headers)


# --------------------------------------------------------------------------- #
# Trial
# --------------------------------------------------------------------------- #
def _write_config(base: dict, symbol: str, k: int, ip_limit: int, path: str):
    cfg = copy.deepcopy(base)
    cfg["oms_id"] = f"{base['oms_id']}_sweep"
    cfg["portfolio_id"] = f"{base['portfolio_id']}_sweep"
    cfg["num_orders"] = k
    cfg["ip_limit"] = ip_limit
    cfg["entry_offset"] = base.get("entry_offset", "0.002")
    # Churn cadence (placement interval, expiry, replace windows) comes from the
    # base config verbatim — earlier hardcoded 2s/4s overrides here made every
    # order perpetually expired and amplified churn beyond anything realistic.
    # Single-symbol run so K maps cleanly to one book
    base_asset = symbol.replace("USDT", "")
    cfg["base_asset_to_symbol_table"] = {base_asset: symbol}
    with open(path, "w") as f:
        json.dump(cfg, f)
    return cfg, base_asset


def _needed_weight(cfg: dict, min_qty: Decimal, price: Decimal, k: int) -> Decimal:
    """Weight that yields qty ~= (k+1)*min_qty so the split can make K orders."""
    bal = Decimal(str(cfg["initial_balance"]))
    lev = Decimal(str(cfg["leverage"]))
    target_qty = (k + 1) * min_qty
    return (target_qty * price) / (bal * lev)


async def _book_state(oms: OMS, symbol: Symbol):
    async with oms.opm.order_pools.get_order_pool() as pool:
        resting = sum(1 for o in pool.values() if o.symbol == str(symbol))
    desired = oms.position.desired.get(symbol)
    pending = oms.position.pending.get(symbol)
    exch = oms.position.exchange.get(symbol)
    d = desired.quantity if desired else Decimal("0")
    p = (pending.quantity if pending else Decimal("0")) + (
        exch.quantity if exch else Decimal("0")
    )
    return resting, d, p


def _signal_msg(base_asset: str, weight: Decimal):
    """Build a portfolio-signal Msg the OMS handler accepts. Driving the signal
    in-process (vs publishing to the shared NATS) makes the test deterministic:
    no subscribe/publish delivery race, no latency, no cross-talk with real
    portfolios on the shared server."""
    payload = {
        "assets": {base_asset: str(weight)},
        "timestamp": int(time.time() * 1000),
    }
    return SimpleNamespace(data=json.dumps(payload).encode(), reply="")


async def _inject_signal(oms: OMS, base_asset: str, weight: Decimal):
    # Sets latest_signal exactly as the NATS callback would; the scheduled
    # on_process_latest_signal tick then reprices desired from it.
    await oms.on_portfolio_signal(_signal_msg(base_asset, weight))  # type: ignore[arg-type]


async def run_trial(base, symbol_str, k, ip_limit, intervals, ms):
    symbol = Symbol(symbol_str)
    os.makedirs("scratch_loadtest", exist_ok=True)
    cfg_path = "scratch_loadtest/sweep.json"
    cfg, base_asset = _write_config(base, symbol_str, k, ip_limit, cfg_path)

    config = SweepConfigManager(cfg_path)
    await config.setup()
    recorder = Recorder()
    limiter = InstrumentedIPRateLimiter(config, ip_limit, recorder)
    await limiter.init()
    eff_ip = limiter.limit_profile.limits[BybitRateLimitPool.IP_GLOBAL]

    info = config.symbol_infos[symbol]
    price = await config.exchange.get_current_price(symbol=symbol)
    if not price or price <= 0:
        logging.error(f"[SWEEP] could not fetch price for {symbol}")
        return None
    min_qty = max(info.min_limit_qty, info.min_notional / price)
    weight = _needed_weight(cfg, min_qty, price, k)
    if weight > Decimal("1"):
        logging.error(
            f"[SWEEP] K={k} needs weight {weight:.3f} > 1 "
            f"(balance*leverage too small for {k} x min_qty {min_qty} @ {price}). "
            f"Raise initial_balance/leverage or lower K."
        )
        return None

    oms = SweepOMS(config=config, metric_stream=ms, rate_limiter=limiter)
    run_task = asyncio.create_task(oms.run())
    await asyncio.sleep(2)  # let subscriptions + init settle

    interval = cfg["order_placement_interval"]

    async def converged(target_side_weight: Decimal) -> int | None:
        """Inject a target, return #intervals until the book fully churns to it,
        or None if it exceeds the bound. Raises Runaway if the pool balloons
        past the Bybit 50-order page limit (pending under-counts there and the
        OMS over-places) — that regime is unsafe and must not be let to grow."""
        await _inject_signal(oms, base_asset, target_side_weight)
        for i in range(1, intervals + 1):
            await asyncio.sleep(interval + 1)
            resting, d, held = await _book_state(oms, symbol)
            remaining = abs(d - held)
            logging.info(
                f"[SWEEP] K={k} ip={ip_limit}(eff {eff_ip}) interval {i}/{intervals} "
                f"resting={resting} |desired-held|={remaining} "
                f"peak1s={recorder.peak_1s} throttled={recorder.throttles}"
            )
            if resting > 45:
                raise Runaway(f"pool ballooned to {resting} (>45), over-placing")
            if remaining < min_qty:
                return i
        return None

    # Flatten residual account position first so the build delta equals the
    # target, not target-minus-leftovers from a prior run.
    held = Decimal("0")
    await _inject_signal(oms, base_asset, Decimal("0"))
    for _ in range(3):
        await asyncio.sleep(interval + 1)
        _, _, held = await _book_state(oms, symbol)
        if abs(held) < min_qty:
            break
    logging.info(f"[SWEEP] K={k} flattened, held={held}")

    # Measure only build+churn from here.
    recorder.reset()

    # Phase A: build the book long. Phase B: flip short -> cancel storm + rebuild.
    runaway = False
    try:
        ia = await converged(weight)
        ib = await converged(-weight)
    except Runaway as e:
        logging.error(f"[SWEEP] K={k} ip={ip_limit}: RUNAWAY — {e}")
        ia = ib = None
        runaway = True

    # Snapshot telemetry BEFORE shutdown: _handle_shutdown cancels every resting
    # order (~K cancels), which would inflate build+churn counts if included.
    snap = {
        "peak_1s": recorder.peak_1s,
        "total_req": recorder.total,
        "throttled": recorder.throttles,
        "banned": recorder.bans,
        "per_ep": dict(recorder.per_ep),
    }

    # Cancelling run_task runs OMS.run()'s finally -> _handle_shutdown, which
    # raises SystemExit(0); gather(return_exceptions=True) re-raises SystemExit,
    # so swallow it explicitly or it kills the whole sweep.
    run_task.cancel()
    try:
        await asyncio.gather(run_task, return_exceptions=True)
    except BaseException:
        # _handle_shutdown raises SystemExit(0); do not let it end the sweep.
        pass

    # A cooldown still active means an exchange 403/10006 fired mid-trial and
    # blocked the measurement window — the counts are meaningless, flag it.
    poisoned = limiter.retry_after >= limiter.get_synced_time_ms() or (
        snap["total_req"] == 0 and snap["throttled"] > 0
    )
    ok = ia is not None and ib is not None
    result = {
        "k": k,
        "ip_limit": ip_limit,
        "eff_ip": eff_ip,
        "converged": ok,
        "poisoned": poisoned or runaway,
        "runaway": runaway,
        "intervals_build": ia,
        "intervals_churn": ib,
        **snap,
        "top_endpoints": Counter(snap["per_ep"]).most_common(6),
    }
    logging.info(f"[SWEEP] RESULT {json.dumps({k2: str(v) for k2, v in result.items()})}")
    return result


def _fit_nk(ks: list[int], totals: list[int]) -> tuple[float, float]:
    """Least-squares fit total = n*K + k. Returns (n, k)."""
    m = len(ks)
    if m < 2:
        return 0.0, float(totals[0]) if totals else 0.0
    sk = sum(ks)
    st = sum(totals)
    skk = sum(x * x for x in ks)
    skt = sum(x * y for x, y in zip(ks, totals))
    denom = m * skk - sk * sk
    if denom == 0:
        return 0.0, st / m
    n = (m * skt - sk * st) / denom
    k = (st - n * sk) / m
    return n, k


async def main():
    parser = argparse.ArgumentParser(prog="oms_ip_sweep")
    parser.add_argument("-f", "--file", default="scratch_loadtest_base.json")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--orders", type=int, default=10, help="K child orders")
    parser.add_argument("--ip-limit", type=int, default=30)
    parser.add_argument(
        "--sweep",
        default=None,
        help="comma list of ip-limits to try for the fixed K, high->low",
    )
    parser.add_argument(
        "--sweep-k",
        default=None,
        help="comma list of K values to try (IP held high); fits requests = n*K + k",
    )
    parser.add_argument(
        "--high-ip",
        type=int,
        default=120,
        help="IP limit for --sweep-k: high enough not to throttle a single OMS "
        "(~30/s peak) but MUST stay <= the real Bybit ceiling (~120/s) or the "
        "local limiter stops protecting and the exchange 403-bans the IP 10 min",
    )
    parser.add_argument("--intervals", type=int, default=4, help="convergence bound")
    args = parser.parse_args()

    setup_logger(log_level=logging.INFO, handlers=[make_colorlog_stream_handler()])

    with open(args.file) as f:
        base = json.load(f)

    ms = NullMetricStream()
    await ms.init()

    if args.sweep_k:
        ks = [int(x) for x in args.sweep_k.split(",")]
        rows = []
        for kk in ks:
            logging.info(f"[SWEEP-K] ===== K={kk} ip_limit={args.high_ip} (no throttle) =====")
            try:
                r = await run_trial(base, args.symbol, kk, args.high_ip, args.intervals, ms)
            except BaseException as e:
                logging.error(f"[SWEEP-K] K={kk} trial crashed: {e!r}")
                r = None
            if r:
                rows.append(r)
            await asyncio.sleep(3)
        logging.info("[SWEEP-K] ==== requests_per_churn = n*K + k ====")
        for r in rows:
            logging.info(
                f"[SWEEP-K] K={r['k']:3d} total_req={r['total_req']:4d} "
                f"peak1s={r['peak_1s']:3d} throttled={r['throttled']} "
                f"{'POISONED' if r.get('poisoned') else ''} per_ep={r['per_ep']}"
            )
        good = [r for r in rows if not r.get("poisoned") and r["total_req"] > 0]
        if len(good) < len(rows):
            logging.warning(
                f"[SWEEP-K] excluded {len(rows) - len(good)} poisoned trial(s) from the fit"
            )
        if len(good) >= 2:
            n, kconst = _fit_nk([r["k"] for r in good], [r["total_req"] for r in good])
            logging.info(f"[SWEEP-K] FIT: requests_per_churn ≈ {n:.2f}*K + {kconst:.2f}")
            logging.info(
                f"[SWEEP-K] => per-order n≈{n:.2f}, fixed k≈{kconst:.2f}. "
                f"min_ip_limit(K) ≈ (n*K + k) / (soft_limit * window_seconds)"
            )
        return

    ip_limits = (
        [int(x) for x in args.sweep.split(",")] if args.sweep else [args.ip_limit]
    )

    results = []
    for ip in ip_limits:
        logging.info(f"[SWEEP] ===== trial K={args.orders} ip_limit={ip} =====")
        try:
            r = await run_trial(base, args.symbol, args.orders, ip, args.intervals, ms)
        except BaseException as e:
            logging.error(f"[SWEEP] ip={ip} trial crashed: {e!r}")
            r = None
        if r:
            results.append(r)
        await asyncio.sleep(3)  # let cancels settle between trials

    interval = base.get("order_placement_interval", 10)
    window_s = args.intervals * interval
    logging.info(f"[SWEEP] ==== SUMMARY: safe = throttle clears within {window_s}s, no ban ====")
    floor = None
    for r in sorted(results, key=lambda x: x["ip_limit"]):
        # safe = both phases recovered within the window AND no real exchange ban
        safe = r["converged"] and not r.get("poisoned")
        clear = max(r["intervals_build"] or 999, r["intervals_churn"] or 999) * interval
        clear_s = f"{clear}s" if clear < 999 * interval else ">window"
        tag = (
            "SAFE" if safe
            else "RUNAWAY" if r.get("runaway")
            else "BANNED" if r.get("poisoned")
            else "UNSAFE"
        )
        logging.info(
            f"[SWEEP] ip={r['ip_limit']:3d} eff={r['eff_ip']:3d} peak1s={r['peak_1s']:3d} "
            f"throttled={r['throttled']:4d} clears={clear_s:>7} -> {tag}"
        )
        if safe:
            floor = r["ip_limit"]
    logging.info(
        f"[SWEEP] ===> LOWEST SAFE ip_limit for K={args.orders}: {floor} "
        f"(throttle clears within {window_s}s, no exchange ban)"
    )


if __name__ == "__main__":
    asyncio.run(main())

"""Evaluator.eval must compute P&L in chronological order on any Polars thread count.

Polars 2.0 stopped keeping the left row order of a multi-threaded left join
unless `maintain_order` is set. `eval` joins signals onto the resampled price
grid and then runs order-dependent steps (shift, pct_change, forward_fill,
cum_sum), so a shuffled join silently corrupts pnl and equity.

Every scenario resamples 1m candles, as the exchange feeds deliver them, and
compares `eval` row by row against a plain-Python reference.

The Polars thread pool size is fixed at import, so each thread count runs all
scenarios in one subprocess with POLARS_MAX_THREADS set explicitly. On a single
thread the bug never shows; the multi-threaded runs are the regression check.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

SCENARIOS = {
    # name: interval_min, phase_min, buckets, sparse, shuffle, price_shift, window
    "1m_to_5m": (5, 0, 800, False, False, 0, None),
    "1m_to_1h": (60, 0, 720, False, False, 0, None),
    "1m_to_4h": (240, 0, 360, False, False, 0, None),
    "1m_to_1d": (1440, 0, 200, False, False, 0, None),
    "phase_1h_offset_30m": (60, 30, 720, False, False, 0, None),
    "phase_4h_offset_1h": (240, 60, 360, False, False, 0, None),
    "sparse_signals_forward_filled": (60, 0, 720, True, False, 0, None),
    "unsorted_signal_input": (60, 0, 720, False, True, 0, None),
    "price_shift_1": (60, 0, 720, False, False, 1, None),
    "price_shift_5": (60, 0, 720, False, False, 5, None),
    "sub_window": (60, 0, 720, False, False, 0, (100, 600)),
    "combined": (60, 15, 720, True, True, 3, (50, 650)),
}

SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import random
    import sys
    from datetime import datetime, timedelta, timezone

    import polars as pl

    from adrs.data.datamap import Datamap
    from adrs.data.types import DataColumn, DataInfo
    from adrs.performance import Evaluator
    from adrs.types import Topic

    TOPIC = "binance|candle?symbol=BTCUSDT&interval=1m"
    INFO = DataInfo(topic=TOPIC, columns=[DataColumn(src="close", dst="price")], lookback_size=1)
    START = datetime(2024, 1, 1, tzinfo=timezone.utc)  # aligned to every interval used
    FEES = 0.05
    MIN = timedelta(minutes=1)


    class FakeDatamap(Datamap):
        def __init__(self, prices):
            super().__init__([INFO])
            self.topics = {Topic.from_str(TOPIC)}
            self.prices = prices

        def keys(self):
            return [Topic.from_str(TOPIC)]

        def get(self, info):
            return self.prices


    def scenario(seed, interval_min, phase_min, buckets, sparse, shuffle, price_shift, window):
        rng = random.Random(seed)
        # One spare bucket of minutes at the end so price_shift never runs off the data.
        n_min = phase_min + (buckets + 1) * interval_min
        p, minute_prices = 100.0, []
        for _ in range(n_min):
            p *= 1.0 + rng.uniform(-0.001, 0.001)
            minute_prices.append(p)
        prices = pl.DataFrame({
            "start_time": [START + i * MIN for i in range(n_min)],
            "price": minute_prices,
        })

        bucket_time = [START + (phase_min + b * interval_min) * MIN for b in range(buckets)]
        raw = {b: float(rng.choice((-1, 0, 1))) for b in range(buckets)
               if not sparse or rng.random() < 0.3}
        rows = sorted(raw)
        if shuffle:
            rng.shuffle(rows)
        signal_lf = pl.LazyFrame({
            "start_time": [bucket_time[b] for b in rows],
            "signal": [raw[b] for b in rows],
        })

        lo, hi = window or (0, buckets)
        out = (
            Evaluator(assets={"BTC": INFO})
            .eval(signal_lf=signal_lf, base_asset="BTC", datamap=FakeDatamap(prices),
                  start_time=bucket_time[lo], end_time=bucket_time[hi - 1] + interval_min * MIN,
                  fees=FEES, interval=timedelta(minutes=interval_min),
                  price_shift=price_shift)
            .collect()
        )

        # Plain-Python reference. A bucket's price is the last 1m price inside it,
        # after shifting the minute series back by price_shift. The window filter
        # runs before forward_fill / shift / pct_change, so state resets at `lo`.
        expected_pnl, equity, prev_s, prev_p = [], 0.0, 0.0, None
        for b in range(lo, hi):
            close = minute_prices[phase_min + (b + 1) * interval_min - 1 + price_shift]
            s = raw.get(b, prev_s)
            ret = 0.0 if prev_p is None else close / prev_p - 1.0
            pnl = prev_s * ret - abs(s - prev_s) * FEES / 100
            equity += pnl
            expected_pnl.append(pnl)
            prev_s, prev_p = s, close

        got_times = [t.replace(tzinfo=timezone.utc) for t in
                     out["start_time"].dt.replace_time_zone(None).to_list()]
        if got_times != bucket_time[lo:hi]:
            first = next((i for i, (a, b) in enumerate(zip(got_times, bucket_time[lo:hi]))
                          if a != b), min(len(got_times), hi - lo))
            return (f"rows not chronological: {out.height} rows (want {hi - lo}), "
                    f"first mismatch at row {first}")
        worst = max(abs(a - b) for a, b in zip(out["pnl"].to_list(), expected_pnl))
        if worst > 1e-12:
            return f"pnl differs from reference by up to {worst:.3e}"
        if abs(out["equity"][-1] - equity) > 1e-9:
            return f"final equity {out['equity'][-1]} != reference {equity}"
        return "OK"


    want = int(os.environ["POLARS_MAX_THREADS"])
    if pl.thread_pool_size() != want:
        sys.exit(f"thread pool is {pl.thread_pool_size()}, expected {want}")
    results = {}
    for seed, (name, args) in enumerate(json.loads(sys.argv[1]).items()):
        try:
            results[name] = scenario(seed, *args)
        except Exception as exc:  # report per scenario, never hide
            results[name] = f"{type(exc).__name__}: {exc}"
    print("RESULTS " + json.dumps(results))
    """
)

_cache: dict[str, dict[str, str]] = {}


def _results(threads: str) -> dict[str, str]:
    if threads not in _cache:
        env = dict(os.environ, POLARS_MAX_THREADS=threads)
        proc = subprocess.run(
            [sys.executable, "-c", SCRIPT, json.dumps(SCENARIOS)],
            env=env,
            capture_output=True,
            text=True,
            check=False,  # the assert below reports stdout/stderr on failure
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULTS ")]
        assert proc.returncode == 0 and line, proc.stdout + proc.stderr
        _cache[threads] = json.loads(line[0][len("RESULTS ") :])
    return _cache[threads]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("threads", ["1", "8"])
def test_eval_pnl_matches_chronological_reference(threads, scenario):
    assert _results(threads)[scenario] == "OK"

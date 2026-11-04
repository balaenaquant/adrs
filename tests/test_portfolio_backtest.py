"""Portfolio.backtest when assets do not share every timestamp.

Production builds prices_df as `btc.join(eth, on="start_time", how="full",
coalesce=True)`. A candle missing on one asset leaves a null there, so that
asset's alphas have no bar at that time. Two bugs followed from that:

* pnl was forward-filled across those rows, repeating the alpha's last bar of
  pnl (and only for alphas after the first, so results depended on alpha order);
* prices_df was used in whatever order the caller passed it, with no sort
  before pct_change; Polars promises no row order for a full join.
"""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from adrs.portfolio import Portfolio

S = datetime(2024, 1, 1, tzinfo=timezone.utc)
TIMES = [S + timedelta(hours=h) for h in range(10)]
BTC = [100.0 + h for h in range(10)]
# ETH missing candles at hours 4..6
ETH = [10.0, 10.5, 11.0, 12.0, None, None, None, 12.6, 12.0, 13.2]


def _expected_total(prices, weight):
    """Always-long alpha, no fees: weighted sum of returns over the bars it has."""
    bars = [p for p in prices if p is not None]
    return weight * sum(b / a - 1.0 for a, b in zip(bars, bars[1:]))


def _run(prices_df, eth_alpha_first: bool):
    ids = ["ETH_LONG", "BTC_FLAT"] if eth_alpha_first else ["BTC_FLAT", "ETH_LONG"]
    assets = {"ETH_LONG": "ETH", "BTC_FLAT": "BTC"}
    signals = {"ETH_LONG": 1.0, "BTC_FLAT": 0.0}
    signal_df = pl.DataFrame(
        {"start_time": TIMES, **{i: [signals[i]] * len(TIMES) for i in ids}}
    )
    metadata_df = pl.DataFrame(
        {
            "custom_id": ids,
            "fees": [0.0, 0.0],
            "shift_backtest_candle_minute": [0, 0],
            "base_asset": [assets[i] for i in ids],
        }
    )
    weight_df = pl.DataFrame({"custom_id": ids, "weights": [0.5, 0.5]})
    return Portfolio("p", signal_df, metadata_df, weight_df).backtest(prices_df)


@pytest.mark.parametrize("eth_alpha_first", [True, False])
def test_missing_candles_earn_nothing_in_either_alpha_order(eth_alpha_first):
    prices_df = pl.DataFrame({"start_time": TIMES, "BTC": BTC, "ETH": ETH})
    perf, df = _run(prices_df, eth_alpha_first)

    expected = _expected_total(ETH, 0.5)
    assert df["equity"][-1] == pytest.approx(expected, abs=1e-12)
    assert perf.total_return == pytest.approx(expected, abs=1e-12)
    gap = df.filter(pl.col("start_time").is_between(TIMES[4], TIMES[6]))
    assert gap["ETH_LONG_pnl"].to_list() == [0.0, 0.0, 0.0]


def test_unsorted_full_join_prices_match_sorted():
    # BTC missing hour 5, so the full join has an ETH-only row.
    btc = pl.DataFrame({"start_time": TIMES, "BTC": BTC}).filter(
        pl.col("start_time") != TIMES[5]
    )
    eth = pl.DataFrame(
        {"start_time": TIMES, "ETH": [10.0 + 0.3 * h for h in range(10)]}
    )
    joined = btc.join(eth, on="start_time", how="full", coalesce=True)
    shuffled = joined.sort("start_time", descending=True)  # worst case ordering

    _, from_sorted = _run(joined.sort("start_time"), eth_alpha_first=True)
    for prices_df in (joined, shuffled):
        _, got = _run(prices_df, eth_alpha_first=True)
        assert got["start_time"].to_list() == from_sorted["start_time"].to_list()
        assert got["pnl"].to_list() == pytest.approx(from_sorted["pnl"].to_list())

    eth_prices = [10.0 + 0.3 * h for h in range(10)]
    assert from_sorted["equity"][-1] == pytest.approx(_expected_total(eth_prices, 0.5))

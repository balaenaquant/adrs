"""run_portfolio renames Alpha ids via alpha_id_map; _remap_portfolio_ids keeps
the portfolio's frames in the same id-space so live signals (published under the
renamed id) actually match the portfolio lookup."""

import polars as pl

from adrs.execution.runner import _remap_portfolio_ids
from adrs.portfolio import Portfolio


def _portfolio() -> Portfolio:
    metadata_df = pl.DataFrame(
        {
            "custom_id": ["a1", "a2"],
            "base_asset": ["BTC", "ETH"],
            "fees": [0.0, 0.0],
            "shift_backtest_candle_minute": [0, 0],
        }
    )
    signal_df = pl.DataFrame({"start_time": [1, 2], "a1": [1, 1], "a2": [0, -1]})
    weight_df = pl.DataFrame({"custom_id": ["a1", "a2"], "weights": [0.5, 0.5]})
    return Portfolio(
        id="pf", signal_df=signal_df, metadata_df=metadata_df, weight_df=weight_df
    )


def test_remap_rewrites_all_frames():
    pf = _portfolio()
    _remap_portfolio_ids(pf, {"a1": "u_a1", "a2": "u_a2"})
    assert pf.metadata_df["custom_id"].to_list() == ["u_a1", "u_a2"]
    assert pf.weight_df["custom_id"].to_list() == ["u_a1", "u_a2"]
    assert set(pf.signal_df.columns) == {"start_time", "u_a1", "u_a2"}


def test_live_signal_matches_after_remap():
    pf = _portfolio()
    _remap_portfolio_ids(pf, {"a1": "u_a1", "a2": "u_a2"})
    # a live signal arrives under the RENAMED id (what AlphaExecutor publishes)
    pf.update_signal("u_a1", 1)
    out = pf.get_signal()  # raises if a custom_id isn't found in either source
    assert "u_a1" in pf.lastest_signal
    # BTC weighted by 0.5 from the live signal (1 * 0.5)
    btc = out.filter(pl.col("base_asset") == "BTC")["weighted_signal"][0]
    assert btc == 0.5


def test_unmapped_ids_unchanged():
    pf = _portfolio()
    _remap_portfolio_ids(pf, {"a1": "u_a1"})  # a2 not in map
    assert pf.metadata_df["custom_id"].to_list() == ["u_a1", "a2"]
    assert set(pf.signal_df.columns) == {"start_time", "u_a1", "a2"}

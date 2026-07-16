import json
import polars as pl
from typing import TypedDict

from pathlib import Path


class Metadata(TypedDict):
    """Canonical metadata_df row — the schema Portfolio and
    generate_signal_df consume."""

    custom_id: str
    base_asset: str
    # execution delay in minutes after the decision (see Portfolio.backtest)
    shift_backtest_candle_minute: int
    # percent of notional per unit of turnover (0.035 = 3.5 bps per leg)
    fees: float


def contruct_metadata_df(file_path: Path) -> pl.DataFrame | None:
    metadata_df: pl.DataFrame | None = None
    metadata_json = json.load(open(file_path))
    alphas = metadata_json["bot_info"]

    for value in alphas.values():
        metadata = Metadata(
            custom_id=value["custom_id"],
            base_asset=value["base_asset"],
            shift_backtest_candle_minute=value["shift_backtest_candle_minute"],
            fees=value["fees"],
        )
        if metadata_df is None:
            metadata_df = pl.DataFrame(metadata)
        else:
            metadata_df = metadata_df.vstack(pl.DataFrame(metadata))

    return metadata_df

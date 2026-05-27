import json
import polars as pl
from typing import TypedDict

from pathlib import Path


class Metadata(TypedDict):
    id: str
    base_asset: str
    price_shift: int
    fees: float


def contruct_metadata_df(file_path: Path) -> pl.DataFrame | None:
    metadata_df: pl.DataFrame | None = None
    metadata_json = json.load(open(file_path))
    alphas = metadata_json["bot_info"]

    for value in alphas.values():
        metadata = Metadata(
            id=value["custom_id"],
            base_asset=value["base_asset"],
            price_shift=value["shift_backtest_candle_minute"],
            fees=value["fees"],
        )
        if metadata_df is None:
            metadata_df = pl.DataFrame(metadata)
        else:
            metadata_df = metadata_df.vstack(pl.DataFrame(metadata))

    return metadata_df

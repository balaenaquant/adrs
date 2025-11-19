import pytest
import polars as pl
from datetime import datetime

from adrs.data.dataloader import DataLoader


async def custom_data_handler(topic: str, start_time: datetime, end_time: datetime):
    if topic != "custom|my-data":
        return None

    return pl.read_parquet("tests/data/a_BTC_i_1h_2025-01-03.parquet")


@pytest.mark.asyncio
async def test_dataloader_with_custom_handler():
    dataloader = DataLoader(data_dir="output", handlers=[custom_data_handler])

    df = await dataloader.load(
        topic="custom|my-data",
        start_time=datetime.fromisoformat("2025-10-11T00:00:00Z"),
        end_time=datetime.fromisoformat("2025-11-11T00:00:00Z"),
    )
    df.equals(pl.read_parquet("tests/data/a_BTC_i_1h_2025-01-03.parquet"))

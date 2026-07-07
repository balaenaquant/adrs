import os
import pytest
import polars as pl
from datetime import datetime

from adrs.data.dataloader import DataLoader

TEST_DF = pl.DataFrame([{"a": 1, "c": 2}, {"a": 2, "c": 1}])


async def custom_data_handler(topic: str, start_time: datetime, end_time: datetime):
    if topic != "custom|my-data":
        return None

    return TEST_DF


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dataloader_with_custom_handler():
    dataloader = DataLoader(
        data_dir="output",
        credentials={"datasource_api_key": os.getenv("DATASOURCE_API_KEY")},
        handlers=[custom_data_handler],
    )

    df = await dataloader.load(
        topic="custom|my-data",
        start_time=datetime.fromisoformat("2025-10-11T00:00:00Z"),
        end_time=datetime.fromisoformat("2025-11-11T00:00:00Z"),
    )
    df.equals(TEST_DF)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dataloader_datasource():
    dataloader = DataLoader(
        data_dir="output",
        credentials={"datasource_api_key": os.getenv("DATASOURCE_API_KEY")},
    )

    df = await dataloader.load(
        topic="bybit-linear|candle?symbol=BTCUSDT&interval=1h",
        start_time=datetime.fromisoformat("2025-10-11T00:00:00Z"),
        end_time=datetime.fromisoformat("2025-11-11T00:00:00Z"),
    )
    assert len(df.rows()) != 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dataloader_clickhouse():
    dataloader = DataLoader(
        data_dir="output",
        credentials={
            "type": "clickhouse",
            "host": os.getenv("CLICKHOUSE_HOST"),
            "port": os.getenv("CLICKHOUSE_PORT"),
            "username": os.getenv("CLICKHOUSE_USER"),
            "password": os.getenv("CLICKHOUSE_PASS"),
            "secure": True,
            "verify": False,
        },
    )

    df = await dataloader.load(
        topic="binance-linear|candle?symbol=BTCUSDT&interval=1h",
        start_time=datetime.fromisoformat("2025-10-11T00:00:00Z"),
        end_time=datetime.fromisoformat("2025-11-11T00:00:00Z"),
    )
    assert len(df.rows()) != 0

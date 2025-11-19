import json
import asyncio
import logging
import polars as pl
from datetime import datetime

from cybotrade.logging import setup_logger

from adrs import DataLoader


async def custom_handler(topic: str, start_time: datetime, end_time: datetime):
    if topic != "custom|data":
        return None

    logging.info("custom_handler: loading custom data...")
    return pl.read_parquet(r"tests/data/a_BTC_i_1h_2025-01-01.parquet")


async def main():
    setup_logger(log_level=logging.INFO)

    start_time, end_time = (
        datetime.fromisoformat("2024-01-01T00:00:00Z"),
        datetime.fromisoformat("2024-01-02T00:00:00Z"),
    )

    symbol = "USDTUSD"
    dataloader = DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
        handlers=[custom_handler],
    )
    df = await dataloader.load(
        topic=f"coinbase|candle?symbol={symbol}&interval=15m",
        start_time=start_time,
        end_time=end_time,
    )
    print(df)
    df.write_parquet(f"coinbase_candle_interval_15m_symbol_{symbol}.parquet")

    df = await dataloader.load(
        topic="custom|data",
        start_time=start_time,
        end_time=end_time,
    )
    print(df)
    df.write_parquet("custom_data.parquet")


if __name__ == "__main__":
    asyncio.run(main())

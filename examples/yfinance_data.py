import json
import logging
import asyncio
from datetime import datetime

from adrs import DataLoader
from adrs.data import yfinance_handler

from adrs.logging import setup_logger


async def main():
    setup_logger(log_level=logging.INFO)

    start_time, end_time = (
        datetime.fromisoformat("2026-01-01T00:00:00Z"),
        datetime.fromisoformat("2026-02-01T00:00:00Z"),
    )

    dataloader = DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
        handlers=[yfinance_handler],
    )
    df = await dataloader.load(
        topic="yfinance|candle?ticker=SPY&interval=1d",
        start_time=start_time,
        end_time=end_time,
    )
    print(df)
    df = await dataloader.load(
        topic="yfinance|candle?ticker=TSLA&interval=1d",
        start_time=start_time,
        end_time=end_time,
    )
    print(df)
    df = await dataloader.load(
        topic="yfinance|candle?ticker=BTC&interval=1d",
        start_time=start_time,
        end_time=end_time,
    )
    print(df)


if __name__ == "__main__":
    asyncio.run(main())

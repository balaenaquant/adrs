import pandas as pd
import polars as pl
import yfinance as yf
from datetime import datetime

from cybotrade import Topic


async def yfinance_handler(topic: str, start_time: datetime, end_time: datetime):
    _topic = Topic.from_str(topic)
    if _topic.provider() != "yfinance":
        return

    match _topic.endpoint():
        case "candle":
            ticker = _topic.query_params().get("ticker")
            if not ticker:
                raise ValueError(f"Topic {_topic} is missing 'ticker'")

            interval = _topic.query_params().get("interval", "1d")

            ticker = yf.Ticker(ticker)
            df: pd.DataFrame = ticker.history(
                start=start_time.strftime("%Y-%m-%d"),
                end=end_time.strftime("%Y-%m-%d"),
                interval=interval,
            )
            return pl.from_pandas(df.reset_index()).select(
                pl.col("Date").alias("start_time").dt.convert_time_zone("UTC"),
                pl.col("Open").alias("open"),
                pl.col("High").alias("high"),
                pl.col("Low").alias("low"),
                pl.col("Close").alias("close"),
                pl.col("Volume").alias("volume").cast(pl.Float64),
            )
        case _:
            raise ValueError(f"Unsupported endpoint: {_topic.endpoint()}")

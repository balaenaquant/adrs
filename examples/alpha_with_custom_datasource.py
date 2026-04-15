import json
import httpx
import asyncio
import logging
import polars as pl
from typing import override, cast
from datetime import datetime, timedelta

from pathlib import Path
from adrs import Alpha, DataLoader
from adrs.report import AlphaReportV1
from adrs.performance import Evaluator
from adrs.utils import backforward_split
from adrs.data import DataInfo, DataColumn, Datamap
from adrs.tests import Sensitivity, SensitivityParameter

from adrs.types import Data, Topic
from adrs.logging import setup_logger
from adrs.data.processor import DataProcessor
from adrs.data.datasource import Datasource, ms_to_dt
from adrs.data.cache import Cache

logger = logging.getLogger(__name__)


class CustomDatasource(Datasource):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(
            api_key,
            base_url,
            max_limit=1000,
        )
        self.client = httpx.AsyncClient(timeout=60)

    async def query(
        self,
        topic: Topic | str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        flatten: bool = False,
    ) -> list[Data]:
        if isinstance(topic, str):
            topic = Topic.from_str(topic)
        params: dict[str, str] = dict(topic.params)
        if start_time is not None:
            params["startTime"] = str(int(start_time.timestamp() * 1000))
        if end_time is not None:
            params["endTime"] = str(int(end_time.timestamp() * 1000))
        if limit is not None:
            params["limit"] = str(limit)

        try:
            resp = await self.client.get(
                url=self.base_url + topic.endpoint,
                params=params,
            )
            resp.raise_for_status()
            batch = resp.json()
            return cast(
                list[Data],
                [
                    {
                        "start_time": ms_to_dt(item[0]),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    }
                    for item in batch
                ],
            )

        except Exception as e:
            logger.error(f"Failed to fetch data due to {e}")
            raise


# NOTE: This function dictates the handling for custom topic which we used here.
#       Override this with your own handler to download from external API
def custom_handler(
    datasource: CustomDatasource,
    cache: Cache,
):
    async def handler(input_topic: str, start_time: datetime, end_time: datetime):
        topic = Topic(
            provider="binance",
            endpoint="/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": "500"},
        )
        if str(topic) != input_topic:
            return None
        cache.init(topic=topic, start_time=start_time, end_time=end_time)

        datapoints = await cache.download(datasource, topic)

        logger.info("[%s] downloaded %d datapoints", topic, datapoints)
        return await cache.read(topic, start_time, end_time)

    return handler


class CoinbaseBinancePremiumAlpha(Alpha):
    def __init__(
        self, window: int, long_entry_threshold: float, long_exit_threshold: float
    ) -> None:
        super().__init__(
            id="coinbase_binance_premium_zscore",
            data_infos=[
                DataInfo(
                    topic="binance|/api/v3/klines?interval=1h&limit=500&symbol=BTCUSDT",
                    columns=[DataColumn(src="close", dst="close_binance_custom")],
                    lookback_size=window,
                ),
                DataInfo(
                    topic="coinbase|candle?symbol=BTCUSD&interval=1h",
                    columns=[DataColumn(src="close", dst="close_coinbase")],
                    lookback_size=window,
                ),
            ],
            data_processor=DataProcessor(),
        )
        self.window = window
        self.long_entry_threshold = long_entry_threshold
        self.long_exit_threshold = long_exit_threshold

    @override
    def next(self, data_df: pl.DataFrame):
        # alpha formula
        df = data_df.select(
            pl.col("start_time"),
            (pl.col("close_coinbase") - pl.col("close_binance_custom")).alias("data"),
        )

        # pre-process
        df = df

        # modeling
        df = df.with_columns(
            (
                (pl.col("data") - pl.col("data").rolling_mean(self.window))
                / pl.col("data").rolling_std(self.window, ddof=1)
            ).alias("zscore")
        ).filter(pl.col("zscore").is_finite())

        # signal
        df = df.with_columns(
            pl.when(pl.col("zscore") >= self.long_entry_threshold)
            .then(1)
            .when(pl.col("zscore") <= self.long_exit_threshold)
            .then(0)
            .otherwise(None)
            .forward_fill()
            .fill_null(strategy="zero")
            .alias("signal")
        )

        return df


async def main():
    setup_logger(log_level=logging.INFO)

    rolling_window = 40
    fees = 0.035
    base_asset = "BTC"
    start_time, end_time = (
        datetime.fromisoformat("2020-05-11T00:00:00Z"),
        datetime.fromisoformat("2025-01-01T00:00:00Z"),
    )

    custom_datasource = CustomDatasource(
        api_key="",
        base_url="https://api.binance.com",
    )

    cache = Cache(
        data_path=Path("outdir"),
        format="parquet",
        override_existing=False,
    )

    binance_custom = custom_handler(datasource=custom_datasource, cache=cache)

    dataloader = DataLoader(
        data_dir="outdir",
        credentials=json.load(open("credentials.json")),
        handlers=[binance_custom],
    )
    evaluator = Evaluator(
        assets={
            "BTC": DataInfo(
                topic="bybit-linear|candle?symbol=BTCUSDT&interval=1m",
                columns=[DataColumn(src="close", dst="price")],
                lookback_size=0,
            )
        }
    )
    alpha = CoinbaseBinancePremiumAlpha(
        window=rolling_window,
        long_entry_threshold=0.825,
        long_exit_threshold=-0.825,
    )

    datamap = Datamap()

    # Setup the datamap (download data)
    await datamap.init(
        dataloader=dataloader,
        infos=alpha.data_infos,
        start_time=start_time,
        end_time=end_time,
    )

    # download data with (+1 day offset for candle shift)
    await datamap.init(
        dataloader=dataloader,
        infos=list(evaluator.assets.values()),
        start_time=start_time,
        end_time=end_time + timedelta(days=1),
    )

    data_df = alpha.data_processor.process(datamap)
    if data_df is None:
        raise Exception("Failed to process datamap to get the datas_df")

    performance, df = alpha.backtest(
        evaluator=evaluator,
        base_asset=base_asset,
        datamap=datamap,
        data_df=data_df,
        start_time=start_time,
        end_time=end_time,
        fees=fees,
        price_shift=10,  # assume 10 minutes delay
    )
    print(performance)
    print(df)

    B_start, B_end, F_start, F_end = backforward_split(
        start_time=start_time, end_time=end_time, size=(0.7, 0.3)
    )
    sensitivity = Sensitivity(
        alpha=alpha,
        parameters={
            "window": SensitivityParameter(min_val=10, min_gap=25),
            "long_entry_threshold": SensitivityParameter(min_val=0.1),
        },
        gap_percent=0.15,
    )

    report = AlphaReportV1.compute(
        alpha,
        B_start,
        B_end,
        F_start,
        F_end,
        sensitivity,
        evaluator=evaluator,
        base_asset=base_asset,
        datamap=datamap,
        data_df=data_df,
        fees=fees,
        price_shift=10,  # assume 10 minutes delay
    )
    print("backtest", report.back.sensitivity_sr_summary)
    print("forward test", report.forward.sensitivity_sr_summary)

    report.write_parquet("alpha_with_external_data.parquet")


if __name__ == "__main__":
    asyncio.run(main())

import json
import polars as pl

from pathlib import Path
from datetime import datetime, timedelta, timezone

from adrs import Portfolio
from adrs.performance import Evaluator
from adrs.execution import generate_signal_df
from adrs.data import DataColumn, DataInfo, DataLoader, make_datamap

from adrs.portfolio import MeanWeightAllocator
from example_alphas import BTC_ALPHAS, ETH_ALPHAS


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


def _load_metadata(config_path: Path) -> pl.DataFrame:
    with open(config_path) as f:
        bot_info = json.load(f)["bot_info"]
    return pl.DataFrame(list(bot_info.values()), infer_schema_length=None).select(
        pl.col("custom_id"),
        pl.col("base_asset"),
        pl.col("shift_backtest_candle_minute")
        .abs()
        .alias("shift_backtest_candle_minute"),
        pl.col("fees"),
    )


async def build_inputs(
    alphas,
    evaluator: Evaluator,
    config_path: Path,
    dataloader: DataLoader,
    start_time: datetime,
    end_time: datetime,
):
    metadata_df = _load_metadata(config_path)
    data_infos = list(flat_map(lambda a: a.data_infos, alphas))
    datamap = await make_datamap(
        dataloader=dataloader,
        start_time=start_time,
        end_time=end_time,
        data_infos=data_infos,
        evaluator=evaluator,
        evaluator_offset=timedelta(),
    )

    signal_df = generate_signal_df(
        alphas=alphas,
        metadata_df=metadata_df,
        evaluator=evaluator,
        datamap=datamap,
        start_time=start_time,
        end_time=end_time,
    )
    assert signal_df is not None

    btc_prices = datamap.get(evaluator.assets["BTC"]).select(
        pl.col("start_time"), pl.col("price").alias("BTC")
    )
    eth_prices = datamap.get(evaluator.assets["ETH"]).select(
        pl.col("start_time"), pl.col("price").alias("ETH")
    )
    prices_df = btc_prices.join(eth_prices, on="start_time", how="full", coalesce=True)
    prices_df = prices_df.with_columns(pl.col("start_time").alias("timestamp"))
    signal_df = signal_df.with_columns(pl.col("start_time").alias("timestamp"))

    return signal_df, metadata_df, prices_df


async def setup_portfolio(config_path: str, cache_dir: str = "output"):
    alphas = BTC_ALPHAS + ETH_ALPHAS
    meta_path = Path(config_path)

    evaluator = Evaluator(
        assets={
            "BTC": DataInfo(
                topic="bybit-linear|candle?symbol=BTCUSDT&interval=1m",
                columns=[DataColumn(src="close", dst="price")],
                lookback_size=0,
            ),
            "ETH": DataInfo(
                topic="binance-spot|candle?symbol=ETHUSDT&interval=1m",
                columns=[DataColumn(src="close", dst="price")],
                lookback_size=0,
            ),
        }
    )

    credentials = json.load(open("credentials.json"))
    dataloader = DataLoader(
        data_dir=cache_dir,
        credentials=credentials,
        cybotrade_api_url="https://api.flow.balaenaquant.com",
    )

    start_time = datetime.fromisoformat("2026-01-01T00:00:00Z")
    end_time = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)

    signal_df, metadata_df, prices_df = await build_inputs(
        alphas, evaluator, meta_path, dataloader, start_time, end_time
    )

    def allocator_factory(signal_df, metadata_df, prices_df):
        return MeanWeightAllocator(signal_df=signal_df, metadata_df=metadata_df)

    weight_df = allocator_factory(signal_df, metadata_df, prices_df).weights()

    portfolio = Portfolio(
        id="bqp_ml021_hrp",
        signal_df=signal_df,
        metadata_df=metadata_df,
        weight_df=weight_df,
    )

    return portfolio, alphas

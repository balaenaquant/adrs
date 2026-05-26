import os
import json
import polars as pl

from datetime import datetime, timezone, timedelta

from adrs import Alpha
from adrs.performance import Evaluator
from adrs.execution.portfolio import Portfolio
from adrs.data import DataColumn, DataInfo, DataLoader, make_datamap
from adrs.execution import MeanWeightAllocator, generate_signal_df

from examples.portfolio_sample.alpha001 import Alpha001
from examples.portfolio_sample.alpha002 import Alpha002
from examples.portfolio_sample.alpha003 import Alpha003
from examples.portfolio_sample.alpha004 import Alpha004
from examples.portfolio_sample.alpha005 import Alpha005


def getenv(name: str) -> str:
    env = os.getenv(name)
    if env is None:
        raise ValueError(f"{name} is not present in environment")
    return env


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


async def setup_portfolio():
    start_time, end_time = (
        datetime.fromisoformat("2026-03-01T00:00:00Z"),
        datetime.now(tz=timezone.utc).replace(second=0, microsecond=0),
    )
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

    btc_alphas: list[Alpha] = [
        Alpha001(window=100, entry_threshold=0.5, exit_threshold=0),
        Alpha002(),
        Alpha003(),
    ]

    eth_alphas: list[Alpha] = [
        Alpha004(),
        Alpha005(),
    ]

    all_alphas = btc_alphas + eth_alphas
    metadata_df = pl.DataFrame(
        {
            "custom_id": [a.id for a in all_alphas],
            "base_asset": ["BTC"] * len(btc_alphas) + ["ETH"] * len(eth_alphas),
            "shift_backtest_candle_minute": [70] * len(all_alphas),
            "fees": [0.035] * len(all_alphas),
        }
    )

    data_infos: list[DataInfo] = list(flat_map(lambda a: a.data_infos, all_alphas))

    credentials = json.load(open("credentials.json"))

    dataloader = DataLoader(
        data_dir="output",
        credentials=credentials,
    )

    datamap = await make_datamap(
        dataloader=dataloader,
        start_time=start_time,
        end_time=end_time,
        data_infos=data_infos,
        evaluator=evaluator,
        evaluator_offset=timedelta(0),
    )

    signal_df = generate_signal_df(
        alphas=all_alphas,
        metadata_df=metadata_df,
        evaluator=evaluator,
        datamap=datamap,
        start_time=start_time,
        end_time=end_time,
    )

    assert signal_df is not None
    weight_df = MeanWeightAllocator(signal_df, metadata_df).weights()

    portfolio = Portfolio(
        id="PORTFOLIO",
        signal_df=signal_df,
        metadata_df=metadata_df,
        weight_df=weight_df,
    )
    return portfolio, all_alphas

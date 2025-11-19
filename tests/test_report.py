import json
import polars as pl
from datetime import datetime, timedelta

from adrs.report.alpha import (
    AlphaReportV1,
    AlphaReportV1Performance,
    Performance,
    PerformanceDF,
    SensitivitySharpeRatioSummary,
)
from adrs.data import DataInfo, DataColumn, UniformDataProcessor
from adrs.tests.sensitivity import SensitivityParameter
from adrs import DataLoader, Environment, AlphaConfig
from adrs.performance import Evaluator

rolling_window = 250
start_time, end_time = (
    datetime.fromisoformat("2020-05-11T00:00:00Z"),
    datetime.fromisoformat("2025-01-01T00:00:00Z"),
)
fees = 0.035

dataloader = DataLoader(
    data_dir="outdir",
    credentials=json.load(open("credentials.json")),
)
config = AlphaConfig(
    base_asset="BTC",
    data_infos=[
        DataInfo(
            topic="binance-spot|candle?symbol=BTCUSDT&interval=1h",
            columns=[DataColumn(src="close", dst="close_binance_spot")],
            lookback_size=rolling_window,
        ),
        DataInfo(
            topic="coinbase|candle?symbol=BTCUSD&interval=1h",
            columns=[DataColumn(src="close", dst="close_coinbase")],
            lookback_size=rolling_window,
        ),
    ],
    dataloader=dataloader,
    data_processor=UniformDataProcessor(),
    start_time=start_time,
    end_time=end_time,
    environment=Environment.BACKTEST,
)
evaluator = Evaluator(fees=fees, candle_shift=10)
formula = lambda datas_df: datas_df.select(
    pl.col("start_time"),
    (pl.col("close_coinbase") - pl.col("close_binance_spot")).alias("data"),
)

pdf = PerformanceDF.validate(
    pl.DataFrame(
        {
            "start_time": [datetime.fromisoformat("2025-05-01T00:00:00Z")],
            "price": [0.0],
            "data": [1.0],
            "signal": [-1],
            "prev_signal": [0],
            "returns": [0.21],
            "trade": [0],
            "pnl": [0.12],
            "equity": [0.41],
        }
    ).with_columns(
        pl.col("start_time").dt.cast_time_unit("ms"),
        pl.col("signal").cast(pl.Int8),
        pl.col("prev_signal").cast(pl.Int8),
        pl.col("trade").cast(pl.Int8),
    )
)

report = AlphaReportV1(
    alpha_id="TURTLE000_00254",
    params={
        "window": 40,
        "long_entry_thres": 0.825,
        "long_exit_thres": -0.825,
        "max_hold_duration": timedelta(minutes=5),
    },
    sensitivity_params={"window": SensitivityParameter(min_val=10, min_gap=25)},
    back=AlphaReportV1Performance(
        performance=Performance(
            sharpe_ratio=1.5669,
            calmar_ratio=1.0243,
            sortino_ratio=1.526,
            cagr=0.43,
            annualized_return=0.6747,
            total_return=3.135,
            min_cumu=-0.0525,
            largest_loss=-0.0813,
            num_datapoints=2000,
            num_trades=2989,
            avg_holding_time_in_seconds=48420.602,
            long_trades=1494,
            short_trades=1495,
            win_trades=716,
            lose_trades=779,
            win_streak=10,
            lose_streak=19,
            win_rate=0.4789,
            start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
            end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
            max_drawdown=-0.6587,
            max_drawdown_percentage=-0.2044,
            max_drawdown_start_date=datetime.fromisoformat("2021-10-20T23:00:00+00:00"),
            max_drawdown_end_date=datetime.fromisoformat("2022-06-22T20:00:00+00:00"),
            max_drawdown_recover_date=datetime.fromisoformat(
                "2023-06-21T15:00:00+00:00"
            ),
            max_drawdown_max_duration_in_days=608.6667,
            metadata={
                "params": {
                    "window": 40,
                    "long_entry_thres": 0.825,
                    "long_exit_thres": -0.825,
                    "max_hold_duration": timedelta(minutes=5),
                }
            },
        ),
        performance_df=pdf,
        sensitivity=[],
        sensitivity_sr_summary=SensitivitySharpeRatioSummary(
            best_param=1.5669,
            mean=1.2988857142857142,
            median=1.5209,
            std=0.36585418510918355,
            min=0.7658,
            max=1.6393,
            p25=1.1606,
            p75=1.573,
            num_negative=0,
            num_positive=7,
            total_permutations=7,
            score=1.642017517750836,
        ),
    ),
    forward=AlphaReportV1Performance(
        performance=Performance(
            sharpe_ratio=1.5669,
            calmar_ratio=1.0243,
            sortino_ratio=1.526,
            cagr=0.43,
            annualized_return=0.6747,
            total_return=3.135,
            min_cumu=-0.0525,
            largest_loss=-0.0813,
            num_datapoints=2000,
            num_trades=2989,
            avg_holding_time_in_seconds=48420.602,
            long_trades=1494,
            short_trades=1495,
            win_trades=716,
            lose_trades=779,
            win_streak=10,
            lose_streak=19,
            win_rate=0.4789,
            start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
            end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
            max_drawdown=-0.6587,
            max_drawdown_percentage=-0.2044,
            max_drawdown_start_date=datetime.fromisoformat("2021-10-20T23:00:00+00:00"),
            max_drawdown_end_date=datetime.fromisoformat("2022-06-22T20:00:00+00:00"),
            max_drawdown_recover_date=datetime.fromisoformat(
                "2023-06-21T15:00:00+00:00"
            ),
            max_drawdown_max_duration_in_days=608.6667,
            metadata={},
        ),
        performance_df=pdf,
        sensitivity=[
            (
                {"window": 40, "long_entry_thres": 0.825},
                Performance(
                    sharpe_ratio=1.5669,
                    calmar_ratio=1.0243,
                    sortino_ratio=1.526,
                    cagr=0.43,
                    annualized_return=0.6747,
                    total_return=3.135,
                    min_cumu=-0.0525,
                    largest_loss=-0.0813,
                    num_datapoints=2000,
                    num_trades=2989,
                    avg_holding_time_in_seconds=48420.602,
                    long_trades=1494,
                    short_trades=1495,
                    win_trades=716,
                    lose_trades=779,
                    win_streak=10,
                    lose_streak=19,
                    win_rate=0.4789,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.6587,
                    max_drawdown_percentage=-0.2044,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-10-20T23:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-06-22T20:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-06-21T15:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=608.6667,
                    metadata={},
                ),
            ),
            (
                {"window": 65, "long_entry_thres": 0.949},
                Performance(
                    sharpe_ratio=1.573,
                    calmar_ratio=1.1438,
                    sortino_ratio=1.4613,
                    cagr=0.43,
                    annualized_return=0.6805,
                    total_return=3.1619,
                    min_cumu=-0.0249,
                    largest_loss=-0.0986,
                    num_datapoints=2000,
                    num_trades=2104,
                    avg_holding_time_in_seconds=67171.482,
                    long_trades=1052,
                    short_trades=1052,
                    win_trades=503,
                    lose_trades=549,
                    win_streak=8,
                    lose_streak=12,
                    win_rate=0.4781,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.5949,
                    max_drawdown_percentage=-0.184,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-10-20T15:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-06-18T21:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-01-14T00:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=450.375,
                    metadata={},
                ),
            ),
            (
                {"window": 15, "long_entry_thres": 0.701},
                Performance(
                    sharpe_ratio=1.1606,
                    calmar_ratio=0.6261,
                    sortino_ratio=1.1812,
                    cagr=0.43,
                    annualized_return=0.5181,
                    total_return=2.4072,
                    min_cumu=-0.032,
                    largest_loss=-0.0813,
                    num_datapoints=2000,
                    num_trades=5282,
                    avg_holding_time_in_seconds=28813.631,
                    long_trades=2641,
                    short_trades=2641,
                    win_trades=1304,
                    lose_trades=1337,
                    win_streak=12,
                    lose_streak=14,
                    win_rate=0.4938,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.8275,
                    max_drawdown_percentage=-0.2492,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-11-01T01:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-02-24T13:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-12-20T13:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=779.5,
                    metadata={},
                ),
            ),
            (
                {"window": 90, "long_entry_thres": 1.073},
                Performance(
                    sharpe_ratio=1.6393,
                    calmar_ratio=1.2738,
                    sortino_ratio=1.485,
                    cagr=0.43,
                    annualized_return=0.6936,
                    total_return=3.223,
                    min_cumu=-0.0021,
                    largest_loss=-0.0986,
                    num_datapoints=2000,
                    num_trades=1688,
                    avg_holding_time_in_seconds=78863.033,
                    long_trades=844,
                    short_trades=844,
                    win_trades=393,
                    lose_trades=451,
                    win_streak=10,
                    lose_streak=10,
                    win_rate=0.4656,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.5445,
                    max_drawdown_percentage=-0.1883,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-10-20T15:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-06-22T20:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-01-20T16:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=457.0417,
                    metadata={},
                ),
            ),
            (
                {"window": 10, "long_entry_thres": 0.577},
                Performance(
                    sharpe_ratio=0.8657,
                    calmar_ratio=0.5886,
                    sortino_ratio=0.9009,
                    cagr=0.43,
                    annualized_return=0.3995,
                    total_return=1.8563,
                    min_cumu=-0.1192,
                    largest_loss=-0.0813,
                    num_datapoints=2000,
                    num_trades=6709,
                    avg_holding_time_in_seconds=23858.718,
                    long_trades=3354,
                    short_trades=3355,
                    win_trades=1677,
                    lose_trades=1678,
                    win_streak=11,
                    lose_streak=13,
                    win_rate=0.4999,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.6788,
                    max_drawdown_percentage=-0.2558,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-10-25T16:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-11-21T21:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-11-09T12:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=744.8333,
                    metadata={},
                ),
            ),
            (
                {"window": 115, "long_entry_thres": 1.197},
                Performance(
                    sharpe_ratio=1.5209,
                    calmar_ratio=1.528,
                    sortino_ratio=1.3123,
                    cagr=0.43,
                    annualized_return=0.632,
                    total_return=2.9367,
                    min_cumu=-0.0401,
                    largest_loss=-0.0986,
                    num_datapoints=2000,
                    num_trades=1346,
                    avg_holding_time_in_seconds=92915.304,
                    long_trades=673,
                    short_trades=673,
                    win_trades=312,
                    lose_trades=361,
                    win_streak=9,
                    lose_streak=10,
                    win_rate=0.4636,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.4136,
                    max_drawdown_percentage=-0.1483,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-10-20T15:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-06-22T20:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-01-16T00:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=452.375,
                    metadata={},
                ),
            ),
            (
                {"window": 10, "long_entry_thres": 0.453},
                Performance(
                    sharpe_ratio=0.7658,
                    calmar_ratio=0.4279,
                    sortino_ratio=0.8145,
                    cagr=0.43,
                    annualized_return=0.3612,
                    total_return=1.6782,
                    min_cumu=-0.1246,
                    largest_loss=-0.0813,
                    num_datapoints=2000,
                    num_trades=7113,
                    avg_holding_time_in_seconds=23530.053,
                    long_trades=3556,
                    short_trades=3557,
                    win_trades=1771,
                    lose_trades=1786,
                    win_streak=12,
                    lose_streak=15,
                    win_rate=0.4979,
                    start_time=datetime.fromisoformat("2021-10-20T23:00:00Z"),
                    end_time=datetime.fromisoformat("2024-10-20T23:00:00Z"),
                    max_drawdown=-0.8441,
                    max_drawdown_percentage=-0.3203,
                    max_drawdown_start_date=datetime.fromisoformat(
                        "2021-11-01T01:00:00+00:00"
                    ),
                    max_drawdown_end_date=datetime.fromisoformat(
                        "2022-11-21T21:00:00+00:00"
                    ),
                    max_drawdown_recover_date=datetime.fromisoformat(
                        "2023-12-20T04:00:00+00:00"
                    ),
                    max_drawdown_max_duration_in_days=779.125,
                    metadata={},
                ),
            ),
        ],
        sensitivity_sr_summary=SensitivitySharpeRatioSummary(
            best_param=1.5669,
            mean=1.2988857142857142,
            median=1.5209,
            std=0.36585418510918355,
            min=0.7658,
            max=1.6393,
            p25=1.1606,
            p75=1.573,
            num_negative=0,
            num_positive=7,
            total_permutations=7,
            score=1.642017517750836,
        ),
    ),
)


def test_reportv1_serde():
    buf = report.serialize()
    report2 = AlphaReportV1.deserialize(buf)
    assert report == report2

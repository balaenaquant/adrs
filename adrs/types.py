import pandera.polars as pa
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from pandera.engines.polars_engine import DateTime, Float64, Int8
from typing import Any

from cybotrade import Topic, Symbol

from cybotrade_datasource import Data


class PriceDF(pa.DataFrameModel):
    start_time: DateTime = pa.Field(
        dtype_kwargs={"time_unit": "ms", "time_zone": "UTC"}
    )
    price: Float64


class SignalDF(pa.DataFrameModel):
    start_time: DateTime = pa.Field(
        dtype_kwargs={"time_unit": "ms", "time_zone": "UTC"}
    )
    data: Float64
    signal: Int8


class PerformanceDF(pa.DataFrameModel):
    start_time: DateTime = pa.Field(
        dtype_kwargs={"time_unit": "ms", "time_zone": "UTC"}
    )
    price: Float64
    data: Float64 = pa.Field(nullable=True)
    signal: Float64 = pa.Field(coerce=True)
    prev_signal: Float64 = pa.Field(coerce=True)
    returns: Float64
    trade: Float64 = pa.Field(coerce=True)
    pnl: Float64
    equity: Float64


class Performance(BaseModel):
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    cagr: float
    annualized_return: float
    total_return: float
    min_cumu: float
    largest_loss: float
    num_datapoints: int
    num_trades: int
    avg_holding_time_in_seconds: float
    long_trades: int
    short_trades: int
    win_trades: int
    lose_trades: int
    win_streak: int
    lose_streak: int
    win_rate: float
    start_time: datetime
    end_time: datetime
    max_drawdown: float
    max_drawdown_percentage: float
    max_drawdown_start_date: datetime
    max_drawdown_end_date: datetime
    max_drawdown_recover_date: datetime
    max_drawdown_max_duration_in_days: float
    metadata: dict[str, Any]

    model_config = ConfigDict(extra="allow")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Performance):
            return False
        return self.model_dump(exclude={"metadata"}) == other.model_dump(
            exclude={"metadata"}
        )


__all__ = ["Topic", "Symbol", "Data"]

import polars as pl
from datetime import datetime
from pydantic import BaseModel, ConfigDict

from typing import Any, Self

from adrs.performance.metric import Ratio, Drawdown, Trade

from cybotrade import Topic, Symbol

from cybotrade_datasource import Data


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

    @staticmethod
    def from_df(
        df: pl.DataFrame,
        start_time: datetime,
        end_time: datetime,
        metadata: dict[str, Any] = {},
    ) -> Self:
        return Performance.model_validate(
            {
                **Ratio().compute(df),
                **Drawdown().compute(df),
                **Trade().compute(df),
                "start_time": start_time,
                "end_time": end_time,
                "metadata": metadata,
            }
        )


__all__ = ["Topic", "Symbol", "Data"]

import numpy as np
import polars as pl
from datetime import timedelta
from typing import override, cast

from .metric import Metrics


class Ratio(Metrics[dict[str, np.float64]]):
    def __init__(self, num_periods: int = 365, period: timedelta = timedelta(days=1)):
        self.num_periods = num_periods  # 365 trading days in a year (crypto)
        self.period = period  # 1 day as a base measurement

    @override
    def compute(self, df):
        # determine the interval of data
        interval = df["start_time"].diff().last()
        if not isinstance(interval, timedelta):
            raise Exception("performance_df does not have an interval in between data")

        mean = cast(np.float64, df["pnl"].mean())
        std = df["pnl"].std(ddof=1)
        neg_std = df.filter(pl.col("pnl") < 0)["pnl"].std(ddof=1)

        multiplier = np.float64(self.period / interval)
        sharpe_ratio = (
            mean / std * cast(np.float64, np.sqrt(self.num_periods * multiplier))
            if isinstance(std, float) and std != 0 and not np.isnan(std)
            else np.float64(0.0)
        )
        sortino_ratio = (
            mean / neg_std * cast(np.float64, np.sqrt(self.num_periods * multiplier))
            if isinstance(neg_std, float) and neg_std != 0 and not np.isnan(neg_std)
            else np.float64(0.0)
        )
        ar = mean * self.num_periods * multiplier
        tr = df["equity"][-1]

        total_duration = df["start_time"][-1] - df["start_time"][0]
        years = total_duration / interval / (self.num_periods * multiplier)
        cagr = np.prod(1 + df["pnl"].to_numpy()) ** (1 / years) - 1

        return {
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "min_cumu": round(cast(np.float64, df["equity"].min()), 4),
            "annualized_return": round(ar, 4),
            "total_return": round(tr, 4),
            "cagr": round(cagr, 4),
        }

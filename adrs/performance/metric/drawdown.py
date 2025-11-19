import numpy as np
import polars as pl
from typing import Any, cast, override
from datetime import datetime, timedelta, timezone

from .metric import Metrics


class Drawdown(Metrics[dict[str, Any]]):
    def __init__(self, num_periods: int = 365, period: timedelta = timedelta(days=1)):
        self.num_periods = num_periods  # 365 trading days in a year (crypto)
        self.period = period  # 1 day as a base measurement

    @override
    def compute(self, df):
        drawdown_df = df.with_columns(
            (pl.col("equity") + 1.0).alias("geo_equity"),
            (pl.col("equity") + 1.0).cum_max().alias("geo_equity_cummax"),
            pl.col("equity").cum_max().alias("equity_cummax"),
        ).with_columns((pl.col("equity") - pl.col("equity_cummax")).alias("drawdown"))

        mdd_df = drawdown_df.filter(pl.col("drawdown") == pl.col("drawdown").min())

        mdd = cast(float, mdd_df["drawdown"][0])
        mdd_cummax = cast(float, mdd_df["equity_cummax"][0])
        mdd_geo = cast(float, mdd_df["geo_equity"][0])
        mdd_geo_cummax = cast(float, mdd_df["geo_equity_cummax"][0])
        mdd_pct = (
            0.0 if mdd_cummax == 0 else (mdd_geo - mdd_geo_cummax) / mdd_geo_cummax
        )

        mdd_start_time_df = drawdown_df.join(
            drawdown_df.filter(pl.col("drawdown").shift(1).ge(0)).select(
                pl.col("start_time"), pl.col("start_time").alias("mdd_start_time")
            ),
            how="left",
            left_on="start_time",
            right_on="start_time",
        ).select(
            pl.col("start_time"),
            pl.col("drawdown"),
            pl.col("mdd_start_time").forward_fill().fill_null(pl.col("start_time")),
        )
        mdd_recover_df = drawdown_df.filter(pl.col("equity") > mdd_cummax)
        mdd_time = cast(datetime, mdd_df["start_time"][0])
        mdd_start_time = cast(
            datetime,
            mdd_start_time_df.filter(pl.col("start_time") == mdd_time)[
                "mdd_start_time"
            ][0],
        )
        mdd_recover_time = cast(
            datetime,
            df["start_time"][-1]
            if mdd_recover_df.is_empty()
            else mdd_recover_df["start_time"][0],
        )
        mdd_duration = (
            df["start_time"][-1] - mdd_time
            if mdd_recover_df.is_empty()
            else cast(datetime, mdd_recover_df["start_time"][0]) - mdd_start_time
        )

        # determine the interval of data
        interval = df["start_time"].diff().last()
        if not isinstance(interval, timedelta):
            raise Exception("performance_df does not have an interval in between data")
        mean = cast(float, df["pnl"].mean())

        calmar_ratio = (
            mean * self.num_periods * self.period / interval / abs(mdd)
            if mdd != 0 and not np.isnan(mdd)
            else np.float64(0.0)
        )

        return {
            "max_drawdown": round(mdd, 4),
            "max_drawdown_percentage": round(mdd_pct, 4),
            "max_drawdown_start_date": mdd_start_time.replace(tzinfo=timezone.utc),
            "max_drawdown_end_date": mdd_time.replace(tzinfo=timezone.utc),
            "max_drawdown_recover_date": mdd_recover_time.replace(tzinfo=timezone.utc),
            "max_drawdown_max_duration_in_days": round(
                mdd_duration.total_seconds() / (60 * 60 * 24), 4
            ),
            "calmar_ratio": round(calmar_ratio, 4),
        }

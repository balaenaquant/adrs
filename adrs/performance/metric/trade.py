import numpy as np
import polars as pl
from datetime import timedelta
from typing import override, cast

from .metric import Metrics


class Trade(Metrics[dict[str, np.float64]]):
    @override
    def compute(self, df):
        # Make the dataframe that will be used to compute the metrics
        trades_df = (
            df.select(
                pl.col("start_time"),
                pl.col("signal"),
                pl.col("prev_signal"),
                pl.col("price"),
                pl.col("trade"),
            )
            .filter(pl.col("signal") != pl.col("prev_signal"))
            .with_columns(
                pl.col("price").shift(1).alias("prev_price"),
                pl.col("start_time").diff().alias("holding_time"),
            )
            .filter(pl.col("prev_signal") != 0)
        )

        # Reusable polars expressions for filtering
        close_long_cond = ((pl.col("prev_signal") > 0) & (pl.col("signal") == 0)) | (
            (pl.col("prev_signal") > 0) & (pl.col("signal") < 0)
        )
        close_short_cond = ((pl.col("prev_signal") < 0) & (pl.col("signal") == 0)) | (
            (pl.col("prev_signal") < 0) & (pl.col("signal") > 0)
        )
        close_long_pnl_expr = pl.col("price") - pl.col("prev_price")
        close_short_pnl_expr = pl.col("prev_price") - pl.col("price")

        # Calculate number of win & lose trades
        win_trades = (
            trades_df.filter(close_long_cond & (close_long_pnl_expr > 0))["trade"]
            .abs()
            .sum()
            + trades_df.filter(close_short_cond & (close_short_pnl_expr > 0))["trade"]
            .abs()
            .sum()
        )
        lose_trades = (
            trades_df.filter(close_long_cond & (close_long_pnl_expr <= 0))["trade"]
            .abs()
            .sum()
            + trades_df.filter(close_short_cond & (close_short_pnl_expr <= 0))["trade"]
            .abs()
            .sum()
        )

        avg_holding_time = cast(timedelta | None, trades_df["holding_time"].mean())

        # Calculate win & loss streaks
        streak_df = (
            trades_df.with_columns(
                pl.when(
                    (close_long_cond & (close_long_pnl_expr > 0))
                    | (close_short_cond & (close_short_pnl_expr > 0))
                )
                .then(pl.lit("win"))
                .otherwise(pl.lit("loss"))
                .alias("result")
            )
            .with_columns(pl.col("result").rle_id().alias("streak_id"))
            .group_by(["streak_id", "result"])
            .agg(pl.count().alias("streak_length"))
        )

        win_streak = (
            streak_df.filter(pl.col("result") == "win")
            .select(pl.col("streak_length").max())
            .item()
        ) or 0
        lose_streak = (
            streak_df.filter(pl.col("result") == "loss")
            .select(pl.col("streak_length").max())
            .item()
        ) or 0

        return {
            "largest_loss": round(cast(float, df["pnl"].min()), 4),
            "num_datapoints": df.shape[0],
            "num_trades": round(df["trade"].abs().sum()),
            "avg_holding_time_in_seconds": avg_holding_time.total_seconds()
            if avg_holding_time is not None
            else 0.0,
            "long_trades": round(df.filter(pl.col("trade") > 0)["trade"].sum()),
            "short_trades": round(df.filter(pl.col("trade") < 0)["trade"].abs().sum()),
            "win_trades": round(win_trades),
            "lose_trades": round(lose_trades),
            "win_streak": win_streak,
            "lose_streak": lose_streak,
            "win_rate": round(win_trades / (win_trades + lose_trades), 4)
            if (win_trades + lose_trades) > 0
            else 0,
        }

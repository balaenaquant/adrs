import math
import polars as pl
from decimal import Decimal
from typing import Callable, cast, Any, Unpack

from adrs.types import Performance
from adrs.alpha import Alpha, AlphaBacktestArgs
from adrs.performance.metric import Metrics, Ratio, Trade, Drawdown


AlphaPerformances = dict[str, tuple[Performance, pl.DataFrame]]
AlphaWeights = dict[str, Decimal]
WeightAllocator = Callable[[AlphaPerformances], AlphaWeights]


class Portfolio:
    """
    Portfolio is a collection of Alphas.
    """

    def __init__(
        self,
        id: str,
        base_asset: str,
        alphas: list[Alpha],
        allocator: WeightAllocator,
        metrics: list[Metrics] = [Ratio(), Trade(), Drawdown()],
    ):
        # Check if there is at least one alpha
        if len(alphas) == 0:
            raise ValueError("Portfolio must have at least one alpha")

        self.base_asset = base_asset
        self.performances: AlphaPerformances = {}
        self.weights: AlphaWeights = {}
        self.id = id

        # Check if alphas id are unique
        if len(alphas) != len(set(a.id for a in alphas)):
            raise ValueError("All alphas must have unique IDs")
        self.alphas = alphas
        self.allocator = allocator
        self.metrics = metrics

    def backtest(
        self, **kwargs: Unpack[AlphaBacktestArgs]
    ) -> tuple[Performance, pl.DataFrame]:
        """Run backtest for the portfolio."""
        # Make sure alpha performances are available
        if len(self.performances) == 0:
            self.backtest_alphas(**kwargs)

        # Allocate weights if not already done
        if len(self.weights) == 0:
            self.weights = self.allocator(self.performances)

        # Make sure weights sum to 1.0
        if not math.isclose(sum(self.weights.values()), 1.0):
            raise Exception("Weights must sum to 1.0")

        merged_df: pl.DataFrame | None = None
        for alpha_id, (_, df) in self.performances.items():
            weight = self.weights[alpha_id]
            pnl_col = f"{alpha_id}_pnl"
            signal_col = f"{alpha_id}_signal"
            query = [
                pl.col("start_time"),
                pl.col("price"),
                (pl.col("pnl") * weight).alias(pnl_col),  # NOTE: Use Decimal
                (pl.col("signal").cast(pl.Float64) * weight).alias(signal_col),
            ]

            if merged_df is None:
                merged_df = df.select(query)
            else:
                merged_df = (
                    merged_df.join(
                        df.select(query),
                        on="start_time",
                        how="full",
                    )
                    .drop(["start_time_right", "price_right"])
                    .with_columns(
                        pl.col(pnl_col).forward_fill(),
                        pl.col(signal_col).forward_fill(),
                    )
                )

        merged_df = cast(pl.DataFrame, merged_df)
        performance_df = merged_df.select(
            pl.col("start_time"),
            pl.col("price"),
            pl.lit(None).alias("data").cast(pl.Float64),
            pl.sum_horizontal(
                [pl.col(c) for c in merged_df.columns if c.endswith("_pnl")]
            ).alias("pnl"),
            pl.sum_horizontal(
                [pl.col(c) for c in merged_df.columns if c.endswith("_signal")]
            ).alias("signal"),
        ).with_columns(
            pl.col("signal")
            .shift(1)
            .alias("prev_signal")
            .forward_fill()
            .fill_null(strategy="zero"),
            pl.col("price").pct_change().alias("returns").fill_null(strategy="zero"),
            pl.col("signal").diff().alias("trade").fill_null(strategy="zero"),
            pl.col("pnl").cum_sum().alias("equity").fill_null(strategy="zero"),
        )

        # Compute the metrics
        performance: dict[str, Any] = {
            "start_time": kwargs["start_time"],
            "end_time": kwargs["end_time"],
            "metadata": {},
        }
        for metric in self.metrics:
            result = metric.compute(performance_df)
            performance = {**performance, **result}

        return Performance.model_validate(performance), performance_df

    def backtest_alphas(self, **kwargs: Unpack[AlphaBacktestArgs]):
        """Run backtest for each alpha in the portfolio."""
        for alpha in self.alphas:
            self.performances[alpha.id] = alpha.backtest(**kwargs)

    def compile_signals(self, alpha_signals: dict[str, Decimal]) -> Decimal:
        signal_strength = Decimal("0")

        for alpha in self.alphas:
            if alpha.id not in alpha_signals.keys():
                continue

            weight = self.weights[alpha.id]
            signal_strength += weight * alpha_signals[alpha.id]

        return signal_strength

    def alpha_ids(self):
        return [alpha.id for alpha in self.alphas]

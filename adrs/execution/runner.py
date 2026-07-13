import asyncio
from datetime import timedelta

import polars as pl

from adrs.alpha import Alpha
from adrs.portfolio import Portfolio
from adrs.data import DataLoader, MetricStream, DatasourceStream
from adrs.execution.executor import PortfolioExecutor, AlphaExecutor
from adrs.types import Topic


def _remap_portfolio_ids(portfolio: Portfolio, alpha_id_map: dict[str, str]) -> None:
    """Rewrite the portfolio's alpha ids in place using alpha_id_map.

    run_portfolio renames the Alpha objects' ids (publish side) but the
    portfolio's frames are keyed by the original ids. Without this, the live
    signal an alpha publishes (under its renamed id) never matches the
    portfolio's lookup and any "is this my alpha?" guard rejects every signal.

    metadata_df/weight_df key by a `custom_id` column; signal_df keys each alpha
    by a column name. Unmapped values are left unchanged."""
    portfolio.metadata_df = portfolio.metadata_df.with_columns(
        pl.col("custom_id").replace(alpha_id_map)
    )
    portfolio.weight_df = portfolio.weight_df.with_columns(
        pl.col("custom_id").replace(alpha_id_map)
    )
    portfolio.signal_df = portfolio.signal_df.rename(
        {
            old: new
            for old, new in alpha_id_map.items()
            if old in portfolio.signal_df.columns
        }
    )


def _finest_interval(alphas: list[Alpha]) -> timedelta | None:
    """Smallest signal interval across all alphas' data topics. Used to default
    the signal-freshness window so a daily portfolio isn't rejected by a flat
    2h rule. Returns None if no topic declares an interval."""
    intervals: list[timedelta] = []
    for alpha in alphas:
        for di in getattr(alpha, "data_infos", []):
            try:
                iv = Topic.from_str(di.topic).interval()
            except Exception:
                iv = None
            if iv is not None:
                intervals.append(iv)
    return min(intervals) if intervals else None


async def run_portfolio(
    portfolio: Portfolio,
    alphas: list[Alpha],
    dataloader: DataLoader,
    metric_stream: MetricStream,
    datasource_stream: DatasourceStream,
    run_alphas: bool = True,
    alpha_id_map: dict[str, str] = {},
    max_signal_age: timedelta | None = None,
    signal_namespace: str | None = None,
    insert_prefix: str = "public_ts",
    resync_interval: timedelta | None = None,
):
    for alpha in alphas:
        if alpha.id in alpha_id_map:
            alpha.id = alpha_id_map[alpha.id]

    # The rename above only touches the Alpha objects (publish side); remap the
    # portfolio's frames with the same map too (see _remap_portfolio_ids).
    if alpha_id_map:
        _remap_portfolio_ids(portfolio, alpha_id_map)

    # Freshness window for PortfolioExecutor's staleness guard. When unset,
    # default to twice the finest signal interval (daily portfolio → 2d) but
    # never tighter than the legacy 2h — so intraday strategies are unaffected
    # and slow strategies aren't spuriously rejected.
    if max_signal_age is None:
        finest = _finest_interval(alphas)
        max_signal_age = (
            max(timedelta(hours=2), finest * 2)
            if finest is not None
            else timedelta(hours=2)
        )

    portfolio_executor = PortfolioExecutor(
        portfolio=portfolio,
        metric_stream=metric_stream,
        max_signal_age=max_signal_age,
        signal_namespace=signal_namespace,
        insert_prefix=insert_prefix,
    )

    if run_alphas:
        alpha_executor = AlphaExecutor(
            alphas=alphas,
            dataloader=dataloader,
            metric_stream=metric_stream,
            datasource_stream=datasource_stream,
            signal_namespace=signal_namespace,
            insert_prefix=insert_prefix,
            resync_interval=resync_interval,
        )
        await asyncio.gather(alpha_executor.start(), portfolio_executor.start())
    else:
        await portfolio_executor.start()

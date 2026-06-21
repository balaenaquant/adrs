import asyncio
from datetime import timedelta

from aion import Trigger

from adrs.alpha import Alpha
from adrs.portfolio import Portfolio
from adrs.data import DataLoader, MetricStream, DatasourceStream
from adrs.execution.executor import PortfolioExecutor, AlphaExecutor
from adrs.types import Topic


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
    health_check_trigger: Trigger = Trigger.Cron("*/1 * * * *"),
    max_signal_age: timedelta | None = None,
):
    for alpha in alphas:
        if alpha.id in alpha_id_map:
            alpha.id = alpha_id_map[alpha.id]

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
    )

    if run_alphas:
        alpha_executor = AlphaExecutor(
            alphas=alphas,
            dataloader=dataloader,
            metric_stream=metric_stream,
            datasource_stream=datasource_stream,
            health_check_trigger=health_check_trigger,
        )
        await asyncio.gather(alpha_executor.start(), portfolio_executor.start())
    else:
        await portfolio_executor.start()

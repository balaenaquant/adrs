import json
import asyncio

from aion import Trigger

from adrs.alpha import Alpha
from adrs.portfolio import Portfolio
from adrs.data import DataLoader, MetricStream, DatasourceStream
from adrs.execution.executor import PortfolioExecutor, AlphaExecutor


async def run_portfolio(
    portfolio: Portfolio,
    alphas: list[Alpha],
    datasource_api_key: str,
    metric_stream: MetricStream,
    datasource_stream: DatasourceStream,
    run_alphas: bool = True,
    alpha_id_map: dict[str, str] = {},
    health_check_trigger: Trigger = Trigger.Cron("*/1 * * * *"),
):
    for alpha in alphas:
        if alpha.id in alpha_id_map:
            alpha.id = alpha_id_map[alpha.id]

    portfolio_executor = PortfolioExecutor(
        portfolio=portfolio,
        metric_stream=metric_stream,
    )

    if run_alphas:
        credentials = json.load(open("credentials.json"))
        dataloader = DataLoader(
            data_dir="outdir",
            credentials=credentials,
        )
        alpha_executor = AlphaExecutor(
            alphas=alphas,
            dataloader=dataloader,
            datasource_api_key=datasource_api_key,
            metric_stream=metric_stream,
            datasource_stream=datasource_stream,
            health_check_trigger=health_check_trigger,
        )
        await asyncio.gather(alpha_executor.start(), portfolio_executor.start())
    else:
        await portfolio_executor.start()

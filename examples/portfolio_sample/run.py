import os
import json
import asyncio
import logging

from adrs.logging import (
    setup_logger,
    make_logging_timed_rotating_file_handler,
    make_colorlog_stream_handler,
)

from nats_client import NATSClient
from adrs.execution import run_portfolio
from adrs.io.stream import PublicDatasourceStream, PublicMetricStream

from examples.portfolio_sample.portfolio import setup_portfolio

logger: logging.Logger = logging.getLogger(__name__)


def getenv(name: str) -> str:
    env = os.getenv(name)
    if env is None:
        raise ValueError(f"{name} is not present in environment")
    return env


async def main():
    setup_logger(
        log_level=logging.INFO,
        handlers=[
            make_colorlog_stream_handler(),
            make_logging_timed_rotating_file_handler(
                filename="portfolio.logs", backupCount=24
            ),
        ],
    )

    portfolio, alphas = await setup_portfolio()

    metric_nats = NATSClient(nats_url=getenv("NATS_URL"))
    ms = PublicMetricStream(metric_nats)

    ds = PublicDatasourceStream()

    await run_portfolio(
        portfolio,
        alphas=alphas,
        datasource_api_key=json.load(open("credentials.json"))["cybotrade_api_key"],
        metric_stream=ms,
        datasource_stream=ds,
    )


asyncio.run(main())

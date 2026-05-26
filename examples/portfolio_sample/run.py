import os
import json
import asyncio
import logging

from adrs.logging import (
    setup_logger,
    make_logging_timed_rotating_file_handler,
    make_colorlog_stream_handler,
)

from adrs.data.connector import connect_nats
from adrs.execution.run import run_portfolio
from adrs.io.stream import PublicDatasourceStream, PublicMetricStream

from examples.portfolio_sample.portfolio import setup_portfolio

logger: logging.Logger = logging.getLogger(__name__)


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

    ms = PublicMetricStream(
        await connect_nats(
            url=os.getenv("NATS_URL", "nats://localhost:4222"),
            user=os.getenv("NATS_USER", ""),
            password=os.getenv("NATS_PASSWORD", ""),
        )
    )
    await ms.setup()

    ds = PublicDatasourceStream()

    await run_portfolio(
        portfolio,
        alphas=alphas,
        datasource_api_key=json.load(open("credentials.json"))["cybotrade_api_key"],
        metric_stream=ms,
        datasource_stream=ds,
    )


asyncio.run(main())

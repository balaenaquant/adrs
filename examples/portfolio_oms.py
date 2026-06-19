import os
import json
import asyncio
import logging
import argparse
import importlib

from adrs.data import MetricStream, DatasourceStream
from adrs.io.stream import (
    PublicMetricStream,
    PublicNatsDatasourceStream,
)
from adrs.logging import (
    setup_logger,
    make_logging_timed_rotating_file_handler,
    make_colorlog_stream_handler,
)
from adrs.execution import run_portfolio

from aion import Scheduler
from nats_client import NATSClient

from adrs.oms.oms import OMS
from adrs.oms.config import FileConfigManager
from adrs.oms.rate_limit.rate_limiter import BybitRateLimiter

logger: logging.Logger = logging.getLogger(__name__)


def getenv(name: str) -> str:
    env = os.getenv(name)
    if env is None:
        raise ValueError(f"{name} is not present in environment")
    return env


async def main():
    parser = argparse.ArgumentParser(prog="new_run")
    parser.add_argument(
        "module",
        help="Python module with setup_portfolio function",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        help="Path to the metadata config JSON file",
    )
    parser.add_argument(
        "-c",
        "--cache-dir",
        default="output",
        help="Directory for cached market data (default: output)",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        default="portfolio_oms.log",
        help="Log file path",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="config.log",
        help="The configuration file to read from",
    )
    args = parser.parse_args()

    setup_logger(
        log_level=logging.INFO,
        handlers=[
            make_colorlog_stream_handler(),
            make_logging_timed_rotating_file_handler(
                filename=args.log_file, backupCount=24
            ),
        ],
    )

    setup_portfolio = importlib.import_module(args.module).setup_portfolio
    portfolio, alphas = await setup_portfolio(
        config_path=args.metadata,
        cache_dir=args.cache_dir,
    )

    metric_nats = NATSClient(grpc_addr=os.environ.get("BQ_AEGIS_NATS_GRPC_ADDR"))
    ms: MetricStream = PublicMetricStream(nats=metric_nats)
    await ms.init()

    flow_nats = NATSClient(grpc_addr=os.environ.get("BQ_FLOW_NATS_GRPC_ADDR"))
    ds: DatasourceStream = PublicNatsDatasourceStream(flow_nats=flow_nats)

    scheduler = Scheduler()

    config = FileConfigManager(args.file)
    await config.setup()
    # rate_limiter = BinanceRateLimiter(config=config)
    rate_limiter = BybitRateLimiter(config=config)
    await rate_limiter.init()
    oms = OMS(
        config=config,
        metric_stream=ms,
        rate_limiter=rate_limiter,
    )

    await asyncio.gather(
        scheduler.start(),
        run_portfolio(
            portfolio,
            alphas=alphas,
            datasource_api_key=json.load(open("credentials.json"))["cybotrade_api_key"],
            metric_stream=ms,
            datasource_stream=ds,
        ),
        oms.run(),
    )


asyncio.run(main())

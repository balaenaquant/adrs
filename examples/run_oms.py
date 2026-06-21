import os
import asyncio
import logging
import argparse

from adrs.logging import (
    setup_logger,
    make_logging_timed_rotating_file_handler,
    make_colorlog_stream_handler,
)

from nats_client import NATSClient
from adrs.io.stream import PublicMetricStream

from adrs.oms.oms import OMS
from adrs.oms.config import FileConfigManager
from adrs.oms.rate_limit.rate_limiter import BybitRateLimiter


def getenv(name: str) -> str:
    env = os.getenv(name)
    if env is None:
        raise ValueError(f"{name} is not present in environment")
    return env


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_oms",
        description="Run an order management system",
    )
    parser.add_argument(
        "-l",
        "--log-file",
        default="portfolio.log",
        help="The file that running logs store to",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="config.log",
        help="The configuration file to read from",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    setup_logger(
        log_level=logging.INFO,
        handlers=[
            make_colorlog_stream_handler(),
            make_logging_timed_rotating_file_handler(
                filename=args.log_file, backupCount=24
            ),
        ],
    )

    config = FileConfigManager(args.file)
    await config.setup()
    metric_nats = NATSClient(grpc_addr=os.environ.get("BQ_AEGIS_NATS_GRPC_ADDR"))

    ms = PublicMetricStream(nats=metric_nats)
    await ms.init()
    # rate_limiter = BinanceRateLimiter(config=config)
    rate_limiter = BybitRateLimiter(config=config)
    await rate_limiter.init()
    oms = OMS(
        config=config,
        metric_stream=ms,
        rate_limiter=rate_limiter,
    )
    await oms.run()


if __name__ == "__main__":
    asyncio.run(main())

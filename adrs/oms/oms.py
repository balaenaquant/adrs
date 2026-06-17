import os
import json
import copy
import asyncio
import logging
import signal

from typing import Dict
from decimal import Decimal
from pydantic import BaseModel
from datetime import datetime, timezone

from aion import Scheduler, Trigger
from nats_client import Msg
from adrs.data import MetricStream, MetricBuilder
from cybotrade import Symbol
from cybotrade.models import Position, OrderSide, OrderStatus

from oms.config import ConfigManager
from oms.ops.order_executer import OrderExecutor
from oms.ops.order_pool import CancelBacklogs
from oms.position import PositionManager
from oms.ops.order_placement_manager import OrderPlacementManager
from oms.rate_limit.rate_limiter import RateLimiter
from oms.rate_limit.exchange_limit_profiles import Endpoints

logger = logging.getLogger(__name__)


class PortfolioSignal(BaseModel):
    assets: Dict[str, Decimal]
    timestamp: int


def generate_cron(total_seconds: int):
    if total_seconds < 1:
        return "* * * * * *"

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{seconds} {minutes} */{hours} * * *"
    elif minutes > 0:
        return f"{seconds} */{minutes} * * * *"
    else:
        return f"*/{seconds} * * * * *"


def getenv(name: str) -> str:
    env = os.getenv(name)
    if env is None:
        raise ValueError(f"{name} is not present in environment")
    return env


class OMS:
    # Subclasses override this to swap in a custom execution strategy,
    executor_cls: type[OrderExecutor] = OrderExecutor

    def __init__(
        self,
        config: ConfigManager,
        metric_stream: MetricStream,
        rate_limiter: RateLimiter,
    ):
        super().__init__()
        self.config = config
        self.metric_stream = metric_stream
        self.metric_builder = MetricBuilder(self.metric_stream)
        self.position = PositionManager(
            config=config,
            rate_limiter=rate_limiter,
        )
        self.opm = OrderPlacementManager(
            position=self.position,
            config=self.config,
            rate_limiter=rate_limiter,
            executor_cls=self.executor_cls,
        )
        self.scheduler = Scheduler()
        self.previous_signal: PortfolioSignal | None = None
        self.latest_signal: PortfolioSignal | None = None
        self.rate_limiter = rate_limiter

    async def init(self):
        """Initialise the OMS state when first started."""
        logger.info("[INIT] Initilalizing position values")

        # LATEST SIGNAL
        self.latest_signal = await self.get_latest_signal(
            self.config.config.portfolio_id,
        )
        logger.info(f"Latest signal at startup {self.latest_signal}")

        # ORDER POOL
        await self.opm.order_pools.resync_order_pool()

        # POSITION
        await self.position.update_exchange()
        await self.position.update_pending()
        for s in self.config.config.base_asset_to_symbol_table.values():
            symbol = Symbol(s)
            if symbol not in self.position.pending.keys():
                self.position.pending[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal("0"),
                    entry_price=Decimal("0"),
                    updated_time=datetime.now(tz=timezone.utc),
                )
            if symbol not in self.position.exchange.keys():
                self.position.exchange[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal("0"),
                    entry_price=Decimal("0"),
                    updated_time=datetime.now(tz=timezone.utc),
                )

            # Init using desired and exchange
            quantity = (
                self.position.exchange[symbol].quantity
                + self.position.pending[symbol].quantity
            )
            self.position.desired[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=Decimal("0"),
                updated_time=datetime.now(tz=timezone.utc),
            )

    async def get_latest_signal(self, portfolio_id: str) -> PortfolioSignal | None:
        """
        Cold-start hook: fetch the last known signal so the OMS does not start
        blind. The base has no signal store, so it starts without one and waits
        for the first live signal; subclasses override to read their own store.
        """
        return None

    def _setup_signals(self):
        loop = asyncio.get_event_loop()

        loop.add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(self._handle_shutdown())
        )

    async def _handle_shutdown(self):
        """To close all pending orders on shutdown"""
        logger.info("Shutdown signal received. Cancelling orders...")

        cancel_tasks = [
            self.opm.executor.cancel_single_order(
                Symbol(order.symbol), order.client_order_id
            )
            for order in self.opm.order_pools.order_pool.values()
        ]

        if not cancel_tasks:
            exit(0)

        cancel_results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

        cancel_retries = []
        for result in cancel_results:
            if isinstance(result, Exception):
                logger.error(f"Failed to cancel order: {result}")
            if isinstance(result, CancelBacklogs):
                cancel_retries.append(result)

        if not cancel_retries:
            logger.info("All orders cancelled successfully.")
            return

        logger.warning(f"Retrying {len(cancel_retries)} cancellations in 60s...")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.error("Shutdown forced during retry wait.")
            return

        retry_tasks = [
            self.opm.executor.cancel_single_order(
                Symbol(retry.symbol), retry.client_order_id
            )
            for retry in cancel_retries
        ]

        if retry_tasks:
            await asyncio.gather(*retry_tasks, return_exceptions=True)
            logger.info("Retry attempts finished.")

        exit(0)

    async def on_refresh_config(self):
        """To refresh and update config if there are any changes made during runtime"""
        old_config = copy.deepcopy(self.config.config)
        await self.config.refresh()

        # Restart with a new exchange events when credentials have changed
        if old_config.credentials != self.config.config.credentials:
            logger.info(
                "Detected credentials update, refreshing exchange event handler..."
            )
            self.exchange_events_task.cancel()
            self.exchange_event = self.config.config.credentials.to_exchange_event()
            self.exchange_event.on_event = self.opm.on_exchange_event
            self.exchange_events_task = asyncio.create_task(self.exchange_event.start())

        if (
            old_config.base_asset_to_symbol_table
            != self.config.config.base_asset_to_symbol_table
        ):
            endpoint = Endpoints.GET_SYMBOL_INFO
            try:
                async with self.rate_limiter.guard(endpoint=endpoint):
                    await self.config.update_symbol_info()
            except Exception as e:
                logger.warning(f"update symbol info failed due to {e}")

            await self.init()

    async def on_portfolio_signal(self, msg: Msg):
        """To store the latest signal from portfolio server"""
        try:
            payload = json.loads(msg.data.decode())
            portfolio_signal = PortfolioSignal(
                assets=payload["assets"],
                timestamp=int(payload["timestamp"]),
            )
            logger.info(f"Portfolio signal retrieved {portfolio_signal}")

            if sum(abs(p) for p in portfolio_signal.assets.values()) > Decimal("1"):
                logger.warning(
                    "[ON_PROCESS_LATEST_SIGNAL] Current sum of positions in latest signal is more than 1"
                )
                return

            self.latest_signal = portfolio_signal
        except Exception as e:
            logger.error(f"Failed to process portfolio signal due to {e}")

    async def on_process_latest_signal(self):
        """
        To calculate and upsert desired position based on time window
        """
        if not self.latest_signal:
            logger.warning("[ON_PROCESS_LATEST_SIGNAL] There is no signal to act on")
            return

        market_quotes: dict[Symbol, Decimal] = {}
        PRECISION_4 = Decimal("0.0001")

        # Process latest signal into Desired Position
        for asset, weightage in self.latest_signal.assets.items():
            # weightage remained the same, no updates needed
            symbol = Symbol(self.config.config.base_asset_to_symbol_table[asset])
            # .get: a newly added asset has no entry in the previous signal
            previous_weightage = (
                self.previous_signal.assets.get(asset) if self.previous_signal else None
            )
            if previous_weightage is not None and previous_weightage.quantize(
                exp=PRECISION_4
            ) == weightage.quantize(exp=PRECISION_4):
                logger.info(
                    f"There is no significant change in signal weights for {symbol}, it is ignored"
                )
                continue

            # clear out any backlog as new signal will account for any unmade orders/cancels
            async with self.opm.order_pools.get_order_backlog() as order_backlog:
                order_backlog.clear()

            if symbol not in market_quotes.keys():
                try:
                    async with self.rate_limiter.guard(
                        endpoint=Endpoints.GET_ORDERBOOK_SNAPSHOT
                    ):
                        market_quotes[
                            symbol
                        ] = await self.config.exchange.get_current_price(symbol=symbol)
                except Exception as e:
                    logger.warning(f"Failed to process latest signal due to {e}")
                    return

            quantity = self.position.compute_base_quantity(
                price=market_quotes[symbol], weightage=weightage
            )
            self.position.desired[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=market_quotes[symbol],
                updated_time=datetime.now(tz=timezone.utc),
            )
            desired_position = self.position.desired[symbol]
            logger.debug(
                f"[ON_PROCESS_LATEST_SIGNAL] {self.config.config.portfolio_id} wants {desired_position.quantity} of {desired_position.symbol} at the price of {desired_position.entry_price}"
            )

        self.previous_signal = self.latest_signal

    async def on_aegis_update(self):
        """
        To upsert latest equity value from exchange to aegis``
        """
        logger.info("[ON_AEGIS_UPDATE]")

        async def _check_and_sync_trade(
            sem,
            oms_id: str,
            package_id: str,
            record: tuple[Symbol, str, Decimal, datetime],
        ):
            """
            Worker function.
            Returns: (package_id, record) IF it is fully filled and synced or 0-filled.
            Returns: None IF it is pending or failed.
            """
            async with sem:
                symbol, client_order_id, start_price, start_time = (
                    record[0],
                    record[1],
                    record[2],
                    record[3],
                )
                asset = parts[0] if symbol and (parts := symbol.split()) else ""

                try:
                    async with self.rate_limiter.guard(
                        endpoint=Endpoints.GET_ORDER_DETAILS
                    ):
                        result = (
                            await self.config.exchange.get_order_details_from_history(
                                symbol=symbol,
                                client_order_id=client_order_id,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch {client_order_id}: {e}")
                    return None  # Failed fetch, keep in pool

                if not result:
                    logger.warning(
                        f"[ON_AEGIS_UPDATE] {client_order_id} NOT FOUND IN EXCHANGE HISTORY"
                    )
                    return None  # Not found, keep in pool

                if result.filled_size == Decimal("0"):
                    if result.status in (
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                    ):
                        # Terminal with nothing filled, no trade will ever come
                        return (package_id, record)
                    # Still live with no fills yet; dropping it now would lose
                    # the trade record when it fills later
                    return None

                match result.side:
                    case OrderSide.BUY:
                        order_side = 1
                    case OrderSide.SELL:
                        order_side = -1
                    case OrderSide.NONE:
                        logger.error("[ON_AEGIS_UPDATE] Orderside shouldn't be NONE")
                        order_side = 0

                try:
                    await self.metric_builder.create_trade(
                        oms_id=oms_id,
                        package_id=package_id,
                        client_order_id=client_order_id,
                        asset=asset,
                        symbol=str(symbol),
                        exchange=self.config.exchange.exchange(),
                        start_quantity=str(order_side * result.size),
                        executed_quantity=str(order_side * result.filled_size),
                        executed_price=str(result.price),
                        executed_time=int(
                            result.updated_time.astimezone(tz=timezone.utc).timestamp()
                            * 1_000_000
                        )
                        * 1_000,
                        start_time=int(
                            start_time.astimezone(tz=timezone.utc).timestamp()
                            * 1_000_000
                        )
                        * 1_000,
                        start_price=str(start_price),
                    )

                    return (package_id, record)

                except Exception as e:
                    logger.error(f"Failed to save {client_order_id} to Aegis: {e}")
                    return None  # DB Write failed, keep in pool to retry next time

        balance_endpoint = Endpoints.GET_WALLET_BALANCE
        try:
            oms_id = self.config.config.oms_id
            logger.debug("Inserting trades to aegis")

            sem = asyncio.Semaphore(2)
            tasks = []

            for package_id, records in list(self.opm.order_pools.order_records.items()):
                for record in records:
                    tasks.append(_check_and_sync_trade(sem, oms_id, package_id, record))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, BaseException):
                        logger.error(f"Trade sync error: {res}")
                        continue
                    if res:
                        filled_pkg_id, filled_record = res
                        try:
                            self.opm.order_pools.order_records[filled_pkg_id].remove(
                                filled_record
                            )
                            if not self.opm.order_pools.order_records[filled_pkg_id]:
                                del self.opm.order_pools.order_records[filled_pkg_id]
                        except (ValueError, KeyError):
                            pass

            # EQUITY UPDATE
            async with self.rate_limiter.guard(endpoint=balance_endpoint):
                logger.debug("Inserting equity to aegis")
                balance = await self.config.exchange.get_wallet_balance()
                await self.metric_builder.create_equity(
                    oms_id=oms_id, equity=str(balance.margin_balance)
                )

            # POSITION UPDATE
            logger.debug("Inserting position to aegis")
            await self.position.update_exchange()
            for position in self.position.exchange.values():
                asset = parts[0] if (parts := position.symbol.split()) else ""
                await self.metric_builder.create_position(
                    oms_id=oms_id,
                    asset=asset,
                    symbol=str(position.symbol),
                    exchange=self.config.exchange.exchange(),
                    quantity=str(position.quantity),
                    price=str(position.entry_price),
                    updated_time=int(
                        position.updated_time.astimezone(tz=timezone.utc).timestamp()
                        * 1_000_000
                    )
                    * 1_000,
                )
        except Exception as e:
            logger.warning(f"Failed to upsert to aegis due to {e}")

    async def run(self):
        try:
            await self.init()
            await self.metric_stream.subscribe(
                f"portfolio_signal.{self.config.config.portfolio_id}",
                callback=self.on_portfolio_signal,
            )

            await self.scheduler.schedule(
                id="on_refresh_config",
                handler=self.on_refresh_config,
                trigger=Trigger.Cron("*/2 * * * * *"),  # every 2 seconds
            )

            await self.scheduler.schedule(
                id="on_aegis_update",
                handler=self.on_aegis_update,
                trigger=Trigger.Cron("*/1 * * * *"),  # every 1 minute
            )

            await self.scheduler.schedule(
                id="on_process_latest_signal",
                handler=self.on_process_latest_signal,
                trigger=Trigger.Cron("*/5 * * * * *"),  # every 15 seconds
            )

            await self.scheduler.schedule(
                id="on_order_placement",
                handler=self.opm.on_order_placement,
                trigger=Trigger.Cron(
                    generate_cron(self.config.config.order_placement_interval)
                ),  # every 15 seconds
            )

            await self.scheduler.schedule(
                id="on_retry_backlog",
                handler=self.opm.on_retry_backlog,
                trigger=Trigger.Cron("*/2 * * * * *"),  # 2 seconds
            )

            await self.scheduler.schedule(
                id="on_order_expiry_check",
                handler=self.opm.on_order_expiry_check,
                trigger=Trigger.Cron(generate_cron(self.config.config.expiry_check)),
            )

            await self.scheduler.schedule(
                id="on_resync_time",
                handler=self.rate_limiter.on_resync_time,
                trigger=Trigger.Cron("0 0 * * *"),  # every day
            )

            self.exchange_event = self.config.config.credentials.to_exchange_event()
            self.exchange_event.on_event = self.opm.on_exchange_event
            self.exchange_events_task = asyncio.create_task(self.exchange_event.start())
            await self.scheduler.start()
        finally:
            await self._handle_shutdown()

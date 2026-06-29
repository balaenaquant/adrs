from abc import abstractmethod
import json

from typing import TYPE_CHECKING

from cybotrade import Symbol
from pydantic import BaseModel, ValidationError
from decimal import Decimal

from cybotrade.models import Exchange, SymbolInfo
from cybotrade.io import ExchangeClient, ExchangeEvent
from cybotrade.bybit import BybitLinearClient, BybitPrivateWS
from cybotrade.binance import BinanceLinearClient, BinancePrivateWS
from cybotrade.kucoin import KucoinLinearClient, KucoinPrivateWS
from cybotrade.edgex import EdgeXClient, EdgeXPrivateWS

from adrs.oms.logging import PrefixedLogger
from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.error_policy import (
    ExchangeErrorPolicy,
    BybitErrorPolicy,
    BinanceErrorPolicy,
    DefaultErrorPolicy,
)

if TYPE_CHECKING:
    from adrs.oms.rate_limit.rate_limiter import RateLimiter


class Credentials(BaseModel):
    exchange: Exchange
    api_key: str
    api_secret: str
    api_passphrase: str | None = None
    testnet: bool = False
    demo: bool = False

    def to_exchange_client(self) -> ExchangeClient:
        match self.exchange:
            case Exchange.BYBIT_LINEAR:
                return BybitLinearClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                    demo=self.demo,
                )
            case Exchange.BINANCE_LINEAR:
                return BinanceLinearClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                )
            case Exchange.KUCOIN_LINEAR:
                if self.api_passphrase is None:
                    raise ValueError("'api_passphrase' is required for Kucoin")

                return KucoinLinearClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                    sandbox=self.testnet,
                )
            case Exchange.EDGEX:
                return EdgeXClient(
                    account_id=self.api_key,
                    private_key=self.api_secret,
                )
            case _:
                raise Exception(f"Unsupported exchange {self.exchange}")

    def to_exchange_event(self) -> ExchangeEvent:
        match self.exchange:
            case Exchange.BYBIT_LINEAR:
                return BybitPrivateWS(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                    demo=self.demo,
                    topics=["order"],
                )
            case Exchange.BINANCE_LINEAR:
                return BinancePrivateWS(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=self.testnet,
                )
            case Exchange.KUCOIN_LINEAR:
                if self.api_passphrase is None:
                    raise ValueError("'api_passphrase' is required for Kucoin")

                return KucoinPrivateWS(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                    topics=["/contractMarket/tradeOrders"],
                )
            case Exchange.EDGEX:
                return EdgeXPrivateWS(
                    account_id=self.api_key,
                    private_key=self.api_secret,
                )
            case _:
                raise Exception(f"Unsupported exchange {self.exchange}")

    def to_exchange_topic(self) -> str:
        match self.exchange:
            case Exchange.BYBIT_LINEAR:
                return "bybit-linear"
            case Exchange.BINANCE_LINEAR:
                return "binance-linear"
            # TODO Get actual values
            case Exchange.KUCOIN_LINEAR:
                return "kucoin-linear"
            case Exchange.EDGEX:
                return "edgex"
            case _:
                raise Exception(f"Unsupported exchange {self.exchange}")

    def to_error_policy(self) -> ExchangeErrorPolicy:
        match self.exchange:
            case Exchange.BYBIT_LINEAR:
                return BybitErrorPolicy()
            case Exchange.BINANCE_LINEAR:
                return BinanceErrorPolicy()
            case _:
                # No exchange-specific policy: keep legacy retry-everything behaviour
                return DefaultErrorPolicy()


class Config(BaseModel):
    oms_id: str
    portfolio_id: str

    initial_balance: Decimal
    leverage: Decimal
    base_asset_to_symbol_table: dict[str, str]

    credentials: Credentials

    order_placement_interval: int
    expiry_check: int
    replace_best_bid_ask_time: int
    max_limit_replace_interval: int
    min_limit_replace_interval: int
    max_retries_allowed: int
    soft_limit_percent: Decimal


class ConfigManager:
    # Subclasses override to validate against an extended config schema,
    # e.g. class BqFileConfigManager(FileConfigManager): config_cls = BqConfig
    config_cls: type[Config] = Config

    config: Config
    exchange: ExchangeClient
    symbol_infos: dict[Symbol, SymbolInfo]

    def __init__(self, prefix: str = "[ConfigManager]"):
        self.logger: PrefixedLogger = PrefixedLogger(prefix=prefix, name=__name__)
        self.symbol_infos = {}

    @abstractmethod
    async def setup(self, *args, **kwars):
        pass

    @abstractmethod
    async def load(self) -> Config:
        pass

    async def refresh(self):
        self.config = await self.load()
        self.exchange = self.config.credentials.to_exchange_client()

    async def update_symbol_info(self, rate_limiter: "RateLimiter | None" = None):
        for symbol in self.config.base_asset_to_symbol_table.values():
            if rate_limiter is not None:
                async with rate_limiter.guard(endpoint=Endpoints.GET_SYMBOL_INFO):
                    self.symbol_infos[
                        Symbol(symbol)
                    ] = await self.exchange.get_symbol_info(symbol=Symbol(symbol))
            else:  # bootstrap (setup) — no limiter yet, single startup burst
                self.symbol_infos[Symbol(symbol)] = await self.exchange.get_symbol_info(
                    symbol=Symbol(symbol)
                )


class FileConfigManager(ConfigManager):
    def __init__(
        self,
        path: str,
    ):
        super().__init__(prefix="[FileConfigManager]")
        self.path = path

    async def setup(self):
        await self.refresh()
        await self.update_symbol_info()

    async def load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                return self.config_cls.model_validate(obj=data)
        except FileNotFoundError:
            self.logger.error("Config file does not exist")
            raise
        except json.JSONDecodeError:
            self.logger.error("Config file contains invalid json data")
            raise
        except ValidationError as e:
            self.logger.error(f"'{self.path}' does not match the Config schema: {e}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise


class NATSConfigManager(ConfigManager):
    def __init__(
        self,
        url: str,
        bucket: str,
        key: str,
        user: str | None = None,
        password: str | None = None,
    ):
        super().__init__(prefix="[NATSConfigManager]")
        self.url = url
        self.bucket = bucket
        self.key = key

    async def setup(self):
        await self.refresh()

    async def load(self):
        raise NotImplementedError(
            "NATSConfigManager.load is not implemented. "
            "Subclass NATSConfigManager and override load() to fetch config from NATS."
        )

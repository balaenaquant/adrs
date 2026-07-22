"""ConfigManager.refresh() — REST client lifecycle.

Regression coverage: refresh() runs on a 2s cron and used to rebuild the
exchange client unconditionally, churning sessions and stranding every
component that captured the previous client at construction.
"""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

from cybotrade.models import Exchange

from adrs.oms.config import Config, ConfigManager, Credentials


def _config(api_key="key-1") -> Config:
    return Config(
        oms_id="oms-1",
        portfolio_id="p1",
        initial_balance=Decimal("1000"),
        leverage=Decimal("1"),
        base_asset_to_symbol_table={"BTC": "BTCUSDT"},
        credentials=Credentials(
            exchange=Exchange.BYBIT_LINEAR,
            api_key=api_key,
            api_secret="secret",
        ),
        order_placement_interval=15,
        expiry_check=15,
        replace_best_bid_ask_time=120,
        max_limit_replace_interval=15,
        min_limit_replace_interval=5,
        max_retries_allowed=5,
        soft_limit_percent=Decimal("0.8"),
    )


class _Manager(ConfigManager):
    def __init__(self, configs):
        super().__init__()
        self._configs = list(configs)

    async def setup(self):
        pass

    async def load(self):
        return self._configs.pop(0)


def test_refresh_keeps_client_when_credentials_unchanged():
    mgr = _Manager([_config(), _config()])
    with patch.object(
        Credentials, "to_exchange_client", return_value=MagicMock()
    ) as make_client:
        asyncio.run(mgr.refresh())
        first = mgr.exchange
        asyncio.run(mgr.refresh())

    assert mgr.exchange is first
    assert make_client.call_count == 1


def test_refresh_rebuilds_client_when_credentials_change():
    mgr = _Manager([_config("key-1"), _config("key-2")])
    clients = [MagicMock(), MagicMock()]
    with patch.object(Credentials, "to_exchange_client", side_effect=clients):
        asyncio.run(mgr.refresh())
        asyncio.run(mgr.refresh())

    assert mgr.exchange is clients[1]

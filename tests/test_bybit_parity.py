"""
Bybit parity wiring.

Two things adrs owns that the cybotrade adapters cannot do for themselves:
subscribing the `position` topic (Bybit only sends topics you ask for, unlike
Binance's listenKey which delivers everything), and choosing which public feed
adapter to construct.
"""

import pytest
from cybotrade.binance import BinancePublicWS
from cybotrade.bybit import BybitPrivateWS, BybitPublicWS

from adrs.oms.config import Credentials, Exchange


def _creds(exchange: Exchange, **kw) -> Credentials:
    return Credentials(
        exchange=exchange,
        api_key="k",
        api_secret="s",
        api_passphrase="p",
        testnet=kw.get("testnet", False),
        demo=kw.get("demo", False),
    )


def test_bybit_subscribes_the_position_topic():
    """
    Without this the cybotrade position parser is dead code: Bybit only pushes
    topics that were explicitly subscribed.
    """
    ws = _creds(Exchange.BYBIT_LINEAR).to_exchange_event()
    assert isinstance(ws, BybitPrivateWS)
    assert "position" in ws.topics
    assert "order" in ws.topics


def test_bybit_gets_the_bybit_public_feed():
    feed = _creds(Exchange.BYBIT_LINEAR).to_public_exchange_event(symbols=["BTCUSDT"])
    assert isinstance(feed, BybitPublicWS)
    assert feed.topics == ["orderbook.1.BTCUSDT"]


def test_binance_still_gets_the_binance_public_feed():
    feed = _creds(Exchange.BINANCE_LINEAR).to_public_exchange_event(symbols=["BTCUSDT"])
    assert isinstance(feed, BinancePublicWS)


@pytest.mark.parametrize("exchange", [Exchange.KUCOIN_LINEAR, Exchange.EDGEX])
def test_exchanges_without_an_adapter_get_none_rather_than_an_error(exchange):
    """
    Kucoin and EdgeX have no public adapter and must keep running on REST
    prices, exactly as they did before the price feed existed. Raising here
    would break two exchanges that have nothing to do with this change.
    """
    assert _creds(exchange).to_public_exchange_event(symbols=["BTCUSDT"]) is None


def test_the_public_feed_follows_the_credentials_environment():
    live = _creds(Exchange.BYBIT_LINEAR).to_public_exchange_event(symbols=["BTCUSDT"])
    test = _creds(Exchange.BYBIT_LINEAR, testnet=True).to_public_exchange_event(
        symbols=["BTCUSDT"]
    )
    assert "stream.bybit.com" in live.url
    assert "stream-testnet" in test.url

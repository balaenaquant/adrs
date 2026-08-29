import pytest
from cybotrade.hyperliquid import (
    HyperliquidClient,
    HyperliquidPrivateWS,
    HyperliquidPublicWS,
)
from cybotrade.models import Exchange

from adrs.oms.config import Credentials
from adrs.oms.rate_limit.error_policy import HyperliquidErrorPolicy

KEY = "0x0123456789012345678901234567890123456789012345678901234567890123"
ADDRESS = "0x" + "42" * 20
VAULT = "0x" + "cd" * 20


def _creds(**overrides) -> Credentials:
    kwargs = dict(
        exchange=Exchange.HYPERLIQUID,
        api_key=ADDRESS,  # master account address
        api_secret=KEY,  # signing key
    )
    kwargs.update(overrides)
    return Credentials(**kwargs)  # type: ignore[arg-type]


def test_exchange_client():
    client = _creds().to_exchange_client()
    assert isinstance(client, HyperliquidClient)
    assert client.address == ADDRESS
    assert client.url == "https://api.hyperliquid.xyz"


def test_exchange_client_testnet():
    client = _creds(testnet=True).to_exchange_client()
    assert client.url == "https://api.hyperliquid-testnet.xyz"


def test_vault_address_comes_from_the_passphrase_field():
    client = _creds(api_passphrase=VAULT).to_exchange_client()
    assert client.vault_address == VAULT


def test_no_vault_by_default():
    assert _creds().to_exchange_client().vault_address is None


@pytest.mark.parametrize("bad", ["", None])
def test_missing_account_address_is_rejected(bad):
    """
    api_key must be the MASTER account address. cybotrade can resolve it from
    userRole, but that call is awaited and to_exchange_event() is synchronous,
    so there is nowhere to await it -- and putting the agent wallet's own
    address here makes every state read return empty while orders still place.
    """
    creds = _creds(api_key=bad or "")
    with pytest.raises(ValueError, match="master account address"):
        creds.to_exchange_client()
    with pytest.raises(ValueError, match="master account address"):
        creds.to_exchange_event()


def test_private_stream_uses_the_account_address():
    stream = _creds().to_exchange_event()
    assert isinstance(stream, HyperliquidPrivateWS)
    assert stream.address == ADDRESS
    assert "testnet" not in stream.url


def test_private_stream_testnet():
    assert "testnet" in _creds(testnet=True).to_exchange_event().url


def test_public_feed_is_wired():
    """
    Unlike Kucoin and EdgeX, Hyperliquid has a real top-of-book stream, so this
    must not return None -- polling REST would spend the per-IP weight budget
    on prices.
    """
    feed = _creds().to_public_exchange_event(["BTCUSDC", "ETHUSDC"])
    assert isinstance(feed, HyperliquidPublicWS)
    assert [str(s) for s in feed.symbols] == ["BTCUSDC", "ETHUSDC"]


def test_public_feed_resolves_the_hyperliquid_coin():
    """
    Guards a silent failure: HyperliquidPublicWS resolves each symbol to a coin
    via base_from_symbol(), which returns "BTC" for Symbol("BTCUSDC") but
    "BTCUSDC" for the plain string. Passing raw strings would subscribe to a
    coin that does not exist, with no error raised.
    """
    feed = _creds().to_public_exchange_event(["BTCUSDC", "ETHUSDC"])
    coins = [sub["coin"] for sub in feed._subscriptions()]
    assert coins == ["BTC", "ETH"]


def test_topic():
    assert _creds().to_exchange_topic() == "hyperliquid"


def test_error_policy():
    assert isinstance(_creds().to_error_policy(), HyperliquidErrorPolicy)

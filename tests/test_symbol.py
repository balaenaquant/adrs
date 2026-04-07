import pytest
from adrs.types import Symbol


def test_symbol_split_separator():
    assert Symbol.from_str("BTC-USDT").split() == ("BTC", "USDT")
    assert Symbol.from_str("BTC/USDT").split() == ("BTC", "USDT")
    assert Symbol.from_str("BTC_USDT").split() == ("BTC", "USDT")
    assert Symbol.from_str("BTC:USDT").split() == ("BTC", "USDT")


def test_symbol_split_suffix():
    assert Symbol.from_str("BTCUSDT").split() == ("BTC", "USDT")
    assert Symbol.from_str("BTCUSDC").split() == ("BTC", "USDC")
    assert Symbol.from_str("BTCUSD").split() == ("BTC", "USD")
    assert Symbol.from_str("BTCFDUSD").split() == ("BTC", "FDUSD")
    assert Symbol.from_str("BTCUSDTM").split() == ("BTC", "USDTM")
    assert Symbol.from_str("BTCUSDCM").split() == ("BTC", "USDCM")


def test_symbol_split_suffix_lowercase():
    assert Symbol.from_str("btcusdt").split() == ("btc", "usdt")
    assert Symbol.from_str("btcusd").split() == ("btc", "usd")
    assert Symbol.from_str("btcfdusd").split() == ("btc", "fdusd")


def test_symbol_split_none():
    assert Symbol.from_str("BTC").split() is None
    assert Symbol.from_str("XY").split() is None
    assert Symbol.from_str("ABCDEF").split() is None


def test_symbol_hash():
    s = Symbol.from_str("BTC")
    expected = ord("B") + ord("T") + ord("C")
    assert s.hash() == expected
    assert hash(s) == expected


def test_symbol_copy():
    import copy
    s = Symbol.from_str("BTCUSDT")
    assert copy.copy(s) is s
    assert copy.deepcopy(s) is s


def test_symbol_pickle():
    import pickle
    for sym_str in ("BTCUSDT", "BTC-USDT", "ETH"):
        s = Symbol.from_str(sym_str)
        assert pickle.loads(pickle.dumps(s)) == s


def test_symbol_eq():
    assert Symbol.from_str("BTCUSDT") == Symbol.from_str("BTCUSDT")
    assert Symbol.from_str("BTCUSDT") == "BTCUSDT"
    assert Symbol.from_str("BTCUSDT") != Symbol.from_str("ETHUSDT")


def test_symbol_str():
    assert str(Symbol.from_str("BTCUSDT")) == "BTCUSDT"


def test_symbol_max_len():
    with pytest.raises(ValueError):
        Symbol.from_str("A" * 25)


def test_symbol_serde():
    from pydantic import BaseModel

    class Wrapper(BaseModel):
        symbol: Symbol

    w = Wrapper(symbol=Symbol.from_str("BTCUSDT"))
    assert w.model_dump() == {"symbol": "BTCUSDT"}
    assert Wrapper.model_validate({"symbol": "BTCUSDT"}).symbol == Symbol.from_str("BTCUSDT")

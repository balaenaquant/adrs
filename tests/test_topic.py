import pytest
from adrs.types import Topic, TopicError


def test_topic_from_str():
    testcases = [
        (
            "cryptoquant|btc/exchange-flows/reserve?exchange=binance&limit=20000&window=hour",
            Topic(
                provider="cryptoquant",
                endpoint="btc/exchange-flows/reserve",
                params={"exchange": "binance", "limit": "20000", "window": "hour"},
            ),
        ),
        (
            "glassnode|derivatives/futures_funding_rate_perpetual_v2?a=BTC&i=1h&e=bybit&s=1728313560&u=1739113560",
            Topic(
                provider="glassnode",
                endpoint="derivatives/futures_funding_rate_perpetual_v2",
                params={
                    "a": "BTC",
                    "e": "bybit",
                    "i": "1h",
                    "s": "1728313560",
                    "u": "1739113560",
                },
            ),
        ),
        (
            # same topic as below but different query param order — equality is order-independent
            "cryptoquant|btc/market-data/taker-buy-sell-stats?exchange=binance&window=hour&to=20250209T150500&limit=1000",
            Topic(
                provider="cryptoquant",
                endpoint="btc/market-data/taker-buy-sell-stats",
                params={
                    "exchange": "binance",
                    "limit": "1000",
                    "to": "20250209T150500",
                    "window": "hour",
                },
            ),
        ),
        (
            # same topic as above but different query param order and different limit
            "cryptoquant|btc/market-data/taker-buy-sell-stats?window=hour&exchange=binance&to=20250209T150500&limit=3700",
            Topic(
                provider="cryptoquant",
                endpoint="btc/market-data/taker-buy-sell-stats",
                params={
                    "exchange": "binance",
                    "limit": "3700",
                    "to": "20250209T150500",
                    "window": "hour",
                },
            ),
        ),
        (
            # no query params
            "glassnode|derivatives/futures_funding_rate_perpetual_v2",
            Topic(
                provider="glassnode",
                endpoint="derivatives/futures_funding_rate_perpetual_v2",
                params={},
            ),
        ),
    ]

    for topic_str, expected in testcases:
        assert Topic.from_str(topic_str) == expected

    with pytest.raises(TopicError):
        Topic.from_str(
            "glassnode|derivatives/futures_funding_rate_perpetual_v2|Normalized"
        )


def test_topic_to_string():
    testcases = [
        (
            "cryptoquant|btc/exchange-flows/reserve?exchange=binance&limit=20000&window=hour",
            Topic(
                provider="cryptoquant",
                endpoint="btc/exchange-flows/reserve",
                params={"exchange": "binance", "limit": "20000", "window": "hour"},
            ),
        ),
        (
            "glassnode|derivatives/futures_funding_rate_perpetual_v2?a=BTC&e=bybit&i=1h&s=1728313560&u=1739113560",
            Topic(
                provider="glassnode",
                endpoint="derivatives/futures_funding_rate_perpetual_v2",
                params={
                    "a": "BTC",
                    "e": "bybit",
                    "i": "1h",
                    "s": "1728313560",
                    "u": "1739113560",
                },
            ),
        ),
        (
            "cryptoquant|btc/market-data/taker-buy-sell-stats?exchange=binance&limit=1000&to=20250209T150500&window=hour",
            Topic(
                provider="cryptoquant",
                endpoint="btc/market-data/taker-buy-sell-stats",
                params={
                    "exchange": "binance",
                    "limit": "1000",
                    "to": "20250209T150500",
                    "window": "hour",
                },
            ),
        ),
        (
            "glassnode|derivatives/futures_funding_rate_perpetual_v2",
            Topic(
                provider="glassnode",
                endpoint="derivatives/futures_funding_rate_perpetual_v2",
                params={},
            ),
        ),
    ]

    for expected_str, topic in testcases:
        assert str(topic) == expected_str


def test_topic_serde():
    testcases = [
        "cryptoquant|btc/exchange-flows/reserve?exchange=binance&limit=20000&window=hour",
        "glassnode|derivatives/futures_funding_rate_perpetual_v2?a=BTC&e=bybit&i=1h&s=1728313560&u=1739113560",
        "cryptoquant|btc/market-data/taker-buy-sell-stats?exchange=binance&limit=1000&to=20250209T150500&window=hour",
        "glassnode|derivatives/futures_funding_rate_perpetual_v2",
    ]

    for topic_str in testcases:
        topic = Topic.from_str(topic_str)
        serialized = topic.model_dump()
        assert serialized == topic_str
        assert Topic.model_validate(topic_str) == topic


def test_cryptoquant_eth_exchange_flows_delay_is_3min():
    _MINUTE = 60_000
    topic = Topic.from_str(
        "cryptoquant|eth/exchange-flows/inflow?window=hour&exchange=all_exchange"
    )
    assert topic.delay_ms() == 3 * _MINUTE


def test_delay_ms():
    _SECOND = 1_000
    _MINUTE = 60 * _SECOND
    _HOUR = 60 * _MINUTE

    cases = [
        ("cryptoquant|btc/exchange-flows/reserve?window=min", 10 * _SECOND),
        ("cryptoquant|btc/exchange-flows/reserve?window=block", 30 * _SECOND),
        ("cryptoquant|btc/exchange-flows/reserve?window=hour", _MINUTE),
        ("glassnode|derivatives/futures_funding_rate_perpetual_v2?i=10m", 30 * _SECOND),
        ("glassnode|derivatives/futures_funding_rate_perpetual_v2?i=1h", _MINUTE),
        ("glassnode|derivatives/futures_funding_rate_perpetual_v2?i=1w", _HOUR),
        ("coinglass|candle?interval=1m", 10 * _SECOND),
        ("coinglass|candle?interval=15m", 30 * _SECOND),
        ("coinglass|candle?interval=1h", _MINUTE),
        ("coinglass|candle?interval=1d", _HOUR),
    ]
    for topic_str, expected in cases:
        assert Topic.from_str(topic_str).delay_ms() == expected, topic_str


def test_is_block():
    assert Topic.from_str(
        "cryptoquant|btc/exchange-flows/reserve?window=block"
    ).is_block()
    assert not Topic.from_str(
        "cryptoquant|btc/exchange-flows/reserve?window=hour"
    ).is_block()
    assert not Topic.from_str(
        "glassnode|derivatives/futures_funding_rate_perpetual_v2?i=1h"
    ).is_block()


def test_cron():
    cases = [
        ("cryptoquant|btc/exchange-flows/reserve?window=min", "6 * * * * *"),
        ("cryptoquant|btc/exchange-flows/reserve?window=hour", "0 15 * * * *"),
        ("cryptoquant|btc/market-data/taker-buy-sell-stats?window=hour", "0 6 * * * *"),
        ("glassnode|derivatives/futures_funding_rate_perpetual_v2?i=1h", "0 6 * * * *"),
        ("glassnode|derivatives/futures_funding_rate_perpetual_v2?i=1w", "0 0 0 * * 1"),
        ("coinglass|candle?interval=1m", "1 * * * * *"),
        ("coinglass|candle?interval=1h", "1 0 * * * *"),
        ("coinglass|candle?interval=1d", "1 0 0 * * *"),
        ("cryptoquant|btc/exchange-flows/reserve", None),
    ]
    for topic_str, expected in cases:
        assert Topic.from_str(topic_str).cron() == expected, topic_str


def test_topic_copy():
    import copy

    topic = Topic.from_str(
        "glassnode|derivatives/futures_funding_rate_perpetual_v2?i=1h"
    )
    assert copy.copy(topic) is topic
    assert copy.deepcopy(topic) is topic


def test_topic_pickle():
    import pickle

    cases = [
        "cryptoquant|btc/exchange-flows/reserve?exchange=binance&limit=20000&window=hour",
        "glassnode|derivatives/futures_funding_rate_perpetual_v2?a=BTC&e=bybit&i=1h",
        "glassnode|derivatives/futures_funding_rate_perpetual_v2",
    ]
    for topic_str in cases:
        topic = Topic.from_str(topic_str)
        assert pickle.loads(pickle.dumps(topic)) == topic


def test_topic_query_params_alias():
    t1 = Topic(provider="glassnode", endpoint="foo", params={"a": "1"})
    t2 = Topic(provider="glassnode", endpoint="foo", query_params={"a": "1"})
    assert t1 == t2
    assert t1.query_params() == {"a": "1"}


def test_topics_list_serde():
    topic_strs = [
        "cryptoquant|btc/exchange-flows/reserve?exchange=binance&limit=20000&window=hour",
        "glassnode|derivatives/futures_funding_rate_perpetual_v2?a=BTC&e=bybit&i=1h&s=1728313560&u=1739113560",
    ]
    topics = [Topic.from_str(s) for s in topic_strs]

    serialized = [t.model_dump() for t in topics]
    assert serialized == topic_strs

    deserialized = [Topic.model_validate(s) for s in topic_strs]
    assert deserialized == topics

from datetime import timedelta
from types import SimpleNamespace

from adrs.data.types import DataColumn, DataInfo
from adrs.execution.runner import _finest_interval


def _alpha(*topics: str):
    """Duck-typed alpha: _finest_interval only reads `.data_infos[*].topic`."""
    return SimpleNamespace(
        data_infos=[
            DataInfo(
                topic=t, columns=[DataColumn(src="close", dst="price")], lookback_size=0
            )
            for t in topics
        ]
    )


def test_finest_interval_picks_smallest_across_alphas():
    alphas = [
        _alpha("cryptoquant|btc/market-data/price-ohlcv?window=24h"),  # 1 day
        _alpha("bybit-linear|candle?symbol=BTCUSDT&interval=1h"),  # 1 hour
    ]
    assert _finest_interval(alphas) == timedelta(hours=1)


def test_finest_interval_none_when_no_interval():
    # data-alert / point-in-time topics carry no interval.
    assert _finest_interval([_alpha("cybotrade|data-alert")]) is None


def test_finest_interval_empty():
    assert _finest_interval([]) is None

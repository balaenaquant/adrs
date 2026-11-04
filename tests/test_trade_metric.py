"""Trade metric on a hand-checked sequence of trades.

Also guards the Polars 2.x upgrade: `pl.count()` with no arguments raises
TypeError there, which broke Trade().compute (and so every Alpha.backtest and
Portfolio.backtest) while no test exercised it.
"""

from datetime import datetime, timedelta, timezone

import polars as pl

from adrs.performance.metric import Trade

S = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_trade_metric_counts_wins_losses_and_streaks():
    # t1 open long @101, t3 close @102 (win)
    # t4 open short @104, t6 close @103 (win)
    # t7 open long @105, t8 close @104 (loss)
    signal = [0, 1, 1, 0, -1, -1, 0, 1, 0]
    price = [100.0, 101.0, 103.0, 102.0, 104.0, 106.0, 103.0, 105.0, 104.0]
    prev = [0] + signal[:-1]
    df = pl.DataFrame(
        {
            "start_time": [S + timedelta(hours=h) for h in range(len(signal))],
            "signal": signal,
            "prev_signal": prev,
            "price": price,
            "trade": [s - p for s, p in zip(signal, prev)],
            "pnl": [0.0, 0.0, 0.02, -0.01, 0.0, -0.02, 0.03, 0.0, -0.01],
        }
    )

    result = Trade().compute(df)

    assert result["num_trades"] == 6
    assert result["long_trades"] == 3
    assert result["short_trades"] == 3
    assert result["win_trades"] == 2
    assert result["lose_trades"] == 1
    assert result["win_streak"] == 2
    assert result["lose_streak"] == 1
    assert result["win_rate"] == 2 / 3
    assert result["avg_holding_time_in_seconds"] == 6000.0  # (2h + 2h + 1h) / 3
    assert result["max_holding_time_in_seconds"] == 7200.0
    assert result["largest_win"] == 0.03
    assert result["largest_loss"] == -0.02

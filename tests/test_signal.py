import numpy as np
import operator as op
from datetime import datetime, timedelta
from numpy.typing import ArrayLike
from adrs.signal import (
    Signal,
    Long,
    Short,
)

INPUTS = np.array([0, 0.5, 1.0, 1.2, -0.5, -1.0, -1.2, -0.5, 0, 0.5, 1.0])
TS_INPUTS = np.array(
    [
        datetime.fromisoformat("2025-10-10T00:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T01:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T02:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T03:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T04:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T05:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T06:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T07:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T08:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T09:00:00Z").timestamp() * 1000,
        datetime.fromisoformat("2025-10-10T10:00:00Z").timestamp() * 1000,
    ]
)


def reverse_signals(signals: ArrayLike) -> ArrayLike:
    return np.vectorize(lambda s: s * -1)(signals)


LONG_SHORT_NO_EXIT = np.array(
    [
        Signal.NONE,
        Signal.NONE,
        Signal.BUY,
        Signal.BUY,
        Signal.BUY,
        Signal.SELL,
        Signal.SELL,
        Signal.SELL,
        Signal.SELL,
        Signal.SELL,
        Signal.BUY,
    ]
)

LONG_SHORT_WITH_EXIT = np.array(
    [
        Signal.NONE,
        Signal.NONE,
        Signal.BUY,
        Signal.BUY,
        Signal.NONE,
        Signal.SELL,
        Signal.SELL,
        Signal.SELL,
        Signal.NONE,
        Signal.NONE,
        Signal.BUY,
    ]
)


def test_long():
    siggen = Long(long_entry_thres=1.0, long_exit_thres=-1.0)
    assert siggen.id() == "long"
    assert siggen.num_inputs() == 2
    signals = siggen.generate(INPUTS, TS_INPUTS)
    assert np.array_equal(
        signals,
        np.array(
            [
                Signal.NONE,
                Signal.NONE,
                Signal.BUY,
                Signal.BUY,
                Signal.BUY,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.BUY,
            ]
        ),
    )
    siggen = Long(long_entry_thres=-1.0, long_exit_thres=1.0, reverse=True)
    signals = siggen.generate(INPUTS, TS_INPUTS)
    assert np.array_equal(
        signals,
        np.array(
            [
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.BUY,
                Signal.BUY,
                Signal.BUY,
                Signal.BUY,
                Signal.BUY,
                Signal.NONE,
            ]
        ),
    )


def test_long_with_max_hold_duration():
    siggen = Long(
        long_entry_thres=1.0, long_exit_thres=-1.0, max_hold_duration=timedelta(hours=2)
    )
    assert siggen.id() == "long"
    assert siggen.num_inputs() == 2
    signals = siggen.generate(INPUTS, TS_INPUTS)
    assert np.array_equal(
        signals,
        np.array(
            [
                Signal.NONE,
                Signal.NONE,
                Signal.BUY,
                Signal.BUY,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.BUY,
            ]
        ),
    )


def test_short():
    siggen = Short(short_entry_thres=-1.0, short_exit_thres=1.0)
    assert siggen.id() == "short"
    assert siggen.num_inputs() == 2
    signals = siggen.generate(INPUTS, TS_INPUTS)
    assert np.array_equal(
        signals,
        np.array(
            [
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.SELL,
                Signal.SELL,
                Signal.SELL,
                Signal.SELL,
                Signal.SELL,
                Signal.NONE,
            ]
        ),
    )
    siggen = Short(short_entry_thres=1.0, short_exit_thres=-1.0, reverse=True)
    signals = siggen.generate(INPUTS, TS_INPUTS)
    assert np.array_equal(
        signals,
        np.array(
            [
                Signal.NONE,
                Signal.NONE,
                Signal.SELL,
                Signal.SELL,
                Signal.SELL,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.SELL,
            ]
        ),
    )


def test_short_with_max_hold_duration():
    siggen = Short(
        short_entry_thres=-1.0,
        short_exit_thres=1.0,
        max_hold_duration=timedelta(hours=2),
    )
    signals = siggen.generate(INPUTS, TS_INPUTS)
    assert np.array_equal(
        signals,
        np.array(
            [
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
                Signal.SELL,
                Signal.SELL,
                Signal.NONE,  # Exit after 2 periods of holding, no hold time is SELL
                Signal.NONE,
                Signal.NONE,
                Signal.NONE,
            ]
        ),
    )


def test_operator():
    from adrs.signal.operator import ge, le

    assert ge(3, 3) == op.ge(3, 3)
    assert ge(3, 2) == op.ge(3, 2)
    assert le(3, 3) == op.le(3, 3)
    assert le(2, 3) == op.le(2, 3)
    assert not op.le(4.8617e-13, 0)  # original python operator is exact
    assert not op.ge(-4.8617e-13, 0)  # original python operator is exact

    # NOTE: Disabling this for now until there's a need
    # assert le(4.8617e-13, 0)  # smaller than 1e-8 consider as zero
    # assert ge(-4.8617e-13, 0)  # smaller than 1e-8 consider as zero

import numpy as np
import adrs.signal.operator as op
from typing import Sequence, override, cast
from datetime import datetime, timedelta, timezone

from adrs.signal import Signal, SignalGenerator
from adrs.utils import is_indexable, check_max_hold_duration


class Long(SignalGenerator):
    def __init__(
        self,
        long_entry_thres: float | Sequence[float],
        long_exit_thres: float | Sequence[float],
        reverse: bool = False,
        check_equal: bool = True,
        max_hold_duration: timedelta = timedelta(),
    ):
        self.long_entry_thres, self.long_exit_thres = (
            long_entry_thres,
            long_exit_thres,
        )
        self.max_hold_duration = max_hold_duration
        gt, lt = (op.ge, op.le) if check_equal else (op.gt, op.lt)
        self.gt, self.lt = (lt, gt) if reverse else (gt, lt)

    @staticmethod
    def id() -> str:
        return "long"

    @staticmethod
    def num_inputs() -> int:
        return 2

    @override
    def _generate(self, inputs):
        datapoints, timestamps = inputs[0], inputs[1]
        signals = np.zeros(len(datapoints), dtype=Signal)

        # sanity check
        if len(datapoints) != len(timestamps):
            raise ValueError(
                f"datapoints(len: {len(datapoints)}) and timestamps(len: {len(timestamps)}) should have equal length"
            )
        if len(datapoints) == 0:
            return signals

        cpos = 0
        last_timestamp = datetime.fromtimestamp(timestamps[0] // 1000, tz=timezone.utc)

        for i in range(len(datapoints)):
            long_entry_thres = (
                self.long_entry_thres[i]
                if is_indexable(self.long_entry_thres)
                else cast(float, self.long_entry_thres)
            )
            long_exit_thres = (
                self.long_exit_thres[i]
                if is_indexable(self.long_exit_thres)
                else cast(float, self.long_exit_thres)
            )
            if long_entry_thres is None or long_exit_thres is None:
                signals[i] = cpos
                continue

            # fmt: off
            ppos = cpos
            cpos = (
                1 if self.gt(datapoints[i], long_entry_thres)
                else 0 if self.lt(datapoints[i], long_exit_thres)
                else cpos
            )
            # fmt: on

            cpos, last_timestamp = check_max_hold_duration(
                max_hold_duration=self.max_hold_duration,
                last_timestamp=last_timestamp,
                current_timestamp=datetime.fromtimestamp(
                    timestamps[i] // 1000, tz=timezone.utc
                ),
                cpos=cpos,
                ppos=ppos,
            )

            signals[i] = cpos

        return signals

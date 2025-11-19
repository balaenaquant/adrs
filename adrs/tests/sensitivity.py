import copy
import inspect
import numbers
from decimal import Decimal
from typing import cast, Any
from pydantic import BaseModel
from datetime import timedelta

from adrs.alpha import Alpha
from adrs.types import Performance, PerformanceDF

type AllowedParam = int | float | timedelta


class SensitivityParameter(BaseModel):
    min_val: AllowedParam | None = None
    min_gap: AllowedParam | None = None


def get_dp(val: AllowedParam) -> int | None:
    if isinstance(val, timedelta):
        return None
    return abs(cast(int, Decimal(str(val)).as_tuple().exponent))


def derive_gap(val: Any, gap_percent: float, min_gap: AllowedParam | None = None):
    if isinstance(val, int):
        gap = int(val * gap_percent)
    elif isinstance(val, numbers.Real):
        gap = round(float(val) * gap_percent, get_dp(float(val)))
    elif isinstance(val, timedelta):
        gap = timedelta(seconds=int(val.total_seconds() * gap_percent))
    else:
        raise TypeError(f"Unsupported type for value: {type(val)}")

    return gap if min_gap is None else max(gap, min_gap)


class Sensitivity:
    def __init__(
        self,
        alpha: Alpha,
        parameters: dict[str, SensitivityParameter],
        gap_percent: float = 0.15,
        num_steps: int = 3,
    ):
        signature = inspect.signature(alpha.__init__)
        alpha_name = type(alpha).__name__

        if num_steps < 1:
            raise ValueError("num_steps must be a non-zero positive integer")

        if len(parameters) < 2:
            raise ValueError("Must have at least 2 parameters.")

        self.iters = 2 * num_steps + 1
        self.sensitivity_parameters = parameters
        self.parameters: dict[str, list[AllowedParam]] = {}
        for name, param in parameters.items():
            if name not in signature.parameters:
                raise ValueError(
                    f"Parameter '{name}' not found in {alpha_name}'s __init__ method."
                )

            val = alpha.__getattribute__(name)
            if not isinstance(val, (numbers.Number, timedelta)):
                raise ValueError(
                    f"Parameter '{name}' in {alpha_name} is not a number nor timedelta."
                )

            value = cast(AllowedParam, val)
            dp = get_dp(value)
            gap = derive_gap(val, gap_percent, param.min_gap)
            permutations: list[AllowedParam] = [cast(AllowedParam, val)]
            for i in range(1, num_steps + 1):
                upper, lower = value + gap * i, value - gap * i  # type: ignore
                lower = lower if param.min_val is None else max(lower, param.min_val)
                upper = round(upper, dp) if dp is not None else upper  # type: ignore
                lower = round(lower, dp) if dp is not None else lower  # type: ignore
                permutations += [upper, lower]

            self.parameters[name] = permutations

    def test(
        self,
        alpha: Alpha,
    ) -> list[tuple[dict[str, AllowedParam], Performance, PerformanceDF]]:
        alpha = copy.copy(alpha)
        results = []
        for i in range(self.iters):
            ps: dict[str, AllowedParam] = {}
            for name, params in self.parameters.items():
                ps[name] = params[i]
                alpha.__setattr__(name, params[i])
            alpha.backtest()
            performance, performance_df = alpha.evaluate()
            results.append((ps, performance, performance_df))
        return results

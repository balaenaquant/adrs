# import numpy as np
import operator as op
from typing import Any
from operator import lt, gt

ATOL = 1e-8


def ge(a, b) -> Any:
    # NOTE: Disabling this behaviour for now until there is a need
    # if a == 0 or b == 0:
    #     return (a > b) | np.isclose(a, b, atol=ATOL)
    return op.ge(a, b)


def le(a: Any, b: Any) -> Any:
    # NOTE: Disabling this behaviour for now until there is a need
    # if a == 0 or b == 0:
    #     return (a < b) | np.isclose(a, b, atol=ATOL)
    return op.le(a, b)


__all__ = ["lt", "gt", "ge", "le"]

from itertools import product
from typing import Sequence, override

from adrs.search import Search, ParameterGrid, Permutation


class GridSearch(Search):
    def __init__(self):
        pass

    @staticmethod
    def id() -> str:
        return "gridsearch"

    @override
    def search(self, grid: ParameterGrid) -> Sequence[Permutation]:
        return [dict(zip(grid, combo)) for combo in product(*grid.values())]  # type: ignore

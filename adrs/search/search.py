from abc import abstractmethod
from typing import Any, Sequence
from numpy.typing import ArrayLike

type ParameterGrid = dict[str, ArrayLike]
type Permutation = dict[str, Any]


class Search:
    """
    Search represents a search function that takes in a grid of parameters and return
    a set of permutations of the parameter grid.

    It can be expressed mathematically as:

    * f(grid) -> ys

    There are ton of permutations while performing hyperparameter-tuning hence a
    good search function helps reduce the computation needed to perform such operation
    by cutting down the problem space.
    """

    @staticmethod
    @abstractmethod
    def id() -> str:
        """A name of identifier for the search."""
        raise NotImplementedError("All search should have an identifier.")

    @abstractmethod
    def search(self, grid: ParameterGrid) -> Sequence[Permutation]:
        raise NotImplementedError(f"Search {self.id()} does not implement search().")

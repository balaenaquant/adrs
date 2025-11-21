import math
import numpy as np
from typing import Sequence, override, cast

from adrs.search import Search, ParameterGrid, Permutation

from enum import StrEnum


class Distribution(StrEnum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    BETA = "beta"
    GAMMA = "gamma"


class RandomSearch(Search):
    def __init__(
        self,
        samples: int | float | np.float64,
        seed: int,
        dist: Distribution,
        **dist_kwargs,
    ):
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.dist = dist
        self.dist_kwargs = dist_kwargs

    @staticmethod
    def id() -> str:
        return "randomsearch"

    def _dist(self, n: int) -> np.ndarray:
        match self.dist:
            case Distribution.UNIFORM:
                return self.rng.uniform(0, 1, n)
            case Distribution.NORMAL:
                return self.rng.normal(0, 1, n)
            case Distribution.LOGNORMAL:
                return self.rng.lognormal(0, 1, n)
            case Distribution.EXPONENTIAL:
                return self.rng.exponential(1, n)
            case Distribution.BETA:
                return self.rng.beta(
                    self.dist_kwargs.get("a", 2), self.dist_kwargs.get("b", 5), n
                )
            case Distribution.GAMMA:
                return self.rng.gamma(
                    self.dist_kwargs.get("shape", 2),
                    self.dist_kwargs.get("scale", 1),
                    n,
                )
            case dist:
                raise ValueError(f"Invalid distribution: {dist}")

    @override
    def search(self, grid: ParameterGrid) -> Sequence[Permutation]:
        if len(grid) == 0:
            return []

        # determine number of samples
        total_perms = math.prod(map(lambda x: len(x), grid.values()))
        if isinstance(self.samples, float | np.float64):
            n = round(total_perms * self.samples)
        else:
            n = cast(int, self.samples)

        permutations = []

        for _ in range(n):
            perm: Permutation = {}
            for k, v in grid.items():
                # sample from distribution and avoid negatives
                weights = np.abs(self._dist(len(v)))  # type: ignore
                probs = weights / weights.sum()  # normalize to (0, 1)
                perm[k] = self.rng.choice(a=v, p=probs)
            permutations.append(perm)

        return permutations

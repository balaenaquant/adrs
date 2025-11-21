import numpy as np

from adrs.search import GridSearch, RandomSearch, Distribution

SEED = 42


def test_grid_search():
    search = GridSearch()
    permutations = search.search(grid={"a": np.arange(100), "b": np.arange(100)})
    assert len(permutations) == 100 * 100


def test_random_search():
    search = RandomSearch(samples=10, seed=SEED, dist=Distribution.NORMAL)
    permutations = search.search(grid={"a": np.arange(100), "b": np.arange(100)})
    assert len(permutations) == 10

    search = RandomSearch(samples=0.1, seed=SEED, dist=Distribution.NORMAL)
    permutations = search.search(grid={"a": np.arange(100), "b": np.arange(100)})
    assert len(permutations) == (100 * 100) * 0.1

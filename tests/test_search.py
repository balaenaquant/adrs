import numpy as np

from adrs.search import GridSearch, RandomSearch, Distribution

SEED = 42


def test_grid_search():
    search = GridSearch()
    permutations = search.search(grid={"a": np.arange(100), "b": np.arange(100)})
    assert len(permutations) == 100 * 100
    assert len(search.filter(permutations)) == len(permutations)


def test_random_search():
    search = RandomSearch(samples=10, seed=SEED, dist=Distribution.NORMAL)
    permutations = search.search(grid={"a": np.arange(100), "b": np.arange(100)})
    assert len(permutations) == 10
    assert (
        len(
            RandomSearch(samples=5, seed=SEED, dist=Distribution.NORMAL).filter(
                permutations
            )
        )
        == 5
    )

    search = RandomSearch(samples=0.1, seed=SEED, dist=Distribution.NORMAL)
    permutations = search.search(grid={"a": np.arange(100), "b": np.arange(100)})
    assert len(permutations) == (100 * 100) * 0.1
    assert len(search.filter(permutations)) == (100 * 100) * 0.1 * 0.1


def test_random_search_large_arrays():
    search = RandomSearch(samples=100_000, seed=SEED, dist=Distribution.UNIFORM)
    permutations = search.search(
        grid={"a": np.arange(100), "b": np.arange(100), "c": np.arange(100)}
    )
    assert len(permutations) == 100_000

    assert (
        len(
            RandomSearch(samples=10_000, seed=SEED, dist=Distribution.UNIFORM).filter(
                permutations
            )
        )
        == 10_000
    )

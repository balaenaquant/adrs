import pytest
import numpy as np
import polars as pl
from typing import Type, Any
from numpy.testing import assert_array_almost_equal

from adrs.model import Model, ZScore

WINDOW = 100


@pytest.fixture
def datas() -> list[pl.Series]:
    price_df = pl.read_parquet(
        "tests/data/interval_1m_symbol_BTCUSDT_2022-07-03.parquet"
    )
    close = price_df["close"].cast(pl.Float64)

    block_df = pl.read_parquet("tests/data/a_BTC_i_1h_2025-01-*.parquet")
    lth = block_df["lth"].cast(pl.Float64)

    def zscore(s: pl.Series) -> pl.Series:
        return (s - s.rolling_mean(window_size=WINDOW)) / s.rolling_std(
            window_size=WINDOW
        )

    return [
        close,
        zscore(close),
        lth,
        zscore(lth),
    ]


WINDOW_MODELS = [
    (ZScore, {}, 0.678218),
]


@pytest.mark.parametrize(
    argnames=("model_cls", "params", "last"),
    argvalues=WINDOW_MODELS,
    ids=[f"{tc[0].__name__}-{tc[1]}-{tc[2]}" for tc in WINDOW_MODELS],
)
def test_window_models(
    model_cls: Type[Model],
    params: dict[str, Any],
    last: Any,
    datas: list[pl.Series],
):
    model = model_cls(window=WINDOW, **params)  # type: ignore

    for i, data in enumerate(datas):
        pl_output = model.eval(data)[0]
        pd_output = model.eval(data.to_pandas())[0]

        assert len(pl_output) == len(pd_output) == len(data)
        assert pl_output.null_count() == pd_output.isna().sum()
        assert_array_almost_equal(pl_output.to_numpy(), pd_output.to_numpy())

        if i == 0:
            assert np.isclose(pl_output[-1], last)
            assert np.isclose(pd_output.iloc[-1], last)

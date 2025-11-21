import polars as pl
import pandas as pd
from typing import override
from adrs.model import Model


class ZScore(Model):
    def __init__(self, window: int, ddof: int = 1):
        self.window = window
        self.ddof = ddof

    @staticmethod
    def id() -> str:
        return "zscore"

    @staticmethod
    def num_inputs() -> int:
        return 1

    @staticmethod
    def num_outputs() -> int:
        return 1

    @override
    def eval_polars(self, parameters: list[pl.Series]):
        sma = parameters[0].rolling_mean(window_size=self.window)
        std = parameters[0].rolling_std(window_size=self.window, ddof=self.ddof)
        return [(parameters[0] - sma) / std]

    @override
    def eval_pandas(self, parameters: list[pd.Series]):
        sma = parameters[0].rolling(window=self.window).mean()
        std = parameters[0].rolling(window=self.window).std(ddof=self.ddof)
        return [pd.Series((parameters[0] - sma) / std)]

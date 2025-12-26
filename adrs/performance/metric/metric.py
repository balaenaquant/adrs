import polars as pl
from abc import abstractmethod


class Metrics[T: dict]:
    @abstractmethod
    def compute(self, df: pl.DataFrame) -> T:
        """Evaluate the metric."""
        raise NotImplementedError("Metrics is not implemented.")

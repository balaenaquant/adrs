from typing import TypeVar, Generic
from abc import abstractmethod
from pandera.typing.polars import DataFrame

from adrs.types import PerformanceDF

T = TypeVar("T", bound=dict)


class Metrics(Generic[T]):
    @abstractmethod
    def compute(self, df: DataFrame[PerformanceDF]) -> T:
        """Evaluate the metric."""
        raise NotImplementedError("Metrics is not implemented.")

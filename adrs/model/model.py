import polars as pl
import pandas as pd
from abc import abstractmethod
from typing import TypeVar, cast


# Type of series
S = TypeVar("S", pl.Series, pd.Series)


class Model:
    """
    Model represents a calculation performed on a dataframe regardless of `polars` or
    `pandas`. This class abstract away the underlying code difference between both
    libraries.

    Note that since `cudf` is fully compatibile with pandas' API it is supported as well.

    It is a mathematical function where it can be:

    * f(x) -> y
    * f(x) -> (y1, y2, ...)
    * f(x1, x2, ...) -> (y1, y2, ...)
    * f(x1, x2, ...) -> y

    Hence it is possible to adjust the number of inputs or outputs on every model and they
    should strictly follow it, the interface of the function should never ever change. From
    a function perspective, these models can be considered a "pure" function in where
    side-effects is prohibited. In other words, these models can be unit-tested since given
    the same set of inputs, the output will never change.
    """

    @staticmethod
    @abstractmethod
    def id() -> str:
        """A name or identifier for the model."""
        raise NotImplementedError("All model should have an identifier.")

    @staticmethod
    @abstractmethod
    def num_inputs() -> int:
        """Number of inputs."""
        raise NotImplementedError("All model should have at least one input.")

    @staticmethod
    @abstractmethod
    def num_outputs() -> int:
        """Number of outputs."""
        raise NotImplementedError("All model should have at least one output.")

    @abstractmethod
    def eval_polars(self, parameters: list[pl.Series]) -> list[pl.Series]:
        """Implementation for polars."""
        raise NotImplementedError(f"Model {self.id()} does not support polars.")

    @abstractmethod
    def eval_pandas(self, parameters: list[pd.Series]) -> list[pd.Series]:
        """Implementation for pandas."""
        raise NotImplementedError(f"Model {self.id()} does not support pandas.")

    def eval(self, *args: S, **kwargs: S) -> list[S]:
        """
        Evaluate the model.
        """
        params = [*args, *kwargs.values()]
        if len(params) != self.num_inputs():
            raise ValueError(
                f"Invalid number of inputs for model {self.id()}: expected {self.num_inputs()} but got {len(params)}"
            )

        outputs = []
        if isinstance(params[0], pl.Series):
            outputs = cast(list[S], self.eval_polars(cast(list[pl.Series], params)))
        elif isinstance(params[0], pd.Series):
            outputs = cast(list[S], self.eval_pandas(cast(list[pd.Series], params)))
        else:
            raise ValueError(
                f"Invalid inputs for model {self.id()}: input is neither polars nor pandas Series"
            )

        if len(outputs) != self.num_outputs():
            raise Exception(
                f"Invalid number of outputs from model {self.id()}: expected {self.num_outputs()} but got {len(outputs)}"
            )

        return outputs

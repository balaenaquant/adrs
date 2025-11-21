from __future__ import annotations

import numpy as np
from enum import Enum
from typing import override
from abc import abstractmethod
from numpy.typing import NDArray

from cybotrade.models import OrderSide


class Signal(np.int8, Enum):
    """Represents a trading signal."""

    BUY = np.int8(1)
    SELL = np.int8(-1)
    NONE = np.int8(0)

    @override
    def __repr__(self) -> str:
        match self:
            case Signal.BUY:
                return "buy"
            case Signal.SELL:
                return "sell"
            case Signal.NONE:
                return "none"
            case sig:
                raise ValueError(f"Invalid signal: {sig}")

    def to_order_side(self) -> OrderSide:
        match self:
            case Signal.BUY:
                return OrderSide.BUY
            case Signal.SELL:
                return OrderSide.SELL
            case Signal.NONE:
                return OrderSide.NONE
            case sig:
                raise ValueError(f"Invalid signal: {sig}")

    @staticmethod
    def from_order_side(side: OrderSide) -> Signal:
        match side:
            case OrderSide.BUY:
                return Signal.BUY
            case OrderSide.SELL:
                return Signal.SELL
            case OrderSide.NONE:
                return Signal.NONE
            case side:
                raise ValueError(f"Invalid order side: {side}")


class SignalGenerator:
    """
    A signal generator is a mathetical function that collapse one or more inputs
    into a single output (signal). It can be written as:

    * f(x) -> y
    * f(x1, x2, ...) -> y
    """

    @staticmethod
    @abstractmethod
    def id() -> str:
        """A name or identifier for the signal generator."""
        raise NotImplementedError("All signal generator should have an identifier.")

    @staticmethod
    @abstractmethod
    def num_inputs() -> int:
        """Number of inputs."""
        raise NotImplementedError(
            "All signal generator should have at least one input."
        )

    @abstractmethod
    def _generate(self, inputs: list[NDArray[np.float64]]) -> NDArray[Signal]:
        raise NotImplementedError(f"Signal generator {self.id()} is not implemented.")

    def generate(
        self, *args: NDArray[np.float64], **kwargs: NDArray[np.float64]
    ) -> NDArray[Signal]:
        """Generate the signal based on the inputs."""

        inputs = [*args, *kwargs.values()]
        if len(inputs) != self.num_inputs():
            raise ValueError(
                f"Invalid number of inputs for signal generator {self.id()}: expected {self.num_inputs()} but got {len(inputs)}"
            )

        return self._generate(inputs)

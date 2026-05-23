import types
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any


class EventType(str, Enum):
    Unknown = "unknown"
    Authenticated = "authenticated"
    Subscribed = "subscribed"
    OrderUpdate = "order_update"
    DatasourceSubscribed = "datasource_subscribed"
    DatasourceUpdate = "datasource_update"
    Error = "error"


class Event:
    event_type: EventType
    data: Any
    orig: Any

    def __init__(self, event_type: EventType, orig: Any, data: Any | None = None):
        self.event_type = event_type
        self.orig = orig
        self.data = data


class EventHandler(ABC):
    """Base class for event handling."""

    def __setattr__(self, name, value):
        """
        This is to allow more advanced use case where the user does not want to be
        tied to the provided starter class `BaseStrategy`.
        """
        if callable(value) and hasattr(value, "__code__"):
            if value.__class__ == types.MethodType:
                pass
            elif value.__class__ == types.FunctionType and (
                value.__code__.co_argcount > 0
                and value.__code__.co_varnames[0] == "self"
            ):
                value = types.MethodType(value, self)

        super().__setattr__(name, value)

    @abstractmethod
    async def on_event(self, event: Event) -> None:
        """Called when message is parsed and identified."""
        pass

    @abstractmethod
    async def start(self):
        pass

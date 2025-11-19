import math
from datetime import datetime, timedelta
from typing import Callable, Sequence, TypeVar, TypeGuard, Any, cast


def flat_map(f: Callable, xs: list[list]) -> list:
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


def backforward_split(
    start_time: datetime,
    end_time: datetime,
    size: tuple[float, float] | None = None,
    forward_days: int | None = None,
) -> tuple[datetime, datetime, datetime, datetime]:
    duration = end_time - start_time

    if size is not None:
        if size[0] + size[1] != 1.0:
            raise ValueError(f"Size must sum to 1.0, not {size[0] + size[1]}")

        back = timedelta(days=math.ceil(duration.days * size[0]))
        return (start_time, start_time + back, start_time + back, end_time)

    if forward_days is not None:
        if forward_days < 0:
            raise ValueError("forward_days must be non-negative")
        if forward_days > duration.days:
            raise ValueError(
                f"forward_days ({forward_days}) cannot be greater than the total duration ({duration.days} days)"
            )

        return (
            start_time,
            end_time - timedelta(days=forward_days),
            end_time - timedelta(days=forward_days),
            end_time,
        )

    raise ValueError(
        "Either size or forward_days must be provided to split the time range."
    )


C = TypeVar("C", bound=object)


def make_class(
    name: str,
    fields: list[tuple[str, Any]],
    parent: C = object,
    methods: dict[str, Callable] = {},
    static_methods: dict[str, Callable] = {},
    extra_fields: dict[str, Any] = {},
) -> C:
    classmethods = {**methods, **extra_fields, "Parent": parent}

    def make_field(field: tuple[str, Any]) -> str:
        return field[0] if field[1] is None else f"{field[0]} = {field[1]}"

    if len(fields) > 0:
        init_def = f"def __init__(self, {', '.join(map(make_field, fields))}):"
        init_def += "\n    Parent.__init__(self)"
        for field in fields:
            init_def += f"\n    self.{field[0]} = {field[0]}"

        # define and add the __init__ method into classmethods
        exec(init_def, classmethods)

    # add static methods
    for name, func in static_methods.items():
        classmethods[name] = staticmethod(func)

    return cast(C, type(name, (cast(type, parent),), classmethods))


def is_indexable(obj: Any) -> TypeGuard[Sequence[float | None]]:
    return obj is not None and hasattr(obj, "__getitem__") and hasattr(obj, "__len__")


def check_max_hold_duration(
    max_hold_duration: timedelta,
    last_timestamp: datetime,
    current_timestamp: datetime,
    cpos: int,
    ppos: int,
) -> tuple[int, datetime]:
    """
    Check and update the current position and entry timestamp based on the maximum hold duration.

    if `cpos != ppos`, meaning the position has changed, it will update last_timestamp with current_timestamp.

    if `max_hold_duration` is equal to timedelta(0), the function returns the current position unchanged.

    if `current_timestamp - last_timestamp (duration)` is greater than or equal to `max_hold_duration`, the function
    resets the current position to 0.
    """
    last_timestamp = last_timestamp if cpos == ppos else current_timestamp

    if max_hold_duration.total_seconds() == 0:
        return cpos, last_timestamp

    duration = current_timestamp - last_timestamp
    return (
        (0, current_timestamp)
        if duration >= max_hold_duration and ppos == cpos
        else (cpos, last_timestamp)
    )


class aobject(object):
    """
    Async implementation of python's `object`.
    This allows to define Python Classes with `async def __init__(self)`
    """

    async def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        await instance.__init__(*args, **kwargs)
        return instance

    async def __init__(self):
        pass


T = TypeVar("T")


def unwrap_option(opt: T | None) -> T:
    if opt is None:
        raise ValueError("Option is None")
    return opt

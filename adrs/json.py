import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable


class JSONEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle additional types."""

    def default(self, o):
        if isinstance(o, timedelta):
            return {"_type": "timedelta", "seconds": o.total_seconds()}
        elif isinstance(o, datetime):
            return {"_type": "datetime", "isoformat": o.isoformat()}
        elif isinstance(o, Decimal):
            return {"_type": "decimal", "value": str(o)}
        return super().default(o)


class JSONDecoder(json.JSONDecoder):
    """Custom JSON Decoder to handle additional types."""

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=JSONDecoder.custom_object_hook, *args, **kwargs
        )

    @staticmethod
    def custom_object_hook(obj):
        if "_type" in obj:
            match obj["_type"]:
                case "timedelta":
                    return timedelta(seconds=obj["seconds"])
                case "datetime":
                    return datetime.fromisoformat(obj["isoformat"])
                case "decimal":
                    return Decimal(obj["value"])
                case _:
                    raise ValueError(f"Unknown type: {obj['_type']}")
        return obj


def dumps(
    obj: Any,
    *,
    skipkeys: bool = False,
    ensure_ascii: bool = True,
    check_circular: bool = True,
    allow_nan: bool = True,
    cls: type[json.JSONEncoder] | None = JSONEncoder,
    indent: None | int | str = None,
    separators: tuple[str, str] | None = None,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
    **kwds: Any,
) -> Any:
    """Serialize obj to a JSON formatted str using the custom JSONEncoder."""
    return json.dumps(
        obj,
        skipkeys=skipkeys,
        ensure_ascii=ensure_ascii,
        check_circular=check_circular,
        allow_nan=allow_nan,
        cls=cls,
        indent=indent,
        separators=separators,
        default=default,
        sort_keys=sort_keys,
        **kwds,
    )


def loads(
    s: str | bytes | bytearray,
    *,
    cls: type[json.JSONDecoder] | None = JSONDecoder,
    object_hook: Callable[[dict[Any, Any]], Any] | None = None,
    parse_float: Callable[[str], Any] | None = None,
    parse_int: Callable[[str], Any] | None = None,
    parse_constant: Callable[[str], Any] | None = None,
    object_pairs_hook: Callable[[list[tuple[Any, Any]]], Any] | None = None,
    **kwds: Any,
) -> Any:
    """Deserialize s (a str, bytes or bytearray instance containing a JSON document) to a Python object using the custom JSONDecoder."""
    return json.loads(
        s,
        cls=cls,
        object_hook=object_hook,
        parse_float=parse_float,
        parse_int=parse_int,
        parse_constant=parse_constant,
        object_pairs_hook=object_pairs_hook,
        **kwds,
    )

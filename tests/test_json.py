from decimal import Decimal
from datetime import datetime, timedelta

from adrs import json


def test_json_encoder_decoder():
    payload = {
        "int": 1,
        "float": 1.5,
        "str": "test",
        "list": [1, 2, 3],
        "decimal_list": [Decimal("1.1"), Decimal("2.2"), Decimal("3.3")],
        "dict": {
            "a": 1,
            "datetime": datetime.fromisoformat("2023-10-01T12:00:00+00:00"),
            "timedelta": timedelta(days=2, hours=4),
            "decimal": Decimal("5.75"),
        },
        "bool": True,
        "none": None,
        "datetime": datetime.fromisoformat("2023-10-01T12:00:00+00:00"),
        "timedelta": timedelta(days=5, hours=3, minutes=30),
        "decimal": Decimal("10.5"),
    }
    ende_payload = json.loads(json.dumps(payload))
    assert ende_payload == payload

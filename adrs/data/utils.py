import re

from pathlib import Path
from datetime import datetime, timezone

from adrs.types import Topic

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def make_filename(topic: Topic) -> str:
    return (
        (topic.query_params_str() or "data")
        .replace("-", "_")
        .replace("|", "_")
        .replace("?", "_")
        .replace("/", "_")
        .replace("=", "_")
        .replace("&", "_")
    )


def make_filepath(topic: Topic, fmt: str, suffix: str | None = None) -> Path:
    filename = make_filename(topic)
    if suffix:
        filename = f"{filename}_{suffix}"
    path = f"{topic.provider}|{topic.endpoint}".replace("coinglass-v4", "coinglass")
    transformed = path.replace("/", "_").replace("|", "/")
    return Path(transformed) / f"{filename}.{fmt}"


def parse_date_from_filename(filename: str) -> datetime | None:
    m = DATE_RE.search(filename)
    if m:
        try:
            return datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None

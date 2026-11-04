import logging
import polars as pl

from pathlib import Path
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)

from adrs.types import Topic
from adrs.data.datasource import Datasource
from adrs.data.utils import make_filename, make_filepath, parse_date_from_filename


logger = logging.getLogger(__name__)

# How long after a UTC day ends its cached file must have been written to be
# trusted as complete. A file written earlier was fetched while the day was
# still open (or before a slow source, e.g. glassnode, had published its last
# hours) and may be missing bars, so it is downloaded again and overwritten.
DEFAULT_FINAL_AFTER = timedelta(hours=6)


def _generate_date_range(start: datetime, end: datetime) -> list[date]:
    dates, current = [], start.date()
    while current <= end.date():
        dates.append(current)
        current += timedelta(days=1)
    return dates


def is_final(path: Path, day: date, final_after: timedelta) -> bool:
    """Whether a day's cached file can be trusted as complete: it was written at
    least ``final_after`` after that UTC day ended. Judged by the file's
    modification time, so a copied cache must keep mtimes (``cp -p``,
    ``rsync -a``) or every file looks fresh and is downloaded again."""
    day_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return path.stat().st_mtime >= (day_end + final_after).timestamp()


class Cache:
    def __init__(
        self,
        data_path: Path,
        format: str,
        override_existing: bool,
        final_after: timedelta = DEFAULT_FINAL_AFTER,
    ):
        self.data_path: Path = data_path
        self.fmt: str = format
        self.override_existing: bool = override_existing
        self.final_after: timedelta = final_after

    def init(
        self, topic: Topic, start_time: datetime, end_time: datetime
    ) -> tuple[datetime, datetime]:
        filepath = self.data_path / make_filepath(topic, self.fmt)
        parent_dir = filepath.parent
        if not parent_dir.exists():
            logger.debug("[%s] creating directory at %s", topic, parent_dir)
            parent_dir.mkdir(parents=True, exist_ok=True)

        # Floor start, ceil end to day boundaries
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_midnight = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = (
            end_midnight + timedelta(days=1)
            if end_time > end_midnight
            else end_midnight
        )

        if self.override_existing:
            logger.debug("[%s] override is set, skipping check for cache", topic)
            return (start_time, end_time)
        else:
            filename_base = make_filename(topic)
            # A file only counts as downloaded once it is final; a partial day
            # (fetched before the day closed) is fetched again and overwritten.
            downloaded_dates = [
                d.date()
                for f in parent_dir.iterdir()
                if f.is_file() and filename_base in f.name
                if (d := parse_date_from_filename(f.name)) is not None
                if is_final(f, d.date(), self.final_after)
            ]

            needed = _generate_date_range(start_time, end_time - timedelta(days=1))
            filtered = [d for d in needed if d not in downloaded_dates]

            if filtered:
                download_start_time = datetime.combine(
                    min(filtered), datetime.min.time(), tzinfo=timezone.utc
                )
                download_end_time = datetime.combine(
                    max(filtered) + timedelta(days=1),
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
            else:
                download_start_time = end_time
                download_end_time = end_time
            return (download_start_time, download_end_time)

    def _write_file(self, df: pl.DataFrame, path: Path, fmt: str) -> None:
        match fmt:
            case "parquet":
                df.write_parquet(path)
            case "json":
                df.write_json(path)
            case "csv":
                df.write_csv(path)
            case _:
                raise ValueError(f"unsupported format: {fmt}")

    def _read_file(self, path: Path, fmt: str) -> pl.DataFrame:
        match fmt:
            case "parquet":
                return pl.read_parquet(path)
            case "json":
                return pl.read_json(path)
            case "csv":
                return pl.read_csv(path)
            case _:
                raise ValueError(f"unsupported format: {fmt}")

    async def download(
        self,
        datasource: Datasource,
        topic: Topic,
        download_start_time: datetime,
        download_end_time: datetime,
    ) -> int:
        if download_start_time >= download_end_time:
            return 0
        logger.info(
            "[%s] downloading from %s to %s",
            topic,
            download_start_time,
            download_end_time,
        )
        df = await datasource.query_paginated(
            topic=topic,
            start_time=download_start_time,
            end_time=download_end_time,
            flatten=True,
        )
        if df.is_empty():
            return 0

        df = df.with_columns(
            pl.col("start_time").dt.strftime("%Y-%m-%d").alias("date"),
            *[
                pl.col(col).cast(pl.Float64, strict=False)
                for col, dtype in df.schema.items()
                if dtype != pl.Float64
                and col not in ["start_time", "datetime", "date"]
                # Multi-record-per-timestamp topics (full option chains, GEX
                # snapshots) carry a nested List(Struct(...)) column -- not
                # castable to Float64 (structurally, not just a value-level
                # failure strict=False would relax). Leave nested columns
                # as-is; only cast genuinely scalar columns.
                and not isinstance(dtype, (pl.List, pl.Struct))
            ],
        )

        for partition in df.partition_by("date", include_key=True):
            date_str = partition["date"][0]
            partition = partition.drop("date")

            filepath = self.data_path / make_filepath(topic, self.fmt, suffix=date_str)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            logger.debug(
                "[%s] writing %d records to %s", topic, len(partition), filepath
            )
            self._write_file(partition, filepath, self.fmt)

        return len(df)

    async def read(
        self,
        topic: Topic,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        parent_dir = self.data_path / make_filepath(topic, self.fmt).parent
        if not parent_dir.exists():
            raise FileNotFoundError(f"no files available at {parent_dir}")

        filename_base = make_filename(topic)

        entries = sorted(
            [f for f in parent_dir.iterdir() if f.is_file()],
            key=lambda f: parse_date_from_filename(f.name)
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        frames: list[pl.DataFrame] = []

        for f in entries:
            date_dt = parse_date_from_filename(f.name)
            if (
                date_dt is None
                or date_dt.date() < start_time.date()
                or date_dt >= end_time
            ):
                continue
            if filename_base not in f.name:
                continue

            frames.append(self._read_file(f, self.fmt))

        if not frames:
            raise FileNotFoundError("no files read in the output dir")

        return pl.concat(frames, how="diagonal_relaxed").rechunk().sort("start_time")

    async def fetch(
        self,
        datasource: Datasource,
        topic: Topic,
        start_time: datetime,
        end_time: datetime,
    ):
        (download_start_time, download_end_time) = self.init(
            topic=topic,
            start_time=start_time,
            end_time=end_time,
        )
        datapoints = await self.download(
            datasource=datasource,
            topic=topic,
            download_start_time=download_start_time,
            download_end_time=download_end_time,
        )
        logger.info("[%s] downloaded %d datapoints", topic, datapoints)
        read_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        try:
            return await self.read(
                topic=topic, start_time=read_start, end_time=end_time
            )
        except FileNotFoundError:
            return pl.DataFrame()

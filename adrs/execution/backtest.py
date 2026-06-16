import polars as pl

from pathlib import Path
from datetime import datetime, timedelta, timezone

from adrs import Alpha
from adrs.types import Topic
from adrs.performance.evaluator import Evaluator, Datamap


class SignalCache:
    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._memory: dict[str, pl.DataFrame] = {}

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.parquet"

    def get(self, key: str) -> pl.DataFrame | None:
        if key in self._memory:
            return self._memory[key]
        path = self._path(key)
        if path.exists():
            df = pl.read_parquet(path)
            self._memory[key] = df
            return df
        return None

    def set(self, key: str, df: pl.DataFrame) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self._path(key))
        self._memory[key] = df


_cache = SignalCache(Path("output/signal_cache"))


def _min_interval(alphas: list[Alpha]) -> timedelta:
    intervals = [
        iv
        for alpha in alphas
        for info in alpha.data_infos
        if (iv := Topic.from_str(info.topic).interval()) is not None
    ]
    return min(intervals)


def _floor_to_interval(dt: datetime, interval: timedelta) -> datetime:
    interval_s = interval.total_seconds()
    ts = dt.timestamp()
    return datetime.fromtimestamp(ts - (ts % interval_s), tz=timezone.utc)


def _make_key(cache_id: str, start: datetime, end: datetime) -> str:
    return f"{cache_id}__{start.strftime('%Y%m%dT%H%M%S')}__{end.strftime('%Y%m%dT%H%M%S')}"


def generate_signal_df(
    alphas: list[Alpha],
    metadata_df: pl.DataFrame,
    evaluator: Evaluator,
    datamap: Datamap,
    start_time: datetime,
    end_time: datetime,
    cache_id: str = "signals",
) -> pl.DataFrame | None:
    interval = _min_interval(alphas)
    snapped_start = _floor_to_interval(start_time, interval)
    snapped_end = _floor_to_interval(end_time, interval)
    key = _make_key(cache_id, snapped_start, snapped_end)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    signal_df: pl.DataFrame | None = None
    for alpha in alphas:
        alpha_meta = metadata_df.filter(pl.col("custom_id") == alpha.id)
        base_asset = alpha_meta["base_asset"][0]
        _, df = alpha.backtest(
            evaluator=evaluator,
            base_asset=base_asset,
            datamap=datamap,
            start_time=snapped_start,
            end_time=snapped_end,
            fees=alpha_meta["fees"][0],
            price_shift=alpha_meta["shift_backtest_candle_minute"][0],
        )
        alpha_signal = df.select(
            pl.col("start_time"),
            pl.col("signal").alias(alpha.id),
        )
        if signal_df is None:
            signal_df = alpha_signal
        else:
            signal_df = (
                signal_df.join(alpha_signal, on="start_time", how="full", coalesce=True)
                .sort("start_time")
                .with_columns(pl.col(alpha.id).forward_fill())
            )
    if signal_df is not None:
        _cache.set(key, signal_df)

    return signal_df

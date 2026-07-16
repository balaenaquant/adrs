import inspect
import hashlib
import polars as pl

from pathlib import Path
from datetime import datetime, timedelta, timezone

from adrs import Alpha
from adrs.types import Topic
from adrs.utils import infer_interval
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


def _alpha_fingerprint(alphas: list[Alpha], metadata_df: pl.DataFrame) -> str:
    """Digest of everything the cached signal frame depends on besides the
    time range: each alpha's constructor params and the per-alpha backtest
    metadata. Without this, changing an alpha's parameters silently reuses
    signals computed with the old ones."""
    parts = []
    for alpha in sorted(alphas, key=lambda a: a.id):
        try:
            params = {
                name: getattr(alpha, name, None)
                for name in inspect.signature(alpha.__init__).parameters
            }
        except (ValueError, TypeError):
            params = {}
        parts.append(f"{alpha.id}:{sorted(params.items())!r}")
    meta_cols = [
        c
        for c in ("custom_id", "base_asset", "fees", "shift_backtest_candle_minute")
        if c in metadata_df.columns
    ]
    parts.append(repr(metadata_df.sort("custom_id").select(meta_cols).to_dicts()))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _make_key(cache_id: str, fingerprint: str, start: datetime, end: datetime) -> str:
    # v2: signal timestamps are decision times (bar close), see
    # generate_signal_df — must never collide with v1 (bar open) caches
    return f"{cache_id}__v2__{fingerprint}__{start.strftime('%Y%m%dT%H%M%S')}__{end.strftime('%Y%m%dT%H%M%S')}"


def generate_signal_df(
    alphas: list[Alpha],
    metadata_df: pl.DataFrame,
    evaluator: Evaluator,
    datamap: Datamap,
    start_time: datetime,
    end_time: datetime,
    cache_id: str = "signals",
    forward_fill_to_end: bool = False,
) -> pl.DataFrame | None:
    """Build the per-alpha signal matrix.

    Timestamps in the returned frame are DECISION times: each alpha's bar
    labels (bar open) are shifted forward by one alpha bar to the bar close —
    the moment the signal actually became known. A row labeled T is therefore
    actionable from T onward, regardless of the alpha's candle interval. This
    is what makes mixed-cadence portfolios (e.g. 1h + 24h) safe to backtest on
    a fine price grid without lookahead.

    `forward_fill_to_end` (live/OMS only — keep False for backtests so results
    are unchanged): after assembling, forward-fill EVERY alpha column across all
    rows (the incremental join otherwise leaves earlier alphas null at
    timestamps a later alpha introduced) and carry the last decision forward to
    `snapped_end` (~now). This lets a mixed-cadence portfolio (e.g. 1h + 24h)
    present a signal_df whose last row reaches current time, so the OMS isn't
    rejected by the staleness guard while it holds the most recent signal."""
    interval = _min_interval(alphas)
    snapped_start = _floor_to_interval(start_time, interval)
    snapped_end = _floor_to_interval(end_time, interval)
    key = _make_key(
        cache_id, _alpha_fingerprint(alphas, metadata_df), snapped_start, snapped_end
    )
    if forward_fill_to_end:
        key += "__ffe"  # distinct cache namespace so it never bleeds into backtests
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
            execution_delay=timedelta(
                minutes=alpha_meta["shift_backtest_candle_minute"][0]
            ),
            # only the signal column is consumed here
            compute_metrics=False,
        )
        # relabel bar-open labels to decision times (bar close): the signal of
        # bar [T, T+iv) is only known at T+iv
        alpha_interval = infer_interval(df["start_time"])
        alpha_signal = df.select(
            (pl.col("start_time") + alpha_interval).alias("start_time"),
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
    if signal_df is not None and forward_fill_to_end:
        alpha_cols = [a.id for a in alphas]
        # 1) carry every alpha's last signal across ALL rows (the per-join
        #    forward_fill above leaves earlier alphas null at timestamps a later
        #    alpha added).
        signal_df = signal_df.sort("start_time").with_columns(
            [pl.col(c).forward_fill() for c in alpha_cols]
        )
        # 2) extend the last row to snapped_end (~now) so a slow topic whose last
        #    bar is older doesn't make signal_df stale — hold the last decision.
        last_ts = signal_df["start_time"][-1]
        if last_ts is not None and last_ts < snapped_end:
            tail = pl.DataFrame({"start_time": [snapped_end]}).select(
                pl.col("start_time").cast(signal_df["start_time"].dtype)
            )
            signal_df = (
                pl.concat([signal_df, tail], how="diagonal")
                .sort("start_time")
                .with_columns([pl.col(c).forward_fill() for c in alpha_cols])
            )

    if signal_df is not None:
        _cache.set(key, signal_df)

    return signal_df

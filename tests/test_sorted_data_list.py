"""Behaviour of SortedDataList, the in-memory candle store behind Datamap.

Split in two: the CHARACTERIZATION tests pin semantics that already hold and
must survive the columnar refactor untouched; the COLUMNAR tests describe the
new contract — rows stay in Arrow, so `to_df()` is free and a year of 1m
candles costs ~25 MB instead of ~240 MB.
"""

import tracemalloc
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from adrs.types import SortedDataList

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def row(minute: int, **cols) -> dict:
    return {"start_time": BASE + timedelta(minutes=minute), **cols}


def times(sdl: SortedDataList) -> list[datetime]:
    return sdl.to_df()["start_time"].to_list()


# --------------------------------------------------------------------------
# characterization — must hold before and after the refactor
# --------------------------------------------------------------------------


def test_from_dicts_keeps_rows_and_values():
    sdl = SortedDataList([row(0, close=1.0), row(1, close=2.0)])
    assert sdl.to_df()["close"].to_list() == [1.0, 2.0]


def test_append_re_sorts_rows_by_start_time():
    sdl = SortedDataList([row(5, close=5.0)])
    sdl.append(row(1, close=1.0))
    assert times(sdl) == [BASE + timedelta(minutes=1), BASE + timedelta(minutes=5)]
    assert sdl.to_df()["close"].to_list() == [1.0, 5.0]


def test_to_df_labels_naive_start_time_as_utc_milliseconds():
    sdl = SortedDataList([{"start_time": datetime(2024, 1, 1), "close": 1.0}])
    assert sdl.to_df()["start_time"].dtype == pl.Datetime(
        time_unit="ms", time_zone="UTC"
    )


def test_to_df_truncates_sub_millisecond_precision():
    stamp = datetime(2024, 1, 1, 0, 0, 0, 123_456, tzinfo=timezone.utc)
    sdl = SortedDataList([{"start_time": stamp, "close": 1.0}])
    assert sdl.to_df()["start_time"][0] == stamp.replace(microsecond=123_000)


def test_merge_overwrites_overlapping_rows_with_incoming_values():
    sdl = SortedDataList([row(0, close=1.0), row(1, close=2.0)])
    sdl.merge([row(1, close=99.0)])
    assert sdl.to_df()["close"].to_list() == [1.0, 99.0]


def test_merge_appends_new_rows_in_sorted_order():
    sdl = SortedDataList([row(2, close=2.0)])
    sdl.merge([row(0, close=0.0), row(1, close=1.0)])
    assert times(sdl) == [
        BASE,
        BASE + timedelta(minutes=1),
        BASE + timedelta(minutes=2),
    ]


def test_merge_does_not_duplicate_rows_when_pages_overlap():
    sdl = SortedDataList([row(0, close=0.0), row(1, close=1.0)])
    sdl.merge([row(1, close=1.0), row(2, close=2.0)])
    assert len(sdl) == 3


def test_merge_widens_schema_with_columns_only_the_incoming_page_has():
    sdl = SortedDataList([row(0, close=1.0)])
    sdl.merge([row(1, close=2.0, volume=7.0)])
    df = sdl.to_df()
    assert "volume" in df.columns
    assert df["volume"].to_list() == [None, 7.0]


def test_from_df_round_trips_through_to_df():
    df = pl.DataFrame(
        {"start_time": [BASE, BASE + timedelta(minutes=1)], "close": [1.0, 2.0]}
    ).with_columns(pl.col("start_time").dt.cast_time_unit("ms"))
    assert SortedDataList.from_df(df).to_df().equals(df)


def test_len_reports_row_count():
    assert len(SortedDataList([row(0), row(1), row(2)])) == 3


def test_getitem_returns_the_row_at_that_position():
    sdl = SortedDataList([row(0, close=1.0), row(1, close=2.0)])
    assert sdl[0]["close"] == 1.0
    assert sdl[-1]["close"] == 2.0


def test_default_constructor_does_not_share_state_between_instances():
    first = SortedDataList()
    first.append(row(0, close=1.0))
    assert len(SortedDataList()) == 0


# --------------------------------------------------------------------------
# columnar contract — new behaviour the refactor introduces
# --------------------------------------------------------------------------


def test_to_df_returns_the_stored_frame_instead_of_rebuilding_it():
    sdl = SortedDataList([row(i, close=float(i)) for i in range(10)])
    assert sdl.to_df() is sdl.to_df()


def test_from_df_does_not_materialise_rows_as_python_objects():
    n = 50_000
    df = pl.DataFrame(
        {
            "start_time": [BASE + timedelta(minutes=i) for i in range(n)],
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [1.0] * n,
        }
    ).with_columns(pl.col("start_time").dt.cast_time_unit("ms"))

    tracemalloc.start()
    try:
        sdl = SortedDataList.from_df(df)
        sdl.to_df()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Boxing every row as a dict costs ~10x the Arrow frame; anything under 3x
    # proves we never left columnar form.
    assert peak < df.estimated_size() * 3


def test_empty_list_has_an_empty_frame_rather_than_raising():
    assert SortedDataList().to_df().is_empty()


def test_merging_into_an_empty_list_adopts_the_incoming_rows():
    sdl = SortedDataList()
    sdl.merge([row(0, close=1.0)])
    assert sdl.to_df()["close"].to_list() == [1.0]


def test_merging_an_empty_page_leaves_existing_rows_untouched():
    sdl = SortedDataList([row(0, close=1.0)])
    sdl.merge([])
    assert sdl.to_df()["close"].to_list() == [1.0]


def test_merge_keeps_columns_only_the_existing_rows_have():
    sdl = SortedDataList([row(0, close=1.0, volume=7.0)])
    sdl.merge([row(1, close=2.0)])
    df = sdl.to_df()
    assert "volume" in df.columns
    assert df["volume"].to_list() == [7.0, None]


def test_merge_df_accepts_a_frame_directly():
    sdl = SortedDataList([row(0, close=1.0)])
    sdl.merge_df(SortedDataList([row(1, close=2.0)]).to_df())
    assert sdl.to_df()["close"].to_list() == [1.0, 2.0]


def test_merge_df_with_a_row_less_frame_leaves_existing_rows_untouched():
    sdl = SortedDataList([row(0, close=1.0)])
    sdl.merge_df(SortedDataList([row(9, close=9.0)]).to_df().clear())
    assert sdl.to_df()["close"].to_list() == [1.0]


def test_tail_keeps_only_the_most_recent_rows():
    sdl = SortedDataList([row(i, close=float(i)) for i in range(5)])
    sdl.tail(2)
    assert sdl.to_df()["close"].to_list() == [3.0, 4.0]


def test_tail_on_a_shorter_list_keeps_everything():
    sdl = SortedDataList([row(0, close=1.0)])
    sdl.tail(10)
    assert len(sdl) == 1


def test_replace_last_overwrites_the_final_row():
    sdl = SortedDataList([row(0, close=1.0), row(1, close=2.0)])
    sdl.replace_last(row(1, close=99.0))
    assert sdl.to_df()["close"].to_list() == [1.0, 99.0]
    assert len(sdl) == 2


def test_first_and_last_start_time_report_the_range():
    sdl = SortedDataList([row(0), row(7)])
    assert sdl.first_start_time() == BASE
    assert sdl.last_start_time() == BASE + timedelta(minutes=7)


def test_first_and_last_start_time_are_none_when_empty():
    sdl = SortedDataList()
    assert sdl.first_start_time() is None
    assert sdl.last_start_time() is None


def test_filter_time_range_keeps_only_rows_inside_the_half_open_window():
    sdl = SortedDataList([row(i, close=float(i)) for i in range(5)])
    sdl.filter_time_range(BASE + timedelta(minutes=1), BASE + timedelta(minutes=3))
    assert sdl.to_df()["close"].to_list() == [1.0, 2.0]


def test_data_property_still_exposes_rows_as_dicts():
    sdl = SortedDataList([row(0, close=1.0)])
    assert sdl.data == [{"start_time": BASE, "close": 1.0}]


def test_is_empty_reflects_whether_any_rows_are_stored():
    assert SortedDataList().is_empty()
    assert not SortedDataList([row(0)]).is_empty()


@pytest.mark.parametrize("size", [1, 2, 10])
def test_merge_is_idempotent_when_the_same_page_arrives_twice(size):
    page = [row(i, close=float(i)) for i in range(size)]
    sdl = SortedDataList(page)
    sdl.merge(page)
    assert len(sdl) == size
    assert sdl.to_df()["close"].to_list() == [float(i) for i in range(size)]

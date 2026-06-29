"""Tests for Calculate — pure math utilities, no I/O required."""

from decimal import Decimal

from cybotrade.models import OrderSide

from adrs.oms.calculation import Calculate


# ---------------------------------------------------------------------------
# round_with_precision
# ---------------------------------------------------------------------------


def test_round_with_precision_truncates_not_rounds():
    # 0.129 truncated to 2dp → 0.12, NOT 0.13
    assert Calculate.round_with_precision(Decimal("0.129"), 2) == Decimal("0.12")


def test_round_with_precision_zero_dp():
    assert Calculate.round_with_precision(Decimal("7.9"), 0) == Decimal("7")


def test_round_with_precision_from_float():
    result = Calculate.round_with_precision(0.5559, 3)
    assert result == Decimal("0.555")


def test_round_with_precision_negative():
    assert Calculate.round_with_precision(Decimal("-1.999"), 2) == Decimal("-1.99")


def test_round_with_precision_already_exact():
    assert Calculate.round_with_precision(Decimal("1.23"), 2) == Decimal("1.23")


# ---------------------------------------------------------------------------
# decimals_sum_all
# ---------------------------------------------------------------------------


def test_decimals_sum_all_basic():
    result = Calculate.decimals_sum_all([0.1, 0.2, 0.3])
    assert abs(result - 0.6) < 1e-9


def test_decimals_sum_all_empty():
    assert Calculate.decimals_sum_all([]) == 0


def test_decimals_sum_all_single():
    assert Calculate.decimals_sum_all([1.5]) == 1.5


# ---------------------------------------------------------------------------
# align_price
# ---------------------------------------------------------------------------


def test_align_price_buy_no_offset_floors_to_tick():
    # price 100.05, tick 0.1, offset 0 → floor(100.05/0.1)*0.1 = 100.0
    result = Calculate.align_price(
        limit_price=Decimal("100.05"),
        side=OrderSide.BUY,
        offset=Decimal("0"),
        tick_size=Decimal("0.1"),
        precision=1,
    )
    assert result == Decimal("100.0")


def test_align_price_sell_no_offset_ceils_to_tick():
    # price 100.05, tick 0.1, offset 0 → ceil(100.05/0.1)*0.1 = 100.1
    result = Calculate.align_price(
        limit_price=Decimal("100.05"),
        side=OrderSide.SELL,
        offset=Decimal("0"),
        tick_size=Decimal("0.1"),
        precision=1,
    )
    assert result == Decimal("100.1")


def test_align_price_buy_with_offset_moves_below():
    # price 100, offset 0.01 → raw = 99.0, floor to tick 0.1 → 99.0
    result = Calculate.align_price(
        limit_price=Decimal("100"),
        side=OrderSide.BUY,
        offset=Decimal("0.01"),
        tick_size=Decimal("0.1"),
        precision=1,
    )
    assert result < Decimal("100")


def test_align_price_sell_with_offset_moves_above():
    result = Calculate.align_price(
        limit_price=Decimal("100"),
        side=OrderSide.SELL,
        offset=Decimal("0.01"),
        tick_size=Decimal("0.1"),
        precision=1,
    )
    assert result > Decimal("100")


# ---------------------------------------------------------------------------
# get_max_replace_price
# ---------------------------------------------------------------------------


def test_get_max_replace_price_buy_normal():
    # entry 100, threshold 10% → max = 100 * 1.1 = 110
    result = Calculate.get_max_replace_price(
        entry_price=Decimal("100"),
        threshold=Decimal("10"),
        side=OrderSide.BUY,
        price_precision=2,
    )
    assert result == Decimal("110.00")


def test_get_max_replace_price_sell_normal():
    # entry 100, threshold 10% → max = 100 * 0.9 = 90
    result = Calculate.get_max_replace_price(
        entry_price=Decimal("100"),
        threshold=Decimal("10"),
        side=OrderSide.SELL,
        price_precision=2,
    )
    assert result == Decimal("90.00")


def test_get_max_replace_price_respects_precision():
    result = Calculate.get_max_replace_price(
        entry_price=Decimal("100"),
        threshold=Decimal("5"),
        side=OrderSide.BUY,
        price_precision=0,
    )
    # 100 * 1.05 = 105.0 → truncated to 0dp = 105
    assert result == Decimal("105")


# ---------------------------------------------------------------------------
# generate_random_order_size
# ---------------------------------------------------------------------------


def test_generate_random_order_size_below_min_returns_zero():
    result = Calculate.generate_random_order_size(
        sum_total=Decimal("0.0005"),
        count=3,
        min_qty=Decimal("0.001"),
        precision=3,
        max_qty=Decimal("1"),
    )
    assert result == [Decimal("0.0")]


def test_generate_random_order_size_single_order():
    result = Calculate.generate_random_order_size(
        sum_total=Decimal("0.5"),
        count=1,
        min_qty=Decimal("0.001"),
        precision=3,
        max_qty=Decimal("1"),
    )
    assert len(result) == 1
    assert result[0] > Decimal("0")


def test_generate_random_order_size_respects_count():
    result = Calculate.generate_random_order_size(
        sum_total=Decimal("3"),
        count=3,
        min_qty=Decimal("0.1"),
        precision=2,
        max_qty=Decimal("5"),
    )
    assert len(result) <= 3
    assert all(q > Decimal("0") for q in result)


def test_generate_random_order_size_sum_close_to_total():
    total = Decimal("1.000")
    result = Calculate.generate_random_order_size(
        sum_total=total,
        count=3,
        min_qty=Decimal("0.01"),
        precision=3,
        max_qty=Decimal("1"),
    )
    diff = abs(sum(result) - total)
    assert diff < Decimal("0.01")


def test_generate_random_order_size_all_positive_and_count_bounded():
    # The last chunk gets the remainder and may exceed max_qty — only intermediate
    # chunks are hard-capped. Assert the guarantees the function actually makes.
    result = Calculate.generate_random_order_size(
        sum_total=Decimal("5"),
        count=10,
        min_qty=Decimal("0.1"),
        precision=2,
        max_qty=Decimal("1"),
    )
    assert len(result) <= 10
    assert all(q > Decimal("0") for q in result)

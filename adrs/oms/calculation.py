import math
import random
import logging

import numpy as np

from decimal import Decimal, getcontext

from cybotrade.models import OrderSide

logger = logging.getLogger(__name__)

getcontext().prec = 28


class Calculate:
    @staticmethod
    def decimals_sum_all(array):
        decimal_array = [Decimal(str(f)) for f in array]
        return float(sum(decimal_array))

    @staticmethod
    def round_with_precision(number: Decimal, decimal_places: int) -> Decimal:
        """
        Truncate a decimal to the specified number of decimal places.

        Args:
            number (Decimal | float): The number to truncate.
            decimal_places (int): The number of decimal places to keep.

        Returns:
            Decimal: The truncated number.
        """
        number = Decimal(str(number)) if isinstance(number, float) else number

        factor = Decimal(str(10)) ** Decimal(str(decimal_places))
        try:
            return math.trunc(factor * number) / factor
        except Exception as e:
            logger.error(
                f"[ROUND_WITH_PRECISION] Failed to round with precision due to: {e}"
            )
            return Decimal("0.0")

    @staticmethod
    def get_max_replace_price(
        entry_price: Decimal, threshold: Decimal, side: OrderSide, price_precision: int
    ):
        coeff = Decimal(str(1 + threshold / 100))
        coeff1 = Decimal(str(1 - threshold / 100))
        if side == OrderSide.BUY:
            if threshold == str(100.0):
                max_price = entry_price * Decimal(str(10.0))
            else:
                max_price = entry_price * coeff
        else:
            if threshold == str(100.0):
                max_price = entry_price / Decimal(str(10.0))
            else:
                max_price = entry_price * coeff1

        return Calculate.round_with_precision(max_price, price_precision)

    @staticmethod
    def generate_random_order_size(
        sum_total: Decimal,
        count: int,
        min_qty: Decimal,
        precision: int,
        max_qty: Decimal,
    ) -> list[Decimal]:
        random_values: list[Decimal] = []

        # If total is less than min_qty, just return sum_total as one qty
        if sum_total < min_qty:
            random_values.append(Decimal("0.0"))
            logger.info(f"[RANDOM_QTY] sum_total <= min_qty, qty: {random_values}")
            return random_values

        # Calculate maximum possible count given sum_total and min_qty
        max_possible_count = int(
            (sum_total / min_qty).to_integral_value(rounding="ROUND_FLOOR")
        )
        actual_count = min(count, max_possible_count)

        if actual_count == 0:
            # sum_total smaller than min_qty, return as one qty
            random_values.append(Decimal("0.0"))
            return random_values

        remaining_total = sum_total

        for i in range(actual_count - 1):
            # Maximum for current allocation ensuring enough left for min_qty for remaining
            max_for_this = min(
                max_qty,
                (remaining_total - min_qty * (actual_count - i - 1)).copy_abs(),
            )

            # Minimum for current allocation is min_qty or zero if remaining_total < min_qty
            min_for_this = min_qty if remaining_total >= min_qty else remaining_total

            # If max_for_this < min_for_this, set max_for_this = min_for_this to avoid invalid uniform range
            if max_for_this < min_for_this:
                max_for_this = min_for_this

            random_float = np.random.uniform(float(min_for_this), float(max_for_this))
            qty = Decimal(str(random_float)).quantize(Decimal(f"1e-{precision}"))

            # Clamp qty between 0 and max_qty and not more than remaining_total
            qty = max(min_qty, min(qty, min(max_qty, remaining_total)))

            random_values.append(qty)
            remaining_total -= qty

            # Avoid negative remaining_total
            if remaining_total < 0:
                remaining_total = Decimal("0")
                break

        # Append the last qty as the remainder (or zero if negative)
        last_qty = max(min_qty, remaining_total).quantize(Decimal(f"1e-{precision}"))
        random_values.append(last_qty)

        # Remove any zeros if sum_total is very small or rounding causes zeros
        random_values = [qty for qty in random_values if qty > 0]

        # Adjust sum to exactly sum_total by tweaking first qty
        current_sum = sum(random_values)
        diff = sum_total - current_sum

        if random_values and diff != 0:
            adjusted = random_values[0] + diff
            if adjusted > 0 and adjusted <= max_qty:
                random_values[0] = adjusted.quantize(Decimal(f"1e-{precision}"))

        logger.debug(f"[RANDOM_QTY] final random_qty : {random_values}")

        return random_values

    @staticmethod
    def generate_multi_price_level(
        level: int, tick_size: Decimal, precision: int, price: Decimal, side: OrderSide
    ) -> list[list[Decimal]]:
        all_price_level: list[list[Decimal]] = []
        for i in range(0, level):
            random_multi = random.choice(
                [
                    Decimal("0.0001"),
                    Decimal("0.0002"),
                    Decimal("0.0003"),
                    Decimal("0.0004"),
                    Decimal("0.0005"),
                ]
            )
            if side == OrderSide.BUY:
                raw_price = price * (1 - i * random_multi)
                aligned_price = math.floor(raw_price / tick_size) * tick_size
                aligned_price = Calculate.round_with_precision(aligned_price, precision)
                all_price_level.append([aligned_price, 1 - i * random_multi])
            else:
                raw_price = price * (1 + i * random_multi)
                aligned_price = math.ceil(raw_price / tick_size) * tick_size
                aligned_price = Calculate.round_with_precision(aligned_price, precision)
                all_price_level.append([aligned_price, 1 + i * random_multi])
        all_price_level = sorted(
            all_price_level, key=lambda x: x[0], reverse=(side == OrderSide.BUY)
        )

        return all_price_level

    @staticmethod
    def align_price(
        limit_price: Decimal,
        side: OrderSide,
        offset: Decimal,
        tick_size: Decimal,
        precision: int,
    ) -> Decimal:
        if side == OrderSide.BUY:
            raw_price = limit_price * (1 - offset)
            aligned_price = math.floor(raw_price / tick_size) * tick_size
        else:
            raw_price = limit_price * (1 + offset)
            aligned_price = math.ceil(raw_price / tick_size) * tick_size
        return Calculate.round_with_precision(aligned_price, precision)

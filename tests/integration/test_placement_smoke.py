"""Smoke test: prove the real placement stack drives the FakeExchange end to end."""

import asyncio
from decimal import Decimal

from cybotrade import Symbol

from tests.integration.harness import assert_conserved, build_opm, set_desired


def test_single_buy_places_one_order_and_conserves():
    opm, sim, _ = build_opm(["BTCUSDT"])
    set_desired(opm, "BTCUSDT", "0.2")  # exchange=0, no orders, no backlog

    asyncio.run(opm.on_order_placement())

    placed = [c for c in sim.calls if c[0] == "place"]
    assert len(placed) == 1
    assert len(sim.open_orders) == 1
    assert sim.positions[Symbol("BTCUSDT")] == Decimal("0")  # resting, not filled
    assert_conserved(opm, sim)


def test_zero_delta_places_nothing():
    opm, sim, _ = build_opm(["BTCUSDT"])
    set_desired(opm, "BTCUSDT", "0")

    asyncio.run(opm.on_order_placement())

    assert [c for c in sim.calls if c[0] == "place"] == []
    assert_conserved(opm, sim)

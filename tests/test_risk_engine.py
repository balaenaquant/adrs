"""Tests for the base RiskEngine — hooks are no-ops; real controls live in
subclasses. These verify the base contract: cap_desired passes desired through
unchanged, and run_risk_checks does nothing (never trips on_breach)."""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cybotrade import Symbol
from cybotrade.models import Position

from adrs.oms.risk import RiskEngine, RiskConfig


def _pos(symbol: str, quantity: str, price: str) -> Position:
    return Position(
        symbol=Symbol(symbol),
        quantity=Decimal(quantity),
        entry_price=Decimal(price),
        updated_time=datetime.now(timezone.utc),
    )


def _risk(*, balance="10000") -> RiskEngine:
    config = SimpleNamespace(
        config=SimpleNamespace(initial_balance=Decimal(balance)),
        exchange=MagicMock(),
    )
    return RiskEngine(
        config=config,
        position=MagicMock(),
        rate_limiter=MagicMock(),
        on_breach=AsyncMock(),
        risk=RiskConfig(),
    )


# ---------------------------------------------------------------------------
# cap_desired — no-op gate
# ---------------------------------------------------------------------------


def test_cap_desired_returns_input_unchanged():
    engine = _risk()
    desired = {
        Symbol("BTCUSDT"): _pos("BTCUSDT", "0.3", "100000"),
        Symbol("ETHUSDT"): _pos("ETHUSDT", "3.75", "4000"),
    }
    result = engine.cap_desired(desired)
    assert result is desired  # base is a pass-through


def test_cap_desired_empty():
    engine = _risk()
    desired: dict = {}
    assert engine.cap_desired(desired) is desired


# ---------------------------------------------------------------------------
# run_risk_checks — no-op monitor
# ---------------------------------------------------------------------------


def test_run_risk_checks_is_noop():
    engine = _risk()
    asyncio.run(engine.run_risk_checks())  # must not raise
    engine.on_breach.assert_not_awaited()


def test_run_risk_checks_never_fetches_wallet():
    # Base does nothing, so it must not touch the exchange
    engine = _risk()
    asyncio.run(engine.run_risk_checks())
    engine.config.exchange.get_wallet_balance.assert_not_called()


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_riskconfig_defaults():
    cfg = RiskConfig()
    assert cfg.nav_floor == Decimal("0.60")
    assert cfg.max_exposure == Decimal("2.0")


def test_engine_uses_default_riskconfig_when_none():
    config = SimpleNamespace(config=SimpleNamespace(initial_balance=Decimal("1")))
    engine = RiskEngine(
        config=config,
        position=MagicMock(),
        rate_limiter=MagicMock(),
        on_breach=AsyncMock(),
    )
    assert engine.risk.max_exposure == Decimal("2.0")

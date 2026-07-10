"""Unit tests for OMS core handlers.

Uses object.__new__ to bypass __init__ and inject only the state each test
needs — same pattern as test_order_executer.py.
"""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cybotrade import Symbol
from cybotrade.models import OrderSide, OrderStatus

from adrs.oms.ops.order_pool import CancelBacklogs
from adrs.oms.oms import OMS, PortfolioSignal, generate_cron


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oms() -> OMS:
    """OMS with __init__ skipped — set only what each test needs."""
    return object.__new__(OMS)


def _signal(assets: dict, timestamp: int = 1000) -> PortfolioSignal:
    return PortfolioSignal(
        assets={k: Decimal(str(v)) for k, v in assets.items()},
        timestamp=timestamp,
    )


def _msg(payload: dict):
    return SimpleNamespace(data=json.dumps(payload).encode())


def _async_cm(return_value):
    """Minimal async context manager that yields return_value."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class _RecordingObserver:
    def __init__(self):
        self.signals_received = []
        self.desired_updates = []
        self.skipped = []

    def on_signal_received(self, sig):
        self.signals_received.append(sig)

    def on_desired_updated(self, sym, qty):
        self.desired_updates.append((sym, qty))

    def on_signal_skipped(self, sym, reason):
        self.skipped.append((sym, reason))


# ---------------------------------------------------------------------------
# generate_cron  (pure function — no mocking needed)
# ---------------------------------------------------------------------------


def test_generate_cron_sub_second_returns_every_tick():
    assert generate_cron(0) == "* * * * * *"


def test_generate_cron_seconds_only():
    assert generate_cron(5) == "*/5 * * * * *"


def test_generate_cron_minutes():
    # 120s → every 2 minutes, at second 0
    assert generate_cron(120) == "0 */2 * * * *"


def test_generate_cron_hours():
    # 7200s → every 2 hours, at minute 0 second 0
    assert generate_cron(7200) == "0 0 */2 * * *"


def test_generate_cron_mixed_hms():
    # 3661s = 1h 1m 1s → hours branch (1 1 */1 * * *)
    assert generate_cron(3661) == "1 1 */1 * * *"


# ---------------------------------------------------------------------------
# on_portfolio_signal
# ---------------------------------------------------------------------------


def test_on_portfolio_signal_stores_valid_signal():
    oms = _oms()
    oms.latest_signal = None
    oms.observer = None

    asyncio.run(
        oms.on_portfolio_signal(_msg({"assets": {"BTC": "0.5"}, "timestamp": "1"}))
    )

    assert oms.latest_signal is not None
    assert oms.latest_signal.assets["BTC"] == Decimal("0.5")


def test_on_portfolio_signal_rejects_sum_over_one():
    oms = _oms()
    oms.latest_signal = None
    oms.observer = None

    asyncio.run(
        oms.on_portfolio_signal(
            _msg({"assets": {"BTC": "0.7", "ETH": "0.6"}, "timestamp": "1"})
        )
    )

    assert oms.latest_signal is None


def test_on_portfolio_signal_calls_observer_on_valid_signal():
    obs = _RecordingObserver()
    oms = _oms()
    oms.latest_signal = None
    oms.observer = obs

    asyncio.run(
        oms.on_portfolio_signal(_msg({"assets": {"BTC": "0.5"}, "timestamp": "1"}))
    )

    assert len(obs.signals_received) == 1
    assert obs.signals_received[0].assets["BTC"] == Decimal("0.5")


def test_on_portfolio_signal_does_not_call_observer_when_rejected():
    obs = _RecordingObserver()
    oms = _oms()
    oms.latest_signal = None
    oms.observer = obs

    asyncio.run(
        oms.on_portfolio_signal(
            _msg({"assets": {"BTC": "0.7", "ETH": "0.6"}, "timestamp": "1"})
        )
    )

    assert obs.signals_received == []


def test_on_portfolio_signal_tolerates_bad_payload():
    oms = _oms()
    oms.latest_signal = None
    oms.observer = None

    asyncio.run(oms.on_portfolio_signal(SimpleNamespace(data=b"not-json")))

    assert oms.latest_signal is None  # must not raise, must not store


# ---------------------------------------------------------------------------
# on_process_latest_signal
# ---------------------------------------------------------------------------


def _oms_for_signal_processing(
    signal, previous=None, observer=None, market_price="50000"
):
    oms = _oms()
    oms.latest_signal = signal
    oms.previous_signal = previous
    oms.observer = observer

    backlog = []
    opm = MagicMock()
    opm.order_pools.get_order_backlog = MagicMock(return_value=_async_cm(backlog))
    # Signal processing fetches the quote via the executor's canonical helper
    opm.executor.get_current_price = AsyncMock(return_value=Decimal(market_price))
    oms.opm = opm

    rate_limiter = MagicMock()
    rate_limiter.guard = MagicMock(return_value=_async_cm(None))
    oms.rate_limiter = rate_limiter

    exchange = MagicMock()
    exchange.get_current_price = AsyncMock(return_value=Decimal(market_price))

    position = MagicMock()
    position.compute_base_quantity = MagicMock(return_value=Decimal("0.2"))
    position.desired = {}
    oms.position = position

    oms.config = SimpleNamespace(
        config=SimpleNamespace(
            portfolio_id="p1",
            base_asset_to_symbol_table={"BTC": "BTCUSDT"},
        ),
        exchange=exchange,
    )

    return oms


def test_on_process_latest_signal_no_signal_is_noop():
    oms = _oms_for_signal_processing(None)
    asyncio.run(oms.on_process_latest_signal())  # must not raise


def test_on_process_latest_signal_updates_desired_position():
    sig = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig)
    asyncio.run(oms.on_process_latest_signal())

    sym = Symbol("BTCUSDT")
    assert sym in oms.position.desired
    pos = oms.position.desired[sym]
    assert pos.quantity == Decimal("0.2")
    assert pos.entry_price == Decimal("50000")


def test_on_process_latest_signal_skips_unchanged_weight_4dp():
    # 0.50001 vs 0.50002 differ only at 5th decimal — both round to 0.5000 at 4dp
    obs = _RecordingObserver()
    sig = _signal({"BTC": "0.50001"})
    prev = _signal({"BTC": "0.50002"})
    oms = _oms_for_signal_processing(sig, previous=prev, observer=obs)
    asyncio.run(oms.on_process_latest_signal())

    assert obs.skipped == [(Symbol("BTCUSDT"), "no_change")]
    assert Symbol("BTCUSDT") not in oms.position.desired


def test_on_process_latest_signal_processes_materially_changed_weight():
    obs = _RecordingObserver()
    sig = _signal({"BTC": "0.7"})
    prev = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig, previous=prev, observer=obs)
    asyncio.run(oms.on_process_latest_signal())

    assert len(obs.desired_updates) == 1
    sym, qty = obs.desired_updates[0]
    assert str(sym) == "BTCUSDT"
    assert qty == Decimal("0.2")


def test_on_process_latest_signal_clears_backlog_on_weight_change():
    sig = _signal({"BTC": "0.7"})
    prev = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig, previous=prev)

    backlog = [CancelBacklogs(symbol="BTCUSDT", total_retries=1, client_order_id="old")]
    oms.opm.order_pools.get_order_backlog = MagicMock(return_value=_async_cm(backlog))

    asyncio.run(oms.on_process_latest_signal())

    assert backlog == []


def test_on_process_latest_signal_new_asset_has_no_previous():
    # asset not in previous signal → weightage comparison skipped, position updated
    obs = _RecordingObserver()
    sig = _signal({"BTC": "0.5"})
    prev = _signal({})  # BTC not present before
    oms = _oms_for_signal_processing(sig, previous=prev, observer=obs)
    asyncio.run(oms.on_process_latest_signal())

    assert len(obs.desired_updates) == 1


def test_on_process_latest_signal_stores_signal_as_previous():
    sig = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig)
    asyncio.run(oms.on_process_latest_signal())
    assert oms.previous_signal is sig


# ---------------------------------------------------------------------------
# _handle_shutdown
# ---------------------------------------------------------------------------


def test_handle_shutdown_empty_pool_raises_system_exit():
    oms = _oms()
    opm = MagicMock()
    opm.order_pools.get_order_pool = MagicMock(return_value=_async_cm({}))
    oms.opm = opm

    with pytest.raises(SystemExit) as exc:
        asyncio.run(oms._handle_shutdown())
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# _sync_trade_record
# ---------------------------------------------------------------------------


def _make_order_result(
    *, filled_size, status, side=OrderSide.BUY, size=None, price="100"
):
    return SimpleNamespace(
        filled_size=Decimal(str(filled_size)),
        status=status,
        side=side,
        size=Decimal(str(size or filled_size)),
        price=Decimal(price),
        updated_time=datetime.now(timezone.utc),
    )


def _oms_for_sync(exchange_result, metric_raises=False):
    oms = _oms()

    rate_limiter = MagicMock()
    rate_limiter.guard = MagicMock(return_value=_async_cm(None))
    oms.rate_limiter = rate_limiter

    exchange = MagicMock()
    exchange.get_order_details_from_history = AsyncMock(return_value=exchange_result)
    exchange.exchange = MagicMock(return_value="bybit_linear")
    oms.config = SimpleNamespace(exchange=exchange)

    metric_builder = MagicMock()
    if metric_raises:
        metric_builder.create_trade = AsyncMock(side_effect=RuntimeError("db down"))
    else:
        metric_builder.create_trade = AsyncMock()
    oms.metric_builder = metric_builder

    return oms


def _record():
    return (Symbol("BTCUSDT"), "coid-1", Decimal("50000"), datetime.now(timezone.utc))


def test_sync_trade_record_not_found_keeps_in_pool():
    oms = _oms_for_sync(None)
    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", _record())
    )
    assert result is None


def test_sync_trade_record_zero_fill_cancelled_retires_record():
    order = _make_order_result(filled_size=0, status=OrderStatus.CANCELLED)
    oms = _oms_for_sync(order)
    record = _record()
    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", record)
    )
    assert result == ("pkg1", record)


def test_sync_trade_record_zero_fill_rejected_retires_record():
    order = _make_order_result(filled_size=0, status=OrderStatus.REJECTED)
    oms = _oms_for_sync(order)
    record = _record()
    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", record)
    )
    assert result == ("pkg1", record)


def test_sync_trade_record_zero_fill_still_live_keeps_in_pool():
    order = _make_order_result(filled_size=0, status=OrderStatus.CREATED)
    oms = _oms_for_sync(order)
    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", _record())
    )
    assert result is None


def test_sync_trade_record_filled_calls_metric_builder_and_retires():
    order = _make_order_result(filled_size="1.0", size="1.0", status=OrderStatus.FILLED)
    oms = _oms_for_sync(order)
    record = _record()
    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", record)
    )

    assert result == ("pkg1", record)
    oms.metric_builder.create_trade.assert_awaited_once()


def test_sync_trade_record_sell_side_uses_negative_quantity():
    order = _make_order_result(
        filled_size="1.0", size="1.0", status=OrderStatus.FILLED, side=OrderSide.SELL
    )
    oms = _oms_for_sync(order)
    asyncio.run(oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", _record()))

    call_kwargs = oms.metric_builder.create_trade.call_args.kwargs
    assert call_kwargs["executed_quantity"].startswith("-")


def test_sync_trade_record_metric_failure_keeps_in_pool():
    order = _make_order_result(filled_size="1.0", size="1.0", status=OrderStatus.FILLED)
    oms = _oms_for_sync(order, metric_raises=True)
    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", _record())
    )
    assert result is None


def test_sync_trade_record_exchange_error_keeps_in_pool():
    oms = _oms()
    rate_limiter = MagicMock()
    rate_limiter.guard = MagicMock(return_value=_async_cm(None))
    oms.rate_limiter = rate_limiter

    exchange = MagicMock()
    exchange.get_order_details_from_history = AsyncMock(
        side_effect=RuntimeError("timeout")
    )
    oms.config = SimpleNamespace(exchange=exchange)
    oms.metric_builder = MagicMock()

    result = asyncio.run(
        oms._sync_trade_record(asyncio.Semaphore(1), "oms1", "pkg1", _record())
    )
    assert result is None


# ---------------------------------------------------------------------------
# on_command (control-plane dispatch)
# ---------------------------------------------------------------------------


def test_on_command_routes_rebalance():
    oms = _oms()
    oms.rebalance = AsyncMock()
    asyncio.run(oms.on_command(_msg({"command": "rebalance"})))
    oms.rebalance.assert_awaited_once()


def test_on_command_unknown_command_is_noop():
    oms = _oms()
    oms.rebalance = AsyncMock()
    asyncio.run(oms.on_command(_msg({"command": "bogus"})))
    oms.rebalance.assert_not_awaited()


def test_on_command_bad_payload_is_swallowed():
    oms = _oms()
    oms.rebalance = AsyncMock()
    # malformed body must be logged and swallowed, never raised
    asyncio.run(oms.on_command(SimpleNamespace(data=b"not json")))
    oms.rebalance.assert_not_awaited()

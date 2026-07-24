"""Unit tests for OMS core handlers.

Uses object.__new__ to bypass __init__ and inject only the state each test
needs — same pattern as test_order_executer.py.
"""

import asyncio
import json
import signal
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cybotrade import Symbol
from cybotrade.models import OrderSide, OrderStatus, Position

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


def _signal_config(table=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            base_asset_to_symbol_table=table or {"BTC": "BTCUSDT"},
        )
    )


def test_on_portfolio_signal_stores_valid_signal():
    oms = _oms()
    oms.latest_signal = None
    oms.observer = None
    oms.config = _signal_config()

    asyncio.run(
        oms.on_portfolio_signal(_msg({"assets": {"BTC": "0.5"}, "timestamp": "1"}))
    )

    assert oms.latest_signal is not None
    assert oms.latest_signal.assets["BTC"] == Decimal("0.5")


def test_on_portfolio_signal_rejects_unknown_asset():
    # Regression: an asset missing from base_asset_to_symbol_table used to be
    # stored, then KeyError'd every _recompute_desired tick forever (the poison
    # signal stayed in latest_signal). Must be rejected at receipt.
    obs = _RecordingObserver()
    oms = _oms()
    oms.latest_signal = None
    oms.observer = obs
    oms.config = _signal_config()

    asyncio.run(
        oms.on_portfolio_signal(
            _msg({"assets": {"BTC": "0.3", "DOGE": "0.2"}, "timestamp": "1"})
        )
    )

    assert oms.latest_signal is None
    assert obs.signals_received == []


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
    oms.config = _signal_config()

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

    # Risk cap is exercised in test_risk_engine; here it passes desired through
    risk = MagicMock()
    risk.cap_desired = lambda desired: desired
    oms.risk = risk

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


def test_on_process_latest_signal_keeps_backlog_of_unrelated_symbols():
    # Regression: the backlog clear was global (and ran once per changed
    # asset), wiping cancel/expiry retries for symbols whose desired was NOT
    # being recomputed - nothing re-derived those until the expiry pass.
    sig = _signal({"BTC": "0.7"})
    prev = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig, previous=prev)

    btc_entry = CancelBacklogs(symbol="BTCUSDT", total_retries=1, client_order_id="btc")
    eth_entry = CancelBacklogs(symbol="ETHUSDT", total_retries=1, client_order_id="eth")
    backlog = [btc_entry, eth_entry]
    oms.opm.order_pools.get_order_backlog = MagicMock(return_value=_async_cm(backlog))

    asyncio.run(oms.on_process_latest_signal())

    assert backlog == [eth_entry]  # recomputed symbol cleared, unrelated kept


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
# _recompute_desired / rebalance  (force=True bypasses the weight-unchanged skip)
# ---------------------------------------------------------------------------


def test_recompute_desired_force_reprices_unchanged_weight():
    # Same weight prev==latest: force=True must NOT skip, must reprice.
    obs = _RecordingObserver()
    sig = _signal({"BTC": "0.5"})
    prev = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig, previous=prev, observer=obs)

    asyncio.run(oms._recompute_desired(force=True))

    assert obs.skipped == []  # unchanged weight was NOT skipped
    assert len(obs.desired_updates) == 1
    assert Symbol("BTCUSDT") in oms.position.desired


def test_rebalance_reprices_static_signal():
    # rebalance() forces a recompute even when the signal never changed.
    sig = _signal({"BTC": "0.5"})
    prev = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig, previous=prev)

    asyncio.run(oms.rebalance())

    sym = Symbol("BTCUSDT")
    assert sym in oms.position.desired
    assert oms.position.desired[sym].quantity == Decimal("0.2")


def test_recompute_desired_price_none_aborts_cycle():
    # No price for the symbol → whole cycle aborts, desired untouched.
    sig = _signal({"BTC": "0.5"})
    oms = _oms_for_signal_processing(sig)
    oms.opm.executor.get_current_price = AsyncMock(return_value=None)

    asyncio.run(oms._recompute_desired(force=True))

    assert oms.position.desired == {}


# ---------------------------------------------------------------------------
# _handle_shutdown
# ---------------------------------------------------------------------------


def _shutdown_order(symbol="BTCUSDT", client_order_id="coid-1"):
    return SimpleNamespace(symbol=symbol, client_order_id=client_order_id)


def _oms_for_shutdown(order_pool, cancel_result=None, cancel_side_effect=None):
    oms = _oms()
    opm = MagicMock()
    opm.order_pools.get_order_pool = MagicMock(return_value=_async_cm(order_pool))
    if cancel_side_effect is not None:
        opm.executor.cancel_single_order = AsyncMock(side_effect=cancel_side_effect)
    else:
        opm.executor.cancel_single_order = AsyncMock(return_value=cancel_result)
    oms.opm = opm
    return oms


def test_handle_shutdown_empty_pool_raises_system_exit():
    oms = _oms_for_shutdown({})
    with pytest.raises(SystemExit) as exc:
        asyncio.run(oms._handle_shutdown())
    assert exc.value.code == 0


def test_handle_shutdown_cancels_all_orders_no_retry():
    # Every cancel succeeds (returns None, no backlog) → completes, no SystemExit.
    pool = {"coid-1": _shutdown_order(client_order_id="coid-1"),
            "coid-2": _shutdown_order(client_order_id="coid-2")}
    oms = _oms_for_shutdown(pool, cancel_result=None)

    asyncio.run(oms._handle_shutdown())  # returns normally

    assert oms.opm.executor.cancel_single_order.await_count == 2


def test_handle_shutdown_cancel_exception_logged_still_completes():
    # One cancel raises; return_exceptions keeps the pass going, no retry queued.
    pool = {"coid-1": _shutdown_order(client_order_id="coid-1"),
            "coid-2": _shutdown_order(client_order_id="coid-2")}
    oms = _oms_for_shutdown(pool, cancel_side_effect=RuntimeError("boom"))

    asyncio.run(oms._handle_shutdown())  # exception swallowed, returns

    assert oms.opm.executor.cancel_single_order.await_count == 2


def test_handle_shutdown_cancel_backlog_triggers_retry_then_exits():
    # cancel returns CancelBacklogs → retry path: sleep, re-cancel, SystemExit(0).
    backlog = CancelBacklogs(symbol="BTCUSDT", total_retries=1, client_order_id="coid-1")
    pool = {"coid-1": _shutdown_order(client_order_id="coid-1")}
    oms = _oms_for_shutdown(pool, cancel_result=backlog)

    with patch("adrs.oms.oms.asyncio.sleep", new=AsyncMock()) as slept:
        with pytest.raises(SystemExit) as exc:
            asyncio.run(oms._handle_shutdown())

    assert exc.value.code == 0
    slept.assert_awaited_once()
    # one initial cancel + one retry cancel
    assert oms.opm.executor.cancel_single_order.await_count == 2


def test_handle_shutdown_cancelled_during_retry_wait_returns_early():
    # CancelledError during the 60s wait → bail out, no retry, no SystemExit.
    backlog = CancelBacklogs(symbol="BTCUSDT", total_retries=1, client_order_id="coid-1")
    pool = {"coid-1": _shutdown_order(client_order_id="coid-1")}
    oms = _oms_for_shutdown(pool, cancel_result=backlog)

    with patch(
        "adrs.oms.oms.asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)
    ):
        asyncio.run(oms._handle_shutdown())  # returns, no SystemExit

    # only the initial cancel ran; retry never happened
    assert oms.opm.executor.cancel_single_order.await_count == 1


# ---------------------------------------------------------------------------
# _setup_signals
#
# SIGTERM must stop the scheduler, NOT call _handle_shutdown() directly -
# run()'s finally already calls it exactly once whenever scheduler.start()
# returns. Calling it from the signal handler too would race that (a second
# concurrent order-cancel pass), and a SystemExit raised inside a
# fire-and-forget create_task() wouldn't propagate to end the process
# anyway - it would just be a swallowed background-task exception.
# ---------------------------------------------------------------------------


def test_setup_signals_sigterm_stops_scheduler_not_handle_shutdown_directly():
    oms = _oms()
    oms.scheduler = MagicMock()
    oms.scheduler.shutdown = AsyncMock()
    oms._handle_shutdown = AsyncMock()

    async def run():
        loop = asyncio.get_event_loop()
        with patch.object(loop, "add_signal_handler") as add_handler:
            oms._setup_signals()
        assert add_handler.call_count == 1
        registered_signal, callback = add_handler.call_args[0]
        assert registered_signal is signal.SIGTERM
        await callback()  # callback() returns the create_task() Task

    asyncio.run(run())
    oms.scheduler.shutdown.assert_awaited_once()
    oms._handle_shutdown.assert_not_awaited()


def test_trigger_shutdown_stops_scheduler_not_handle_shutdown_directly():
    # Regression: on_breach used to call _handle_shutdown() directly. When all
    # cancels succeeded it returned without stopping the scheduler, so the
    # next placement tick re-placed every order the breach just cancelled.
    oms = _oms()
    oms.scheduler = MagicMock()
    oms.scheduler.shutdown = AsyncMock()
    oms._handle_shutdown = AsyncMock()

    asyncio.run(oms._trigger_shutdown())

    oms.scheduler.shutdown.assert_awaited_once()
    oms._handle_shutdown.assert_not_awaited()


# ---------------------------------------------------------------------------
# on_aegis_update
#
# The trade-sync logic lives in the nested _check_and_sync_trade closure, so it
# is exercised only through on_aegis_update. A record is "retired" when it is
# gone from order_records after the run, "kept" when it survives. on_aegis_update
# also upserts equity and position on every run.
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


def _record(client_order_id="coid-1", start_time=None):
    return (
        Symbol("BTCUSDT"),
        client_order_id,
        Decimal("50000"),
        start_time or datetime.now(timezone.utc),
    )


def _position(symbol="BTCUSDT", quantity="1", entry_price="100"):
    return SimpleNamespace(
        symbol=Symbol(symbol),
        quantity=Decimal(quantity),
        entry_price=Decimal(entry_price),
        updated_time=datetime.now(timezone.utc),
    )


def _oms_for_aegis(
    order_records,
    *,
    live_ids=(),
    exchange_result=None,
    metric_raises=False,
    balance="1000",
    positions=None,
    balance_raises=False,
):
    oms = _oms()

    rate_limiter = MagicMock()
    rate_limiter.guard = MagicMock(return_value=_async_cm(None))
    oms.rate_limiter = rate_limiter

    exchange = MagicMock()
    exchange.get_order_details_from_history = AsyncMock(return_value=exchange_result)
    exchange.exchange = MagicMock(return_value="bybit_linear")
    if balance_raises:
        exchange.get_wallet_balance = AsyncMock(side_effect=RuntimeError("no balance"))
    else:
        exchange.get_wallet_balance = AsyncMock(
            return_value=SimpleNamespace(margin_balance=Decimal(balance))
        )
    oms.config = SimpleNamespace(
        config=SimpleNamespace(oms_id="oms1"),
        exchange=exchange,
    )

    metric_builder = MagicMock()
    metric_builder.create_trade = (
        AsyncMock(side_effect=RuntimeError("db down")) if metric_raises else AsyncMock()
    )
    metric_builder.create_equity = AsyncMock()
    metric_builder.create_position = AsyncMock()
    oms.metric_builder = metric_builder

    opm = MagicMock()
    opm.order_pools.get_order_pool = MagicMock(
        return_value=_async_cm(dict.fromkeys(live_ids))
    )
    opm.order_pools.order_records = order_records
    oms.opm = opm

    position = MagicMock()
    position.update_exchange = AsyncMock()
    position.exchange = {p.symbol: p for p in (positions or [])}
    oms.position = position

    return oms


def _retired(oms, package_id, record):
    """A record is retired when it no longer appears under its package."""
    return record not in oms.opm.order_pools.order_records.get(package_id, [])


def test_aegis_not_found_recent_keeps_record():
    rec = _record()  # start_time = now → age < give-up window
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=None)
    asyncio.run(oms.on_aegis_update())
    assert not _retired(oms, "pkg1", rec)


def test_aegis_not_found_stale_drops_record():
    rec = _record(start_time=datetime.now(timezone.utc) - timedelta(hours=2))
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=None)
    asyncio.run(oms.on_aegis_update())
    assert _retired(oms, "pkg1", rec)


def test_aegis_created_status_keeps_record():
    rec = _record()
    order = _make_order_result(filled_size=0, status=OrderStatus.CREATED)
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order)
    asyncio.run(oms.on_aegis_update())
    assert not _retired(oms, "pkg1", rec)


def test_aegis_partially_filled_keeps_record():
    rec = _record()
    order = _make_order_result(filled_size="0.5", status=OrderStatus.PARTIALLY_FILLED)
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order)
    asyncio.run(oms.on_aegis_update())
    assert not _retired(oms, "pkg1", rec)


def test_aegis_zero_fill_terminal_retires_record():
    rec = _record()
    order = _make_order_result(filled_size=0, status=OrderStatus.CANCELLED)
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order)
    asyncio.run(oms.on_aegis_update())
    assert _retired(oms, "pkg1", rec)
    oms.metric_builder.create_trade.assert_not_awaited()


def test_aegis_filled_creates_trade_and_retires():
    rec = _record()
    order = _make_order_result(filled_size="1.0", size="1.0", status=OrderStatus.FILLED)
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order)
    asyncio.run(oms.on_aegis_update())
    assert _retired(oms, "pkg1", rec)
    oms.metric_builder.create_trade.assert_awaited_once()


def test_aegis_sell_side_uses_negative_quantity():
    rec = _record()
    order = _make_order_result(
        filled_size="1.0", size="1.0", status=OrderStatus.FILLED, side=OrderSide.SELL
    )
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order)
    asyncio.run(oms.on_aegis_update())
    kwargs = oms.metric_builder.create_trade.call_args.kwargs
    assert kwargs["executed_quantity"].startswith("-")


def test_aegis_metric_failure_keeps_record():
    rec = _record()
    order = _make_order_result(filled_size="1.0", size="1.0", status=OrderStatus.FILLED)
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order, metric_raises=True)
    asyncio.run(oms.on_aegis_update())
    assert not _retired(oms, "pkg1", rec)


def test_aegis_exchange_error_keeps_record():
    rec = _record()
    oms = _oms_for_aegis({"pkg1": [rec]})
    oms.config.exchange.get_order_details_from_history = AsyncMock(
        side_effect=RuntimeError("timeout")
    )
    asyncio.run(oms.on_aegis_update())
    assert not _retired(oms, "pkg1", rec)


def test_aegis_skips_orders_still_live_in_pool():
    rec = _record(client_order_id="coid-1")
    oms = _oms_for_aegis({"pkg1": [rec]}, live_ids=("coid-1",))
    asyncio.run(oms.on_aegis_update())
    # never queried, never retired — it is still an open order
    oms.config.exchange.get_order_details_from_history.assert_not_awaited()
    assert not _retired(oms, "pkg1", rec)


def test_aegis_empty_package_deleted_after_retire():
    rec = _record()
    order = _make_order_result(filled_size="1.0", size="1.0", status=OrderStatus.FILLED)
    oms = _oms_for_aegis({"pkg1": [rec]}, exchange_result=order)
    asyncio.run(oms.on_aegis_update())
    assert "pkg1" not in oms.opm.order_pools.order_records


def test_aegis_upserts_equity_and_position():
    oms = _oms_for_aegis({}, balance="1234.5", positions=[_position()])
    asyncio.run(oms.on_aegis_update())
    oms.metric_builder.create_equity.assert_awaited_once()
    assert oms.metric_builder.create_equity.await_args.kwargs["equity"] == "1234.5"
    oms.metric_builder.create_position.assert_awaited_once()


def test_aegis_swallows_top_level_exception():
    oms = _oms_for_aegis({}, balance_raises=True)
    # must not raise even though the balance fetch blows up
    asyncio.run(oms.on_aegis_update())


# ---------------------------------------------------------------------------
# init  (cold-start seeding)
# ---------------------------------------------------------------------------


def _oms_for_init(latest_signal=None, pending=None, exchange=None, symbol_table=None):
    oms = _oms()
    oms.get_latest_signal = AsyncMock(return_value=latest_signal)

    opm = MagicMock()
    opm.order_pools.fetch_open_orders_snapshot = AsyncMock(return_value=MagicMock())
    opm.order_pools.resync_order_pool = AsyncMock()
    oms.opm = opm

    position = MagicMock()
    position.update_exchange = AsyncMock()
    position.update_pending = MagicMock()
    position.pending = pending if pending is not None else {}
    position.exchange = exchange if exchange is not None else {}
    position.desired = {}
    oms.position = position

    oms.config = SimpleNamespace(
        config=SimpleNamespace(
            portfolio_id="p1",
            base_asset_to_symbol_table=symbol_table or {"BTC": "BTCUSDT"},
        )
    )
    return oms


def test_init_stores_latest_signal():
    sig = _signal({"BTC": "0.5"})
    oms = _oms_for_init(latest_signal=sig)
    asyncio.run(oms.init())
    assert oms.latest_signal is sig


def test_init_seeds_missing_symbols_to_zero():
    # pending/exchange start empty → both seeded to zero, desired = 0 + 0.
    oms = _oms_for_init()
    asyncio.run(oms.init())

    sym = Symbol("BTCUSDT")
    assert oms.position.pending[sym].quantity == Decimal("0")
    assert oms.position.exchange[sym].quantity == Decimal("0")
    assert oms.position.desired[sym].quantity == Decimal("0")


def test_init_desired_is_exchange_plus_pending():
    sym = Symbol("BTCUSDT")
    now = datetime.now(timezone.utc)
    pending = {sym: Position(symbol=sym, quantity=Decimal("2"), entry_price=Decimal("0"), updated_time=now)}
    exchange = {sym: Position(symbol=sym, quantity=Decimal("3"), entry_price=Decimal("0"), updated_time=now)}
    oms = _oms_for_init(pending=pending, exchange=exchange)

    asyncio.run(oms.init())

    assert oms.position.desired[sym].quantity == Decimal("5")


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
    oms.metric_stream = MagicMock()
    oms.metric_stream.publish = AsyncMock()
    # malformed body must be logged and swallowed, never raised
    asyncio.run(oms.on_command(SimpleNamespace(data=b"not json", reply="")))
    oms.rebalance.assert_not_awaited()


def test_on_command_replies_ok_to_inbox():
    oms = _oms()
    oms.rebalance = AsyncMock()
    oms.metric_stream = MagicMock()
    oms.metric_stream.publish = AsyncMock()
    msg = SimpleNamespace(
        data=json.dumps({"command": "rebalance"}).encode(), reply="_INBOX.abc"
    )
    asyncio.run(oms.on_command(msg))
    oms.rebalance.assert_awaited_once()
    subject, payload = oms.metric_stream.publish.await_args.args
    assert subject == "_INBOX.abc"
    assert json.loads(payload.decode()) == {"status": "ok", "command": "rebalance"}


def test_on_command_replies_error_when_command_fails():
    oms = _oms()
    oms.rebalance = AsyncMock(side_effect=RuntimeError("boom"))
    oms.metric_stream = MagicMock()
    oms.metric_stream.publish = AsyncMock()
    msg = SimpleNamespace(
        data=json.dumps({"command": "rebalance"}).encode(), reply="_INBOX.abc"
    )
    asyncio.run(oms.on_command(msg))
    _, payload = oms.metric_stream.publish.await_args.args
    body = json.loads(payload.decode())
    assert body["status"] == "error"
    assert "boom" in body["error"]


def test_on_command_no_reply_for_fire_and_forget():
    oms = _oms()
    oms.rebalance = AsyncMock()
    oms.metric_stream = MagicMock()
    oms.metric_stream.publish = AsyncMock()
    # empty reply subject => nothing to reply to
    msg = SimpleNamespace(data=json.dumps({"command": "rebalance"}).encode(), reply="")
    asyncio.run(oms.on_command(msg))
    oms.rebalance.assert_awaited_once()
    oms.metric_stream.publish.assert_not_awaited()

"""Integration harness for the order-placement subsystem.

The real OrderPlacementManager, OrderPoolHandler, PositionManager and
OrderExecutor run against an in-memory FakeExchange. Only the exchange edge and
the rate limiter are doubled, so the delta <-> backlog <-> pool interactions —
where cross-component bugs live — execute for real.

The load-bearing check is `assert_conserved`: everything the OMS wants must be
accounted for by (position on the exchange) + (resting open orders) +
(quantity queued in the backlog). A leak in any of those three breaks it.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from cybotrade import Symbol
from cybotrade.models import (
    Balance,
    Level,
    OrderbookSnapshot,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdate,
    Position,
    SymbolInfo,
    TimeInForce,
)

from adrs.oms.position import PositionManager
from adrs.oms.ops.order_executer import OrderExecutor
from adrs.oms.ops.order_placement_manager import OrderPlacementManager
from adrs.oms.rate_limit.error_policy import DefaultErrorPolicy


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Model factories (slotted dataclasses — every field is required)
# ---------------------------------------------------------------------------


def make_order_update(
    *,
    symbol,
    side,
    remain_size,
    price,
    client_order_id,
    status=OrderStatus.CREATED,
    size=None,
    filled_size="0",
) -> OrderUpdate:
    remain = Decimal(str(remain_size))
    now = _now()
    return OrderUpdate(
        symbol=Symbol(str(symbol)),
        order_type=OrderType.LIMIT,
        side=side,
        time_in_force=TimeInForce.GTC,
        order_id=f"oid-{client_order_id}",
        order_time=now,
        updated_time=now,
        size=Decimal(str(size)) if size is not None else remain,
        filled_size=Decimal(str(filled_size)),
        remain_size=remain,
        price=Decimal(str(price)),
        client_order_id=client_order_id,
        status=status,
        exchange="fake",
        is_reduce_only=False,
        is_hedge_mode=False,
        orig=None,
    )


def make_symbol_info(symbol) -> SymbolInfo:
    # Generous bounds so a small delta places as a single order; tighten
    # per-test when exercising chunking / residual-threshold branches.
    return SymbolInfo(
        symbol=Symbol(str(symbol)),
        quantity_precision=3,
        price_precision=2,
        exchange="fake",
        tick_size=Decimal("0.01"),
        max_post_only_qty=Decimal("1000"),
        max_limit_qty=Decimal("1000"),
        min_limit_qty=Decimal("0.001"),
        max_market_qty=Decimal("1000"),
        min_market_qty=Decimal("0.001"),
        min_notional=Decimal("1"),
        max_notional=Decimal("1000000000"),
        quanto_multiplier=Decimal("1"),
    )


def _position(symbol, quantity) -> Position:
    return Position(
        symbol=Symbol(str(symbol)),
        quantity=Decimal(str(quantity)),
        entry_price=Decimal("0"),
        updated_time=_now(),
        orig=None,
    )


# ---------------------------------------------------------------------------
# FakeExchange — in-memory, consistent state
# ---------------------------------------------------------------------------


class FakeExchange:
    """Minimal stateful exchange. Only raising drives OMS control flow; the
    place/cancel return objects are ignored by the code under test."""

    def __init__(self, symbols, bid="100", ask="101"):
        self._symbols = [Symbol(s) for s in symbols]
        self.positions: dict[Symbol, Decimal] = {s: Decimal("0") for s in self._symbols}
        self.open_orders: dict[str, OrderUpdate] = {}
        self.history: dict[str, OrderUpdate] = {}
        self.bid = Decimal(str(bid))
        self.ask = Decimal(str(ask))
        self._fail_place = 0
        self._fail_cancel = 0
        self.calls: list[tuple] = []  # (op, symbol, client_order_id, *extra)

    # ---- knobs -----------------------------------------------------------
    def throttle_next_place(self, n=1):
        self._fail_place += n

    def throttle_next_cancel(self, n=1):
        self._fail_cancel += n

    def set_price(self, bid, ask):
        self.bid, self.ask = Decimal(str(bid)), Decimal(str(ask))

    def fill(self, client_order_id, price=None):
        """Simulate a resting order fully filling: leave the book, move the
        position, and record a terminal history row for aegis/reprice reads."""
        o = self.open_orders.pop(client_order_id, None)
        if o is None:
            raise KeyError(f"no resting order {client_order_id}")
        signed = o.remain_size if o.side == OrderSide.BUY else -o.remain_size
        self.positions[o.symbol] = self.positions.get(o.symbol, Decimal("0")) + signed
        self.history[client_order_id] = make_order_update(
            symbol=o.symbol,
            side=o.side,
            remain_size="0",
            price=price if price is not None else o.price,
            client_order_id=client_order_id,
            status=OrderStatus.FILLED,
            size=o.remain_size,
            filled_size=o.remain_size,
        )

    # ---- exchange interface ---------------------------------------------
    def exchange(self):
        return "fake"

    async def get_positions(self, symbol=None, **kw):
        return [_position(s, q) for s, q in self.positions.items()]

    async def get_open_orders(self, symbol=None, **kw):
        return [
            o
            for o in self.open_orders.values()
            if symbol is None or o.symbol == Symbol(str(symbol))
        ]

    async def place_order(
        self,
        *,
        symbol,
        side,
        quantity,
        limit=None,
        client_order_id=None,
        post_only=False,
        **kw,
    ):
        self.calls.append(("place", Symbol(str(symbol)), client_order_id, Decimal(str(quantity))))
        if self._fail_place > 0:
            self._fail_place -= 1
            raise RuntimeError("throttled place")
        self.open_orders[client_order_id] = make_order_update(
            symbol=symbol,
            side=side,
            remain_size=quantity,
            price=limit if limit is not None else self.bid,
            client_order_id=client_order_id,
        )
        return OrderResponse(
            exchange="fake",
            order_id=f"oid-{client_order_id}",
            client_order_id=client_order_id,
            exchange_response=None,
        )

    async def cancel_order(self, *, symbol, client_order_id=None, order_id=None, **kw):
        self.calls.append(("cancel", Symbol(str(symbol)), client_order_id))
        if self._fail_cancel > 0:
            self._fail_cancel -= 1
            raise RuntimeError("throttled cancel")
        self.open_orders.pop(client_order_id, None)
        return OrderResponse(
            exchange="fake",
            order_id=order_id or "",
            client_order_id=client_order_id,
            exchange_response=None,
        )

    async def get_orderbook_snapshot(self, symbol, **kw):
        return OrderbookSnapshot(
            symbol=Symbol(str(symbol)),
            last_update_time=0,
            last_update_id=0,
            bids=[Level(price=self.bid, quantity=Decimal("1"))],
            asks=[Level(price=self.ask, quantity=Decimal("1"))],
            exchange="fake",
            orig=None,
        )

    async def get_current_price(self, symbol, **kw):
        return (self.bid + self.ask) / 2

    async def get_order_details_from_history(
        self, *, symbol=None, order_id=None, client_order_id=None, **kw
    ):
        return self.history.get(client_order_id)

    async def get_wallet_balance(self, coin=None, **kw):
        return Balance(
            exchange="fake",
            coin="USDT",
            wallet_balance=Decimal("1000"),
            available_balance=Decimal("1000"),
            equity=Decimal("1000"),
            unrealised_pnl=Decimal("0"),
            initial_margin=Decimal("0"),
            margin_balance=Decimal("1000"),
            maintenance_margin=Decimal("0"),
            orig=None,
        )

    async def get_server_time(self):
        return 1_000_000

    async def get_symbol_info(self, symbol, **kw):
        return make_symbol_info(symbol)


# ---------------------------------------------------------------------------
# FakeRateLimiter — pass-through; throttling is modelled at the exchange
# ---------------------------------------------------------------------------


class FakeRateLimiter:
    def __init__(self):
        self.retry_after = 0
        self._now = 1_000_000

    def get_synced_time_ms(self):
        return self._now

    @asynccontextmanager
    async def guard(self, endpoint=None, **kw):
        yield

    @asynccontextmanager
    async def reserve(self, endpoint=None, **kw):
        yield

    async def on_resync_time(self):
        pass

    def __str__(self):
        return "FakeRateLimiter"


# ---------------------------------------------------------------------------
# FakeConfig — the small ConfigManager surface the stack reads
# ---------------------------------------------------------------------------


class FakeConfig:
    def __init__(self, symbols, exchange):
        self.exchange = exchange
        self.symbol_infos = {Symbol(s): make_symbol_info(s) for s in symbols}
        self.config = SimpleNamespace(
            base_asset_to_symbol_table={s: s for s in symbols},
            credentials=SimpleNamespace(exchange="fake"),
            portfolio_id="pf1",
            oms_id="oms1",
            initial_balance=Decimal("1000"),
            leverage=Decimal("1"),
            max_retries_allowed=3,
            replace_best_bid_ask_time=30,
            min_limit_replace_interval=10,
            max_limit_replace_interval=20,
            order_placement_interval=15,
            expiry_check=30,
            soft_limit_percent=Decimal("0.8"),
        )

    async def update_symbol_info(self, rate_limiter, force=False):
        # Already seeded in __init__; keep them fresh from the exchange.
        for symbol in list(self.symbol_infos):
            self.symbol_infos[symbol] = await self.exchange.get_symbol_info(symbol)


# ---------------------------------------------------------------------------
# Builder + invariant
# ---------------------------------------------------------------------------


def build_opm(symbols=("BTCUSDT",), **exchange_kwargs):
    """Wire the real placement stack over the fakes. Returns (opm, sim, config)."""
    sim = FakeExchange(symbols, **exchange_kwargs)
    config = FakeConfig(symbols, sim)
    rate_limiter = FakeRateLimiter()
    position = PositionManager(config=config, rate_limiter=rate_limiter)
    opm = OrderPlacementManager(
        position=position,
        config=config,
        rate_limiter=rate_limiter,
        error_policy=DefaultErrorPolicy(),
        executor_cls=OrderExecutor,
    )
    return opm, sim, config


def set_desired(opm, symbol, quantity):
    opm.position.desired[Symbol(symbol)] = _position(symbol, quantity)


def _resting_signed(open_orders, symbol) -> Decimal:
    total = Decimal("0")
    for o in open_orders.values():
        if o.symbol != symbol:
            continue
        total += o.remain_size if o.side == OrderSide.BUY else -o.remain_size
    return total


def backlog_signed(backlog_list, symbol) -> Decimal:
    """Signed quantity queued for (re)placement. CancelBacklogs carries no qty;
    OrderBacklogs/ExpiredBacklogs carry a qty that WILL become a position."""
    total = Decimal("0")
    for b in backlog_list:
        if Symbol(b.symbol) != symbol:
            continue
        qty = getattr(b, "qty", None)
        if qty is None:
            continue
        total += qty if b.side == OrderSide.BUY else -qty
    return total


def assert_conserved(opm, sim):
    """Quiescent conservation: with the backlog fully drained and every
    cancel/place confirmed, what the OMS wants must equal what actually exists —
    position on the exchange plus resting orders. A persistent double-position
    (the class of bug a mocked executor hides) breaks this.

    Precondition: the backlog is empty. Mid-flight states (a failed cancel
    leaves a doomed resting order) are not steady states; drain() first.
    """
    backlog = opm.order_pools.order_backlog
    assert len(backlog) == 0, (
        f"assert_conserved called with a non-empty backlog ({len(backlog)} items) "
        f"— not a steady state; call drain() first"
    )
    for symbol, want in opm.position.desired.items():
        on_exchange = sim.positions.get(symbol, Decimal("0"))
        resting = _resting_signed(sim.open_orders, symbol)
        accounted = on_exchange + resting
        assert want.quantity == accounted, (
            f"conservation leak for {symbol}: desired={want.quantity} but "
            f"accounted={accounted} (exchange={on_exchange}, resting={resting})"
        )


def drain(opm, max_ticks=10):
    """Run placement + backlog retry until the backlog empties and stops
    changing, so a steady state can be asserted. Guards against a backlog that
    never converges (returns with it still non-empty; assert_conserved then
    trips loudly rather than looping forever)."""
    for _ in range(max_ticks):
        asyncio.run(opm.on_order_placement())
        asyncio.run(opm.on_retry_backlog())
        if len(opm.order_pools.order_backlog) == 0:
            return
    # left non-empty on purpose: caller's assert_conserved will surface it

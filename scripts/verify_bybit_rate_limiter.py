"""
Manual smoke test for the header-based Bybit rate limiter (see PR #35).

Bursts real calls against Bybit testnet/demo, fast enough to exhaust a UID
pool, and prints the limiter's state after every call so you can eyeball:

  - remaining/limit actually tracking down from Bybit's own
    X-Bapi-Limit-Status header (not just the local optimistic decrement)
  - LocalRateLimitError firing locally *before* Bybit ever returns a real
    403/10006 (a real 10006 here means the header math admitted too
    aggressively - a red flag, not expected)
  - capacity recovering once reset_ts passes

Refuses to run against anything but Bybit testnet/demo - there is no flag
to point this at prod.

Usage:
    export BYBIT_API_KEY=...
    export BYBIT_API_SECRET=...

    # Risk-free: read-only endpoints, no orders placed
    uv run python scripts/verify_bybit_rate_limiter.py --endpoint wallet
    uv run python scripts/verify_bybit_rate_limiter.py --endpoint open_orders --symbol BTCUSDT

    # Exercises UID_PLACE/UID_CANCEL: places+cancels real (testnet) limit
    # orders, priced 50% below the best bid so they never fill
    uv run python scripts/verify_bybit_rate_limiter.py --endpoint place_cancel --symbol BTCUSDT

    # Against demo trading instead of testnet
    uv run python scripts/verify_bybit_rate_limiter.py --endpoint wallet --demo
"""

import argparse
import asyncio
import os
import sys
from decimal import Decimal

from cybotrade import Symbol
from cybotrade.bybit import BybitLinearClient
from cybotrade.models import OrderSide

from adrs.oms.rate_limit.exchange_limit_profiles import Endpoints
from adrs.oms.rate_limit.rate_limiter import BybitRateLimiter, LocalRateLimitError


class _FakeConfig:
    def __init__(self, soft_limit_percent: Decimal):
        self.soft_limit_percent = soft_limit_percent


class _FakeConfigManager:
    """Just enough of ConfigManager's shape for BybitRateLimiter.__init__."""

    def __init__(self, exchange: BybitLinearClient, soft_limit_percent: Decimal):
        self.exchange = exchange
        self.config = _FakeConfig(soft_limit_percent)


async def _burst_read_only(
    limiter: BybitRateLimiter, endpoint: Endpoints, n: int, call
):
    # Fired concurrently, not awaited one at a time: testnet round-trip
    # latency (~100-300ms) means a sequential loop lands each call in its
    # own fresh 1s window and never actually exhausts the pool.
    async def one(i: int) -> str:
        try:
            async with limiter.guard(endpoint=endpoint):
                await call()
            return f"[{i:>3}] ok    {limiter}"
        except LocalRateLimitError as e:
            return f"[{i:>3}] BLOCK (local) {e}"
        except Exception as e:
            return f"[{i:>3}] ERROR {type(e).__name__}: {e}"

    tasks = [asyncio.create_task(one(i)) for i in range(n)]
    for coro in asyncio.as_completed(tasks):
        print(await coro)


async def _burst_place_cancel(limiter: BybitRateLimiter, symbol: Symbol, n: int):
    exchange = limiter.exchange
    info = await exchange.get_symbol_info(symbol=symbol)
    book = await exchange.get_orderbook_snapshot(symbol=symbol)
    best_bid = book.bids[0].price
    # 50% below the best bid, floored to a tick multiple - guaranteed not to fill
    safe_price = ((best_bid * Decimal("0.5")) // info.tick_size) * info.tick_size
    qty = info.min_limit_qty
    print(f"Placing {qty} {symbol} limit buys @ {safe_price} (best bid {best_bid})")

    # Fired concurrently: a sequential loop never exceeds ~1 call/window
    # against real network latency, so it can't exercise the block/recover
    # path - see the identical fix in _burst_read_only.
    async def place(i: int):
        try:
            async with limiter.guard(endpoint=Endpoints.PLACE_ORDER):
                order = await exchange.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=qty,
                    limit=safe_price,
                )
            print(f"[place  {i:>3}] ok    {limiter}")
            return order.order_id
        except LocalRateLimitError as e:
            print(f"[place  {i:>3}] BLOCK (local) {e}")
        except Exception as e:
            print(f"[place  {i:>3}] ERROR {type(e).__name__}: {e}")
        return None

    tasks = [asyncio.create_task(place(i)) for i in range(n)]
    order_ids = [oid for oid in await asyncio.gather(*tasks) if oid is not None]

    async def cancel(i: int, order_id: str) -> bool:
        try:
            async with limiter.guard(endpoint=Endpoints.CANCEL_ORDER):
                await exchange.cancel_order(symbol=symbol, order_id=order_id)
            print(f"[cancel {i:>3}] ok    {limiter}")
            return True
        except LocalRateLimitError as e:
            print(f"[cancel {i:>3}] BLOCK (local) {e}")
        except Exception as e:
            print(f"[cancel {i:>3}] ERROR {type(e).__name__}: {e}")
        return False

    cancel_tasks = [
        asyncio.create_task(cancel(i, oid)) for i, oid in enumerate(order_ids)
    ]
    cancelled = await asyncio.gather(*cancel_tasks)
    leftover = [oid for oid, ok in zip(order_ids, cancelled) if not ok]
    if leftover:
        print(
            f"WARNING: {len(leftover)} order(s) still open, not cancelled: {leftover}"
        )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        choices=["wallet", "open_orders", "position", "place_cancel"],
        default="wallet",
        help="Which UID pool to burst (default: wallet, read-only/risk-free)",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--n", type=int, default=30, help="Number of calls to burst")
    parser.add_argument(
        "--demo", action="store_true", help="Use Bybit demo trading instead of testnet"
    )
    parser.add_argument(
        "--soft-limit-percent",
        type=Decimal,
        default=Decimal("1.0"),
        help="Fraction of the hard limit to use as bootstrap capacity, "
        "before any header has been seen (default: 1.0, i.e. no buffer)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        print("Set BYBIT_API_KEY and BYBIT_API_SECRET (testnet/demo credentials).")
        sys.exit(1)

    exchange = BybitLinearClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=not args.demo,
        demo=args.demo,
    )
    config = _FakeConfigManager(exchange, soft_limit_percent=args.soft_limit_percent)
    limiter = BybitRateLimiter(config)
    await limiter.init()
    print(f"Synced. Starting state: {limiter}")

    symbol = Symbol(args.symbol)
    if args.endpoint == "wallet":
        await _burst_read_only(
            limiter, Endpoints.GET_WALLET_BALANCE, args.n, exchange.get_wallet_balance
        )
    elif args.endpoint == "open_orders":
        await _burst_read_only(
            limiter,
            Endpoints.GET_OPEN_ORDERS,
            args.n,
            lambda: exchange.get_open_orders(symbol=symbol),
        )
    elif args.endpoint == "position":
        await _burst_read_only(
            limiter,
            Endpoints.GET_POSITION,
            args.n,
            lambda: exchange.get_positions(symbol=symbol),
        )
    elif args.endpoint == "place_cancel":
        await _burst_place_cancel(limiter, symbol, args.n)

    print(f"Final state: {limiter}")


if __name__ == "__main__":
    asyncio.run(main())

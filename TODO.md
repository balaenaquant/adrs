# TODO

Known issues and deferred items, recorded so they are not rediscovered. Newest
first. Delete an entry when it is fixed.

## Rate limiter

**A cross-process token bucket would replace the static tenant split.** IP-scoped
budgets are currently divided by `Config.tenants_per_egress_ip`, which cannot lend
unused budget between quiet and busy tenants on a shard. On Binance the IP-wide
`x-mbx-used-weight-1m` reconciliation covers most of this, but Bybit exposes no
equivalent header for its IP pool, so the static split is all it has.

**The window-roll guard only covers the 1-minute window.** `_reconcile_from_headers`
skips adoption when `last_reset_1m_timestamp` moves across `reset_limits()`, but the
10-second order-count window rolls six times as often and its header can still be
adopted across a 10s boundary. Same class of bug, one interval down. Bounded by
`record_usage`'s local tally in the meantime.

**Per-tenant share enforcement rests on an uncorrected local tally.** No per-tenant
header exists to correct its drift, unlike the IP-wide figure. Inherent to the
design, not a defect — but it means the divided-share ceiling is an estimate while
the IP ceiling is measured.

**`local_cache_error()` marks only the local tally spent, not `_ip_wide_used_weight`.**
Benign today: every path into it also arms `retry_after`, and `check_limits` refuses
on `retry_after` before reaching `_has_capacity`. Cosmetic asymmetry.

**`check_limits` docstring still claims "POST_ORDER/Order Creation is exempted".**
The code blocks everything while `retry_after` is active. Pre-existing, now more
visible.

## Price feed

**A single transient crossed frame evicts an otherwise-fresh quote** until the next
good frame arrives. This is the correct direction — serving a knowingly stale price
is worse — but worth watching if `PriceFeed.stats()["fallback_unseen"]` climbs in
production.

**`stats()` tests assert 3 of 6 counters.** `fallback_backstop` and `rejected` are
not covered by name. Two lines whenever that file is next touched.

**`Quote.mid` uses `Decimal("2.0")` while `order_executer.py` uses `Decimal("2")`.**
Numerically identical (`Decimal` division precision is context-driven, not
operand-driven); cosmetic only.

## Tests

**`ruff format` disagrees with two pre-existing files:** `tests/test_oms.py` and
`tests/integration/harness.py`. Untouched by recent work; formatting drift only.

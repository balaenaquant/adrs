"""
PriceFeed staleness policy.

The load-bearing behaviour is that liveness belongs to the *connection* while
freshness belongs to the quote. @bookTicker pushes on every top-of-book change,
so silence on a healthy socket means the book has not moved and an old quote is
still current. Gating on per-symbol age instead would send every quiet symbol
back to REST each tick to re-fetch a price that had not changed, which is the
entire cost this feed exists to remove.

Clock is injected so none of this needs sleeps.
"""

from decimal import Decimal

from cybotrade import Symbol

from adrs.oms.price_feed import PriceFeed, Quote

BTC = Symbol("BTCUSDT")
THIN = Symbol("BITOUSDT")


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _feed(clock: FakeClock, **kw) -> PriceFeed:
    return PriceFeed(clock=clock, **kw)


def test_serves_a_fresh_quote():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    quote = feed.get(BTC)
    assert quote is not None
    assert (quote.bid, quote.ask) == (Decimal("100"), Decimal("102"))


def test_mid_is_the_midpoint_of_the_book():
    """
    Asserted against a hand-computed constant, not against a re-derivation of
    the same expression -- that would pass even if the formula were wrong. The
    cross-check against cybotrade's own implementation lives in
    tests/test_price_feed_wiring.py.
    """
    assert Quote(
        bid=Decimal("100.00"), ask=Decimal("101.00"), received_at=0.0
    ).mid == Decimal("100.50")
    # Odd cent: the mid falls between representable prices
    assert Quote(
        bid=Decimal("63889.10"), ask=Decimal("63889.20"), received_at=0.0
    ).mid == Decimal("63889.15")


def test_quiet_symbol_is_served_while_liveness_is_fresh():
    """The reason this design exists: an idle book costs no REST weight."""
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(30.0)  # thin book has not ticked for 30s...
    feed.apply(BTC, Decimal("100"), Decimal("102"))  # ...but BTC keeps arriving
    assert feed.get(THIN) is not None


def test_quote_is_withheld_once_liveness_goes_stale():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    clock.advance(2.5)  # past heartbeat_max_age of 2.0s
    assert feed.get(BTC) is None


def test_any_symbol_refreshes_liveness_for_every_symbol():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(1.5)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    clock.advance(1.5)  # 3s since THIN's quote, 1.5s since any message
    assert feed.get(THIN) is not None


def test_note_message_refreshes_liveness_without_storing_a_quote():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(1.5)
    feed.note_message()  # e.g. an unparseable frame: proves the socket delivers
    clock.advance(1.5)
    assert feed.get(THIN) is not None
    assert feed.get(BTC) is None  # never had a quote


def test_backstop_withholds_a_quote_even_while_liveness_is_fresh():
    """Covers the one failure liveness cannot see: a dead single subscription."""
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    for _ in range(70):  # BTC keeps the socket provably alive for 70s
        clock.advance(1.0)
        feed.apply(BTC, Decimal("100"), Decimal("102"))
    assert feed.get(BTC) is not None
    assert feed.get(THIN) is None  # past the 60s backstop


def test_unseen_symbol_returns_none():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    assert feed.get(THIN) is None


def test_clear_drops_quotes_and_resets_liveness():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    feed.clear()
    assert feed.get(BTC) is None
    # Even a fresh quote for another symbol must not resurrect the cleared one
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    assert feed.get(BTC) is None


def test_invalidate_affects_only_that_symbol():
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    feed.invalidate(THIN)
    assert feed.get(THIN) is None
    assert feed.get(BTC) is not None


def test_crossed_book_drops_the_cached_quote_but_still_counts_as_liveness():
    """
    A rejected book used to leave the previous quote in the cache. With liveness
    kept fresh by other symbols, a symbol pushing nothing but crossed books
    would be served its last good price for the whole 60s backstop — the one
    way this feed can knowingly quote a wrong price. The frame still proves the
    socket is delivering, so liveness must survive even though the quote does
    not.
    """
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    clock.advance(1.5)
    assert feed.apply(BTC, Decimal("103"), Decimal("102")) is False  # bid > ask
    assert feed.get(BTC) is None  # caller falls back to REST, not to a stale price
    assert feed.stats()["liveness_age_sec"] == 0.0


def test_a_symbol_pushing_only_crossed_books_never_serves_a_stale_price():
    """The production shape: one symbol goes bad while the socket stays healthy."""
    clock = FakeClock()
    feed = _feed(clock)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    for _ in range(10):
        clock.advance(1.0)
        feed.apply(BTC, Decimal("100"), Decimal("102"))  # socket stays alive
        feed.apply(THIN, Decimal("8.70"), Decimal("8.60"))  # crossed
    assert feed.get(THIN) is None  # well inside the 60s backstop
    assert feed.get(BTC) is not None


def test_equal_bid_ask_and_zero_sides_are_rejected():
    clock = FakeClock()
    feed = _feed(clock)
    assert feed.apply(BTC, Decimal("100"), Decimal("100")) is False
    assert feed.apply(BTC, Decimal("0"), Decimal("102")) is False
    assert feed.apply(BTC, Decimal("100"), Decimal("0")) is False
    assert feed.get(BTC) is None


def test_heartbeat_requirement_can_be_disabled_for_bybit():
    """
    Bybit's orderbook.1 re-pushes a snapshot after 3s of no change, so it carries
    per-symbol liveness and needs no heartbeat subscription.
    """
    clock = FakeClock()
    feed = _feed(clock, heartbeat_max_age_sec=None, quote_max_age_sec=4.0)
    feed.apply(THIN, Decimal("8.66"), Decimal("8.67"))
    clock.advance(3.0)
    assert feed.get(THIN) is not None
    clock.advance(2.0)  # 5s > 4s per-symbol cap
    assert feed.get(THIN) is None


def test_stats_count_fallbacks_by_reason():
    clock = FakeClock()
    feed = _feed(clock)
    feed.get(BTC)  # no liveness at all yet
    feed.apply(BTC, Decimal("100"), Decimal("102"))
    feed.get(THIN)  # unseen
    feed.get(BTC)  # served
    stats = feed.stats()
    assert stats["served"] == 1
    assert stats["fallback_no_liveness"] == 1
    assert stats["fallback_unseen"] == 1

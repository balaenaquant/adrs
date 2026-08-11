import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from cybotrade import Symbol

logger = logging.getLogger(__name__)

# How long since *any* frame on the connection before every quote is withheld.
#
# This is the liveness signal, not a per-quote freshness limit. @bookTicker
# pushes on every top-of-book change, so silence on a healthy socket means the
# book has not moved and the last quote is still the current one. The heartbeat
# subscription (BTCUSDT) was measured at 58-212 frames/sec with a 515ms worst
# gap over 60s, so 2s is ~4x the observed worst case: tight enough to catch a
# wedged socket inside one placement tick, loose enough to survive a scheduling
# hiccup.
DEFAULT_HEARTBEAT_MAX_AGE_SEC = 2.0

# Backstop for the single failure liveness cannot see: one symbol's subscription
# dying while the connection keeps delivering others. Deliberately generous — a
# tight value would reintroduce the REST-per-tick cost on quiet symbols that this
# feed exists to remove. The trade is up to this long quoting from a dead
# subscription, versus REST on every tick for every quiet symbol.
DEFAULT_QUOTE_MAX_AGE_SEC = 60.0


@dataclass(frozen=True, slots=True)
class Quote:
    bid: Decimal
    ask: Decimal
    received_at: float  # time.monotonic() at parse time

    @property
    def mid(self) -> Decimal:
        # Must match cybotrade's ExchangeClient.get_current_price exactly, or the
        # feed and REST paths would quote differently for the same book.
        return (self.bid + self.ask) / Decimal("2.0")


class PriceFeed:
    """
    Top-of-book cache with a staleness policy. Pure: it makes no exchange calls,
    knows nothing about websockets, and decides nothing about what to do on a
    miss — callers fall back to REST.

    Set `heartbeat_max_age_sec=None` for a feed whose stream re-pushes an
    unchanged book on a timer (Bybit's orderbook.1 does, after 3s), in which case
    per-symbol age alone is a sufficient guard and `quote_max_age_sec` should be
    tightened accordingly.
    """

    def __init__(
        self,
        heartbeat_max_age_sec: float | None = DEFAULT_HEARTBEAT_MAX_AGE_SEC,
        quote_max_age_sec: float = DEFAULT_QUOTE_MAX_AGE_SEC,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._heartbeat_max_age_sec = heartbeat_max_age_sec
        self._quote_max_age_sec = quote_max_age_sec
        self._clock = clock
        self._quotes: dict[Symbol, Quote] = {}
        self._last_message_at: float | None = None
        self._served = 0
        self._fallback_no_liveness = 0
        self._fallback_unseen = 0
        self._fallback_backstop = 0
        self._rejected = 0

    def note_message(self) -> None:
        """
        Record that the connection delivered something.

        Called for every frame, including ones that do not yield a quote: a
        malformed frame still proves the socket is alive.
        """
        self._last_message_at = self._clock()

    def apply(self, symbol: Symbol, bid: Decimal, ask: Decimal) -> bool:
        """
        Store a quote. False if the book is crossed or zero, in which case this
        symbol's cached quote is dropped too.

        Dropping it is the point. A symbol that starts pushing an unusable book
        — delisting, settlement, one side of the book emptied — otherwise has
        every update rejected while liveness stays fresh off other symbols, so
        get() would keep serving the last good quote for the full backstop
        window. That is the only path in this feed that can knowingly serve a
        wrong price; falling back to REST reads the real book instead. Liveness
        still counts: the frame arrived, so the socket is provably delivering.
        """
        self.note_message()
        if bid <= 0 or ask <= 0 or bid >= ask:
            self._rejected += 1
            self._quotes.pop(symbol, None)
            logger.warning(
                f"[PRICE_FEED] Rejected implausible book for {symbol} and dropped "
                f"its cached quote: bid={bid} ask={ask}"
            )
            return False
        self._quotes[symbol] = Quote(bid=bid, ask=ask, received_at=self._clock())
        return True

    def get(self, symbol: Symbol) -> Quote | None:
        """The current quote, or None when the caller must fall back to REST."""
        now = self._clock()
        if self._heartbeat_max_age_sec is not None:
            if (
                self._last_message_at is None
                or now - self._last_message_at > self._heartbeat_max_age_sec
            ):
                self._fallback_no_liveness += 1
                return None
        quote = self._quotes.get(symbol)
        if quote is None:
            self._fallback_unseen += 1
            return None
        if now - quote.received_at > self._quote_max_age_sec:
            self._fallback_backstop += 1
            return None
        self._served += 1
        return quote

    def clear(self) -> None:
        """
        Drop every quote and reset liveness. Called on reconnect: after a gap in
        delivery no cached quote can be trusted, so each symbol takes one REST
        read before rejoining the feed.
        """
        self._quotes.clear()
        self._last_message_at = None

    def invalidate(self, symbol: Symbol) -> None:
        """Drop one symbol. Unused on Binance; Bybit needs it on a sequence gap."""
        self._quotes.pop(symbol, None)

    def stats(self) -> dict[str, int | float | None]:
        now = self._clock()
        return {
            "served": self._served,
            "fallback_no_liveness": self._fallback_no_liveness,
            "fallback_unseen": self._fallback_unseen,
            "fallback_backstop": self._fallback_backstop,
            "rejected": self._rejected,
            "tracked_symbols": len(self._quotes),
            "liveness_age_sec": (
                None if self._last_message_at is None else now - self._last_message_at
            ),
        }

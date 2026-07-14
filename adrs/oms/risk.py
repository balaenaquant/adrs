import logging

from decimal import Decimal
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from adrs.oms.config import ConfigManager
from adrs.oms.position import PositionManager, Positions
from adrs.oms.rate_limit.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class RiskConfig(BaseModel):
    nav_floor: Decimal = Decimal("0.60")  # margin_balance / initial kill level
    max_exposure: Decimal = Decimal("2.0")  # committed notional / initial, 2.0 = 200%


class RiskEngine:
    """
    Base risk engine: every hook is a no-op. The OMS calls these at fixed
    lifecycle points; subclass and override the hooks to add real controls
    without touching the OMS flow.

    Hooks:
      - cap_desired: pre-placement gate. Return desired unchanged to allow, or
        a scaled/rewritten mapping to constrain exposure.
      - run_risk_checks: monitor, run on a cron. Call self.on_breach() to shut
        the OMS down (same path as SIGINT/SIGTERM).
    """

    def __init__(
        self,
        config: ConfigManager,
        position: PositionManager,
        rate_limiter: RateLimiter,
        on_breach: Callable[[], Awaitable[None]],
        risk: RiskConfig | None = None,
    ):
        self.config = config
        self.position = position
        self.rate_limiter = rate_limiter
        self.on_breach = on_breach  # OMS shutdown, same path as SIGINT/SIGTERM
        self.risk = risk or RiskConfig()

    def cap_desired(self, desired: Positions) -> Positions:
        """No-op exposure gate; override to constrain desired positions."""
        return desired

    async def run_risk_checks(self):
        """No-op monitor; override to check NAV/drawdown and trigger on_breach."""
        return

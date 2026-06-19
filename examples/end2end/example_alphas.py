from examples.portfolio_sample.alpha001 import Alpha001
from examples.portfolio_sample.alpha002 import Alpha002
from examples.portfolio_sample.alpha003 import Alpha003
from examples.portfolio_sample.alpha004 import Alpha004
from examples.portfolio_sample.alpha005 import Alpha005

BTC_ALPHAS = [
    Alpha001(window=100, entry_threshold=0.5, exit_threshold=0),
    Alpha002(),
    Alpha003(),
]

ETH_ALPHAS = [
    Alpha004(),
    Alpha005(),
]

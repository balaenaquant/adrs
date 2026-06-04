from importlib.metadata import version, PackageNotFoundError

from .alpha import Alpha  # noqa: E402
from .portfolio import Portfolio  # noqa: E402
from .data import DataLoader  # noqa: E402

try:
    __version__ = version("adrs")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["Alpha", "Portfolio", "DataLoader", "__version__"]

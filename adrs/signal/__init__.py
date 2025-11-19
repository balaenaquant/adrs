from .signal import Signal, SignalGenerator
from .long import Long
from .short import Short

__all__ = [
    "Signal",
    "SignalGenerator",
    "Long",
    "Short",
]

MAP: dict[str, type[SignalGenerator]] = {
    "long": Long,
    "short": Short,
}

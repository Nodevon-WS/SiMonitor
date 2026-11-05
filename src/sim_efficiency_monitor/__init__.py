"""Engine-neutral simulation efficiency monitoring."""

__version__ = "0.2.1"

from .config import MonitorConfig
from .monitor import Monitor

__all__ = ["Monitor", "MonitorConfig", "__version__"]

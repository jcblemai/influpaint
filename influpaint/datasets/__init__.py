"""
Dataset management module for InfluPaint.

Handles dataset loading, mixing, and transformations.
"""

from .mixer import build_frames

# Optional imports with graceful fallback
try:
    from .loaders import *
except ImportError:
    pass

try:
    from .transforms import *
except ImportError:
    pass

__all__ = ["build_frames"]
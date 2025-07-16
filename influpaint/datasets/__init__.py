"""
Dataset management module for InfluPaint.

Handles dataset loading, mixing, and transformations.

Submodules:
- mixer: Core dataset mixing functionality
- loaders: Dataset loading utilities  
- transforms: Data transformations
- test/: Comprehensive test suite with test dataset
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
"""
Dataset management module for InfluPaint.

Handles dataset loading, mixing, and transformations.

Submodules:
- mixer: Core dataset mixing functionality
- loaders: Dataset loading utilities  
- transforms: Data transformations
- test/: Comprehensive test suite with test dataset
- byclaude_test_dataset_builder: Test dataset creation utility
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

# Test dataset builder function
try:
    from .byclaude_test_dataset_builder import create_test_dataset
except ImportError:
    pass

__all__ = ["build_frames", "create_test_dataset"]
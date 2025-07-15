"""
Batch processing module for InfluPaint.

Handles training, inpainting, job generation, and result aggregation.
"""

# Main functions - these modules are designed to be run as scripts
# Import key functions for programmatic access with graceful fallback
try:
    from .hyperparameter_configs import *
except ImportError:
    pass

__all__ = []
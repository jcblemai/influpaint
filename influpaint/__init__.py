"""
InfluPaint: Epidemic Forecasting with Diffusion Models

A modular package for training and running diffusion models on epidemiological data.
"""

__version__ = "0.1.0"
__author__ = "UNC IDD"

# Main module imports with graceful fallback
available_modules = []

try:
    from . import datasets
    available_modules.append("datasets")
except ImportError:
    pass

try:
    from . import batch
    available_modules.append("batch")
except ImportError:
    pass

try:
    from . import models
    available_modules.append("models")
except ImportError:
    pass

try:
    from . import utils
    available_modules.append("utils")
except ImportError:
    pass

__all__ = available_modules
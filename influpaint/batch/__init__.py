"""
Batch processing module for InfluPaint.

Handles training, inpainting, job generation, and result aggregation.
"""

# Main functions - these modules are designed to be run as scripts
# Import key functions for programmatic access with graceful fallback
try:
    from .scenarios import (
        TrainingScenario, 
        InpaintingScenario,
        get_all_training_scenarios,
        get_all_inpainting_scenarios,
        get_training_scenario,
        get_inpainting_scenario
    )
    from .config import *
    __all__ = ['TrainingScenario', 'InpaintingScenario', 'get_all_training_scenarios', 
               'get_all_inpainting_scenarios', 'get_training_scenario', 'get_inpainting_scenario']
except ImportError:
    __all__ = []
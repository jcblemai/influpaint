"""
Batch processing module for InfluPaint.

Handles training, inpainting, job generation, and result aggregation.
"""

# Main functions - these modules are designed to be run as scripts
# Import key functions for programmatic access with graceful fallback
try:
    from .scenarios import ScenarioLibrary, TrainingScenarioSpec, InpaintingScenarioSpec
    from .factory import ObjectFactory, TrainingRunConfig, ExperimentConfig
    __all__ = ['ScenarioLibrary', 'TrainingScenarioSpec', 'InpaintingScenarioSpec', 
               'ObjectFactory', 'TrainingRunConfig', 'ExperimentConfig']
except ImportError:
    __all__ = []
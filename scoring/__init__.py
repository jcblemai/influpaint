"""
Scoring Package - Forecast evaluation and visualization tools.

This package provides modular components for forecast evaluation:

- scoring.evaluation: Core data structures and scoring functions
  - ForecastRecord, ForecastDataset: Data structures for forecast data
  - score_dataset(): Compute forecast scoring metrics
  - compute_relative_scores(): Relative performance vs baseline

- scoring.plotting: Specialized forecast evaluation visualizations  
  - forecast_scores_heatmap(): Scores across models and dates
  - forecast_components_breakdown(): Metric component breakdown
  - forecast_performance_timeseries(): Performance over time by horizon

- scoring.validation: Data format validation functions
  - validate_forecast_dataset(): Comprehensive dataset validation
  - ValidationError: Custom exception for validation failures

- scoring.weighted_interval_score: WIS computation functions
  - weighted_interval_score_fast(): Core WIS implementation
  - score_Nwk_forecasts_hub(): Hub-format WIS scoring

Usage:
    import scoring.evaluation as eval
    import scoring.plotting as plot
    
    dataset = eval.ForecastDataset(records)
    scores = eval.score_dataset(dataset, ground_truth) 
    plot.forecast_scores_heatmap(scores, dataset, colors, ...)
"""

# Import submodules for access
from . import evaluation
from . import plotting
from . import validation
from . import weighted_interval_score

__all__ = ['evaluation', 'plotting', 'validation', 'weighted_interval_score']
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

__all__ = ['evaluation', 'plotting']
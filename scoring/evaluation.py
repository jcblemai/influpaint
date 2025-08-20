"""
Evaluation Module - Core data structures and scoring functions.

This module provides a clean interface for:
- Loading and structuring forecast data (hubverse CSV format)
- Computing scoring metrics (delegating to evaluate_deprecated.py)  
- Supporting multiple forecast dates, horizons, and aggregation strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Union

import pandas as pd
import numpy as np

# Import WIS computation and validation
from . import weighted_interval_score as wis
from . import validation


@dataclass
class ForecastRecord:
    model: str  # Model identifier (e.g., "i806::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_No::celebahq")
    group: str  # Group identifier (defined by orchestrator, e.g., "influpaint" or "flusight")
    display_name: str  # How to display this model in plots
    forecast_date: pd.Timestamp
    df: pd.DataFrame  # hub-format: columns include target_end_date, location, output_type, output_type_id, value, target, horizon


@dataclass
class ForecastDataset:
    records: List[ForecastRecord]

    def by_model(self) -> Dict[str, Dict[pd.Timestamp, pd.DataFrame]]:
        out: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
        for r in self.records:
            out.setdefault(r.model, {})[r.forecast_date] = r.df
        return out

    def get_models(self) -> List[str]:
        return sorted(set(r.model for r in self.records))
    
    def get_groups(self) -> List[str]:
        return sorted(set(r.group for r in self.records))
    
    def get_forecast_dates(self) -> List[pd.Timestamp]:
        return sorted(set(r.forecast_date for r in self.records))


@dataclass
class ModelGroupConfig:
    """Configuration for model grouping and visualization."""
    group_fn: Callable[[str], str] = field(default=lambda x: "default")
    color_map: Dict[str, str] = field(default_factory=lambda: {"default": "blue"})
    group_labels: Dict[str, str] = field(default_factory=dict)  # optional pretty names
    
    def get_group(self, model_name: str) -> str:
        return self.group_fn(model_name)
    
    def get_color(self, group_or_model: str) -> str:
        # Try group first, then fall back to model name
        group = self.group_fn(group_or_model) if group_or_model not in self.color_map else group_or_model
        return self.color_map.get(group, "gray")
    
    def get_label(self, group: str) -> str:
        return self.group_labels.get(group, group)


def score_dataset(
    dataset: ForecastDataset,
    ground_truth: pd.DataFrame,
    expected_dates: Optional[List] = None,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute absolute WIS and its components for all models and forecast dates.

    Args:
        dataset: ForecastDataset with forecast records
        ground_truth: Ground truth DataFrame
        expected_dates: Optional list of expected forecast dates for missing count calculation
        
    Returns:
        tuple: (scores_dataframe, missing_counts_dict)
        - scores_dataframe: Tidy dataframe with columns: model, forecast_date, target, target_end_date, scoring_metric, location, value  
        - missing_counts_dict: Dictionary mapping model names to missing forecast counts
    """
    # Validate input data format and consistency, get missing counts
    missing_counts = validation.validate_forecast_dataset(dataset, ground_truth, expected_dates)
    validation.validate_group_consistency(dataset)
    
    scores: Dict[str, pd.DataFrame] = {}
    by_model = dataset.by_model()
    
    for model, dated in by_model.items():
        model_scores = {}
        for date, fdf in dated.items():
            try:
                # Use internal WIS computation
                wis_all = wis.score_Nwk_forecasts_hub(ground_truth, fdf)
                model_scores[date] = wis_all
            except Exception as e:
                raise validation.ValidationError(
                    f"WIS computation failed for model '{model}', date {date}: {e}"
                )
        if model_scores:
            scores[model] = pd.concat(model_scores, names=["forecast_date", "target", "target_end_date"])

    if not scores:
        raise validation.ValidationError("No valid scores computed for any model/date combination")

    all_scores = pd.concat(scores, names=["model", "forecast_date", "target", "target_end_date"]).reset_index()
    id_vars = ['model', 'forecast_date', 'target', 'target_end_date', 'scoring_metric']
    location_columns = [col for col in all_scores.columns if col not in id_vars]
    tidy = pd.melt(all_scores, id_vars=id_vars, value_vars=location_columns, var_name='location', value_name='value')
    return tidy, missing_counts


def compute_relative_scores(
    absolute_scores: pd.DataFrame,
    baseline_model: str,
) -> pd.DataFrame:
    """
    Compute relative scores (total metric only) vs a baseline model, aligned on
    forecast_date, target, target_end_date, location.
    """
    if absolute_scores.empty:
        return absolute_scores.copy()

    base = absolute_scores[absolute_scores["model"] == baseline_model]
    if base.empty:
        raise ValueError(f"Baseline model '{baseline_model}' not found in scores")

    rel = absolute_scores[absolute_scores["scoring_metric"] == "wis_total"].copy()
    rel = pd.merge(
        rel,
        base,
        on=['forecast_date', 'target', 'target_end_date', 'scoring_metric', 'location'],
        suffixes=("", "_baseline"),
    )
    rel['value'] = rel['value'] / rel['value_baseline']
    rel = rel.drop(["value_baseline", "model_baseline"], axis=1)
    return rel


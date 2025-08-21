"""
Evaluation Module - Core data structures and scoring functions.

This module provides a clean interface for:
- Loading and structuring forecast data (hubverse CSV format)
- Computing scoring metrics (delegating to evaluate_deprecated.py)  
- Supporting multiple forecast dates, horizons, and aggregation strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Union, Literal

import pandas as pd
import numpy as np

# Import WIS computation and validation
from . import weighted_interval_score as wis
from . import validation


@dataclass
class MetricSpec:
    """Specification for a scoring metric with metadata."""
    name: str                               # Unique metric name
    type: Literal["per_forecast", "per_model"]  # Granularity level
    grain: Tuple[str, ...]                  # Native keys (e.g., ("model", "horizon"))
    orientation: Literal["min", "max"]      # Lower is better vs higher is better
    family: str                             # Metric family ("wis", "crps", "coverage")
    aggregator: str                         # Default aggregation ("mean", "median", "sum", "ratio")
    forecast_types: List[str]               # Supported forecast types (["quantile", "sample"])
    is_relative: bool                       # True for relative metrics (target value 1 or 0)
    depends_on: Optional[List[str]] = None  # Dependencies for relative metrics


@dataclass
class ScoringResults:
    """Container for forecast evaluation results with separate granularities."""
    forecast_metrics: pd.DataFrame          # Per-forecast scores: model, forecast_date, target, target_end_date, horizon, location, scoring_metric, value
    model_metrics: pd.DataFrame            # Per-model scores: model, [horizon], [location], scoring_metric, value
    meta: Dict                             # Metadata: forecast_unit, transform, metrics_used, etc.
    
    def to_tidy(self) -> pd.DataFrame:
        """Unified view with grain column for clean filtering."""
        f = self.forecast_metrics.assign(grain="per_forecast")
        m = self.model_metrics.assign(grain="per_model")
        return pd.concat([f, m], ignore_index=True, sort=False)
    
    def get_forecast_counts(self, by: Optional[List[str]] = None) -> pd.DataFrame:
        """Count forecasts per unit with explicit denominators."""
        # TODO: Implement forecast counting
        raise NotImplementedError("get_forecast_counts not yet implemented")
    
    def summarise_scores(self, by: List[str], agg: str = "mean", 
                        weights: Optional[pd.Series] = None, geom: bool = False) -> pd.DataFrame:
        """Aggregate with orientation-aware handling and geometric means for ratios."""
        # TODO: Implement orientation-aware aggregation
        raise NotImplementedError("summarise_scores not yet implemented")


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


class MetricRegistry:
    """Registry of all available scoring metrics with enhanced metadata."""
    
    # Per-forecast metrics (include horizon in grain)
    WIS_TOTAL = MetricSpec(
        name="wis_total",
        type="per_forecast", 
        grain=("model", "forecast_date", "horizon", "location"),
        orientation="min",
        family="wis",
        aggregator="mean",
        forecast_types=["quantile"],
        is_relative=False
    )
    
    WIS_SHARPNESS = MetricSpec(
        name="wis_sharpness",
        type="per_forecast",
        grain=("model", "forecast_date", "horizon", "location"),
        orientation="min",
        family="wis",
        aggregator="mean",
        forecast_types=["quantile"],
        is_relative=False
    )
    
    WIS_OVERPREDICTION = MetricSpec(
        name="wis_overprediction",
        type="per_forecast",
        grain=("model", "forecast_date", "horizon", "location"),
        orientation="min",
        family="wis",
        aggregator="mean", 
        forecast_types=["quantile"],
        is_relative=False
    )
    
    WIS_UNDERPREDICTION = MetricSpec(
        name="wis_underprediction",
        type="per_forecast",
        grain=("model", "forecast_date", "horizon", "location"),
        orientation="min",
        family="wis",
        aggregator="mean",
        forecast_types=["quantile"],
        is_relative=False
    )
    
    # Relative metrics
    WIS_TOTAL_REL = MetricSpec(
        name="wis_total_relative",
        type="per_forecast",
        grain=("model", "forecast_date", "horizon", "location"),
        orientation="min",
        family="wis",
        aggregator="mean",
        forecast_types=["quantile"],
        is_relative=True,
        depends_on=["wis_total"]
    )
    
    # Per-model metrics with horizon grain
    COVERAGE_95 = MetricSpec(
        name="coverage_95",
        type="per_model",
        grain=("model", "horizon"),
        orientation="max",
        family="coverage",
        aggregator="mean",
        forecast_types=["quantile"],
        is_relative=False
    )
    
    COVERAGE_95_GAP = MetricSpec(
        name="coverage_95_gap",
        type="per_model",
        grain=("model", "horizon"),
        orientation="min",  # Gap should be close to 0
        family="coverage",
        aggregator="mean",
        forecast_types=["quantile"],
        is_relative=False
    )
    
    COMPLETION_RATE = MetricSpec(
        name="completion_rate",
        type="per_model",
        grain=("model", "horizon"),
        orientation="max",
        family="availability",
        aggregator="ratio",
        forecast_types=["quantile"],
        is_relative=False
    )
    
    # Convenient groupings
    WIS_COMPONENTS = [WIS_TOTAL, WIS_SHARPNESS, WIS_OVERPREDICTION, WIS_UNDERPREDICTION]
    WIS_WITH_RELATIVE = [WIS_TOTAL, WIS_TOTAL_REL, WIS_SHARPNESS, WIS_OVERPREDICTION, WIS_UNDERPREDICTION]
    COVERAGE_METRICS = [COVERAGE_95, COVERAGE_95_GAP]
    ALL_MODEL_METRICS = [COVERAGE_95, COVERAGE_95_GAP, COMPLETION_RATE]


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
    metrics: Optional[List[MetricSpec]] = None,
    relative_baseline: Optional[str] = None,
) -> ScoringResults:
    """
    Compute scoring metrics for all models and forecast dates.

    Args:
        dataset: ForecastDataset with forecast records
        ground_truth: Ground truth DataFrame
        expected_dates: Optional list of expected forecast dates for missing count calculation
        metrics: List of metrics to compute (default: WIS components)
        relative_baseline: Model name to use as baseline for relative metrics
        
    Returns:
        ScoringResults: Container with forecast_metrics, model_metrics, and metadata
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

    # Default metrics if not specified
    if metrics is None:
        metrics = MetricRegistry.WIS_COMPONENTS
    
    # Separate per-forecast and per-model metrics
    per_forecast_metrics = [m for m in metrics if m.type == "per_forecast"]
    per_model_metrics = [m for m in metrics if m.type == "per_model"]
    
    all_scores = pd.concat(scores, names=["model", "forecast_date", "target", "target_end_date"]).reset_index()
    
    # Extract horizon from target column (e.g., "0 wk ahead" -> 0)
    all_scores['horizon'] = all_scores['target'].str.extract('(\d+)').astype(int)
    
    id_vars = ['model', 'forecast_date', 'target', 'target_end_date', 'horizon', 'scoring_metric']
    location_columns = [col for col in all_scores.columns if col not in id_vars]
    forecast_metrics = pd.melt(all_scores, id_vars=id_vars, value_vars=location_columns, 
                              var_name='location', value_name='value')
    
    # Compute per-model metrics
    model_metrics_list = []
    
    if per_model_metrics:
        # Compute coverage metrics if requested
        coverage_metrics_requested = [m for m in per_model_metrics if m.family == "coverage"]
        if coverage_metrics_requested:
            coverage_df = compute_coverage_metrics(dataset, ground_truth, expected_dates)
            if not coverage_df.empty:
                model_metrics_list.append(coverage_df)
        
        # Compute completion rate if requested  
        completion_metrics_requested = [m for m in per_model_metrics if m.family == "availability"]
        if completion_metrics_requested:
            completion_df = compute_completion_rate(dataset, expected_dates)
            if not completion_df.empty:
                model_metrics_list.append(completion_df)
    
    # Combine all per-model metrics
    if model_metrics_list:
        model_metrics = pd.concat(model_metrics_list, ignore_index=True)
    else:
        model_metrics = pd.DataFrame(columns=['model', 'horizon', 'scoring_metric', 'value'])
    
    # TODO: Compute relative metrics if baseline specified
    if relative_baseline is not None:
        relative_total = compute_relative_scores(forecast_metrics, relative_baseline)
        # For now, just append to forecast_metrics - will need proper handling later
        forecast_metrics = pd.concat([forecast_metrics, relative_total], ignore_index=True)
    
    # Create metadata
    meta = {
        'missing_counts': missing_counts,
        'metrics_computed': [m.name for m in metrics],
        'relative_baseline': relative_baseline,
        'forecast_unit': ('model', 'forecast_date', 'target', 'target_end_date', 'horizon', 'location')
    }
    
    return ScoringResults(
        forecast_metrics=forecast_metrics,
        model_metrics=model_metrics, 
        meta=meta
    )


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


def compute_coverage_metrics(dataset: ForecastDataset, ground_truth: pd.DataFrame, 
                            expected_dates: Optional[List] = None) -> pd.DataFrame:
    """
    Compute coverage metrics (95% interval coverage and gap) per model and horizon.
    
    Args:
        dataset: ForecastDataset with forecast records
        ground_truth: Ground truth DataFrame  
        expected_dates: Optional list of expected forecast dates
        
    Returns:
        DataFrame with columns: model, horizon, scoring_metric, value, n_units
    """
    coverage_results = []
    by_model = dataset.by_model()
    
    for model, dated in by_model.items():
        # Combine all forecasts for this model
        model_forecasts = []
        for date, fdf in dated.items():
            # Add forecast_date column
            fdf_with_date = fdf.copy()
            fdf_with_date['forecast_date'] = date
            model_forecasts.append(fdf_with_date)
        
        if not model_forecasts:
            continue
            
        combined_forecasts = pd.concat(model_forecasts, ignore_index=True)
        
        # Filter to quantile forecasts only
        if "output_type" in combined_forecasts.columns:
            combined_forecasts = combined_forecasts[combined_forecasts["output_type"] == "quantile"]
        
        # Use existing horizon column if available, otherwise extract from target
        if 'horizon' not in combined_forecasts.columns:
            # Extract horizon from target pattern, handle failures gracefully
            horizon_extracted = combined_forecasts['target'].str.extract('(\d+)')
            # Drop rows where horizon extraction failed
            mask = horizon_extracted.iloc[:, 0].notna()
            combined_forecasts = combined_forecasts[mask].copy()
            combined_forecasts['horizon'] = horizon_extracted[mask].astype(int)
        else:
            # Ensure horizon is numeric and drop any rows with missing horizons
            combined_forecasts = combined_forecasts.dropna(subset=['horizon']).copy()
            combined_forecasts['horizon'] = combined_forecasts['horizon'].astype(int)
        
        # Compute coverage per horizon
        for horizon in combined_forecasts['horizon'].unique():
            horizon_forecasts = combined_forecasts[combined_forecasts['horizon'] == horizon]
            
            # Get 95% prediction intervals (5th and 95th percentiles)
            lower_forecasts = horizon_forecasts[horizon_forecasts['output_type_id'] == 0.05]
            upper_forecasts = horizon_forecasts[horizon_forecasts['output_type_id'] == 0.95]
            
            if lower_forecasts.empty or upper_forecasts.empty:
                continue
                
            # Merge with ground truth
            merged_lower = pd.merge(
                lower_forecasts,
                ground_truth,
                left_on=['target_end_date', 'location'],
                right_on=['date', 'location'],
                how='inner'
            )
            
            merged_upper = pd.merge(
                upper_forecasts, 
                ground_truth,
                left_on=['target_end_date', 'location'],
                right_on=['date', 'location'],
                how='inner'
            )
            
            # Find common observations
            common_keys = set(zip(merged_lower['target_end_date'], merged_lower['location'])) & \
                         set(zip(merged_upper['target_end_date'], merged_upper['location']))
            
            if not common_keys:
                continue
                
            coverage_counts = 0
            total_counts = 0
            
            for target_date, location in common_keys:
                lower_val = merged_lower[
                    (merged_lower['target_end_date'] == target_date) & 
                    (merged_lower['location'] == location)
                ]['value_x'].iloc[0]
                
                upper_val = merged_upper[
                    (merged_upper['target_end_date'] == target_date) & 
                    (merged_upper['location'] == location)
                ]['value_x'].iloc[0]
                
                obs_val = merged_lower[
                    (merged_lower['target_end_date'] == target_date) & 
                    (merged_lower['location'] == location)
                ]['value_y'].iloc[0]
                
                # Check if observation is within prediction interval
                if lower_val <= obs_val <= upper_val:
                    coverage_counts += 1
                total_counts += 1
            
            if total_counts > 0:
                coverage_rate = coverage_counts / total_counts
                coverage_gap = coverage_rate - 0.95  # Target is 95%
                
                coverage_results.extend([
                    {
                        'model': model,
                        'horizon': horizon,
                        'scoring_metric': 'coverage_95',
                        'value': coverage_rate,
                        'n_units': total_counts
                    },
                    {
                        'model': model, 
                        'horizon': horizon,
                        'scoring_metric': 'coverage_95_gap',
                        'value': coverage_gap,
                        'n_units': total_counts
                    }
                ])
    
    return pd.DataFrame(coverage_results)


def compute_completion_rate(dataset: ForecastDataset, expected_dates: Optional[List] = None) -> pd.DataFrame:
    """
    Compute completion rate (% of expected forecasts submitted) per model and horizon.
    
    Args:
        dataset: ForecastDataset with forecast records
        expected_dates: List of expected forecast dates
        
    Returns:
        DataFrame with columns: model, horizon, scoring_metric, value, n_expected
    """
    if expected_dates is None:
        # If no expected dates specified, assume 100% completion for all models
        completion_results = []
        for record in dataset.records:
            # Extract horizon from forecast data
            if 'target' in record.df.columns:
                horizons = record.df['target'].str.extract('(\d+)').astype(int).unique()
                for horizon in horizons:
                    completion_results.append({
                        'model': record.model,
                        'horizon': horizon,
                        'scoring_metric': 'completion_rate',
                        'value': 1.0,
                        'n_expected': 1  # Unknown denominator
                    })
        return pd.DataFrame(completion_results)
    
    # Convert expected_dates to date objects if needed
    if expected_dates and hasattr(expected_dates[0], 'date'):
        expected_dates_set = set(d.date() if hasattr(d, 'date') else d for d in expected_dates)
    else:
        expected_dates_set = set(pd.to_datetime(expected_dates).date)
    
    completion_results = []
    by_model = dataset.by_model()
    
    for model, dated in by_model.items():
        # Get actual forecast dates for this model
        actual_dates = set(d.date() if hasattr(d, 'date') else d for d in dated.keys())
        
        # Get horizons from forecast data
        horizons = set()
        for date, fdf in dated.items():
            if 'horizon' in fdf.columns:
                # Use existing horizon column if available
                model_horizons = fdf.dropna(subset=['horizon'])['horizon'].astype(int).unique()
                horizons.update(model_horizons)
            elif 'target' in fdf.columns:
                # Extract horizon from target pattern, handle failures gracefully
                horizon_extracted = fdf['target'].str.extract('(\d+)')
                valid_horizons = horizon_extracted.dropna().iloc[:, 0].astype(int).unique()
                horizons.update(valid_horizons)
        
        # Compute completion rate per horizon
        for horizon in horizons:
            n_expected = len(expected_dates_set)
            n_actual = len(actual_dates & expected_dates_set)
            completion_rate = n_actual / n_expected if n_expected > 0 else 0.0
            
            completion_results.append({
                'model': model,
                'horizon': horizon,
                'scoring_metric': 'completion_rate', 
                'value': completion_rate,
                'n_expected': n_expected
            })
    
    return pd.DataFrame(completion_results)


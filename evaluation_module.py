"""
Evaluation Module - Core data structures and scoring wrappers.

This module defines a harmonized forecast data structure and functions to
score forecasts (absolute and relative) by delegating metric computation to
evaluate_deprecated.py. Plotting is handled separately in evaluate_plot.py.
"""

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

# Import ground-truth scoring helpers (WIS, hub-compatible scorer)
import evaluate_deprecated as edep


@dataclass
class ForecastRecord:
    model: str
    group: str  # e.g., "groupA" or "groupB"
    season: str
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


def score_dataset(
    dataset: ForecastDataset,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute absolute WIS and its components for all models and forecast dates.

    Returns a tidy dataframe with columns:
    - model, forecast_date, target, target_end_date, wis_type, location, value
    """
    scores: Dict[str, pd.DataFrame] = {}
    by_model = dataset.by_model()
    for model, dated in by_model.items():
        model_scores = {}
        for date, fdf in dated.items():
            try:
                # Delegate the hub-format scoring to deprecated helpers
                wis_all = edep.score_Nwk_forecasts_hub(ground_truth, fdf)
                model_scores[date] = wis_all
            except Exception:
                continue
        if model_scores:
            scores[model] = pd.concat(model_scores, names=["forecast_date", "target", "target_end_date"])

    if not scores:
        return pd.DataFrame()

    all_scores = pd.concat(scores, names=["model", "forecast_date", "target", "target_end_date"]).reset_index()
    id_vars = ['model', 'forecast_date', 'target', 'target_end_date', 'wis_type']
    location_columns = [col for col in all_scores.columns if col not in id_vars]
    tidy = pd.melt(all_scores, id_vars=id_vars, value_vars=location_columns, var_name='location', value_name='value')
    return tidy


def compute_relative_scores(
    absolute_scores: pd.DataFrame,
    baseline_model: str,
) -> pd.DataFrame:
    """
    Compute relative WIS (wis_total only) vs a baseline model, aligned on
    forecast_date, target, target_end_date, location.
    """
    if absolute_scores.empty:
        return absolute_scores.copy()

    base = absolute_scores[absolute_scores["model"] == baseline_model]
    if base.empty:
        raise ValueError(f"Baseline model '{baseline_model}' not found in scores")

    rel = absolute_scores[absolute_scores["wis_type"] == "wis_total"].copy()
    rel = pd.merge(
        rel,
        base,
        on=['forecast_date', 'target', 'target_end_date', 'wis_type', 'location'],
        suffixes=("", "_baseline"),
    )
    rel['value'] = rel['value'] / rel['value_baseline']
    rel = rel.drop(["value_baseline", "model_baseline"], axis=1)
    return rel

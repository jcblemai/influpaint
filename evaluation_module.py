"""
Evaluation Module - Core data structures, scoring, and plotting functions.

This module provides a clean interface for:
- Loading and structuring forecast data (hubverse CSV format)
- Computing WIS scores (delegating to evaluate_deprecated.py)  
- Plotting results with flexible model grouping and coloring
- Supporting multiple forecast dates, horizons, and aggregation strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Union
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import ground-truth scoring helpers (WIS, hub-compatible scorer)
import evaluate_deprecated as edep


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


# ================================
# PLOTTING FUNCTIONS
# ================================

def create_heatmap_plot(
    scores_df: pd.DataFrame,
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    center: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    location_filter: str = "US",
    wis_type: str = "wis_total",
    aggregation: str = "mean"  # "mean" or "sum"
) -> None:
    """
    Create standardized heatmap plot for WIS scores.
    
    Args:
        scores_df: Tidy scores dataframe 
        dataset: ForecastDataset with model display names and groups
        group_colors: Dictionary mapping group names to colors
        title: Plot title
        filename: Output filename
        save_dir: Output directory
        missing_counts: Optional dict of missing forecast counts per model
        center, vmin, vmax: Colormap parameters
        location_filter: Location to filter ("US" or "ALL" for all locations summed)
        wis_type: Type of WIS score to plot
        aggregation: How to aggregate across forecast dates ("mean" or "sum")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Filter and aggregate data
    if location_filter == "ALL":
        # Sum across all locations
        plot_data = (
            scores_df[scores_df["wis_type"] == wis_type]
            .groupby(["model", "forecast_date", "target"], as_index=False)["value"]
            .sum()
        )
    else:
        # Filter to specific location
        plot_data = scores_df[
            (scores_df["location"] == location_filter) & 
            (scores_df["wis_type"] == wis_type)
        ]
    
    if plot_data.empty:
        print(f"No data for heatmap: {filename}")
        return
    
    # Pivot for heatmap
    if aggregation == "mean":
        heatmap_data = plot_data.pivot_table(
            index="model", 
            columns=["forecast_date", "target"], 
            values="value", 
            aggfunc="mean"
        ).fillna(np.nan)
    else:
        heatmap_data = plot_data.pivot_table(
            index="model", 
            columns=["forecast_date", "target"], 
            values="value", 
            aggfunc="sum"
        ).fillna(np.nan)
    
    if heatmap_data.empty or heatmap_data.shape[1] == 0:
        print(f"Empty heatmap data: {filename}")
        return
    
    # Sort by total score
    row_order = heatmap_data.sum(axis=1, skipna=True).sort_values().index
    heatmap_data = heatmap_data.loc[row_order]
    
    # Create plot
    fig, ax = plt.subplots(
        figsize=(max(12, heatmap_data.shape[1] * 0.5), max(8, heatmap_data.shape[0] * 0.4))
    )
    
    # Configure colormap
    if center is not None:
        cmap = sns.diverging_palette(150, 300, as_cmap=True, center="light")
        sns.heatmap(heatmap_data, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax, 
                   annot=False, fmt=".2f", linewidths=0.5)
    else:
        sns.heatmap(heatmap_data, ax=ax, cmap="viridis", 
                   annot=False, fmt=".2f", linewidths=0.5)
    
    # Create display labels with missing counts
    model_labels = []
    for model_id in heatmap_data.index:
        info = model_info.get(model_id, {"display_name": model_id, "group": "unknown"})
        missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
        
        if missing_count > 0:
            label = f"{info['display_name']}\nmissing:{missing_count}"
        else:
            label = info['display_name']
        model_labels.append(label)
    
    # Set y-tick labels
    ax.set_yticklabels(model_labels, fontsize=8)
    
    # Apply model coloring
    for i, (model_id, label) in enumerate(zip(heatmap_data.index, ax.get_yticklabels())):
        info = model_info.get(model_id, {"group": "unknown"})
        missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
        
        if missing_count > 0:
            label.set_color('red')
        else:
            color = group_colors.get(info['group'], 'gray')
            label.set_color(color)
    
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_component_plot(
    scores_df: pd.DataFrame,
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    location_filter: str = "US"
) -> None:
    """
    Create WIS component breakdown plot (sharpness, overprediction, underprediction).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Filter and aggregate data
    if location_filter == "ALL":
        # Sum across all locations
        comp_data = (
            scores_df.groupby(["model", "wis_type"], as_index=False)["value"]
            .sum()
            .pivot(index="model", columns="wis_type", values="value")
            .fillna(0.0)
        )
    else:
        # Filter to specific location
        comp_data = (
            scores_df[scores_df["location"] == location_filter]
            .pivot_table(index="model", columns="wis_type", values="value", aggfunc="sum")
            .fillna(0.0)
        )
    
    if comp_data.empty:
        print(f"No component data: {filename}")
        return
    
    # Sort by total WIS
    if "wis_total" in comp_data.columns:
        comp_data = comp_data.sort_values("wis_total")
        comp_data = comp_data[comp_data["wis_total"] > 0]  # Remove zero rows
    
    # Select components to plot
    components = [c for c in ["wis_total", "wis_sharpness", "wis_calibration", 
                             "wis_overprediction", "wis_underprediction"] 
                 if c in comp_data.columns]
    
    if not components or comp_data.empty:
        print(f"No valid components: {filename}")
        return
    
    to_plot = comp_data[components]
    
    # Create plot
    fig, axes = plt.subplots(
        1, len(components), 
        figsize=(4 * len(components), max(8, 0.3 * len(to_plot)))
    )
    if len(components) == 1:
        axes = [axes]
    
    for ax, component in zip(axes, components):
        ax.scatter(to_plot[component], np.arange(len(to_plot)), s=8)
        ax.set_yticks(np.arange(len(to_plot)))
        
        # Create labels with missing counts
        model_labels = []
        for model_id in to_plot.index:
            info = model_info.get(model_id, {"display_name": model_id, "group": "unknown"})
            missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
            
            if missing_count > 0:
                label = f"{info['display_name']}\nmissing:{missing_count}"
            else:
                label = info['display_name']
            model_labels.append(label)
        
        ax.set_yticklabels(model_labels, fontsize=6)
        
        # Apply model coloring
        for i, (model_id, label) in enumerate(zip(to_plot.index, ax.get_yticklabels())):
            info = model_info.get(model_id, {"group": "unknown"})
            missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
            
            if missing_count > 0:
                label.set_color('red')
            else:
                color = group_colors.get(info['group'], 'gray')
                label.set_color(color)
        
        ax.set_title(component.replace("wis_", "").title())
    
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close(fig)


def create_time_series_plot(
    scores_df: pd.DataFrame,
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    location_filter: str = "US",
    wis_type: str = "wis_total",
    top_n: int = 10,
    is_relative: bool = False
) -> None:
    """
    Create time series plot showing performance over forecast dates.
    Creates 2x2 subplots for each horizon to avoid overlap.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Filter data
    if location_filter == "ALL":
        # Sum across all locations for each model/date
        ts_data = (
            scores_df[scores_df["wis_type"] == wis_type]
            .groupby(["model", "forecast_date", "target", "horizon"], as_index=False)["value"]
            .sum()
        )
    else:
        ts_data = scores_df[
            (scores_df["location"] == location_filter) & 
            (scores_df["wis_type"] == wis_type)
        ]
    
    if ts_data.empty:
        print(f"No time series data: {filename}")
        return
    
    # Extract horizon information from target column (e.g., "0 wk ahead" -> 0)
    ts_data['horizon'] = ts_data['target'].str.extract('(\d+)').astype(int)
    
    # Get unique horizons
    horizons = sorted(ts_data['horizon'].unique())
    if len(horizons) == 0:
        print(f"No horizons found in data: {filename}")
        return
    
    # Get top models per group (instead of overall top models)
    model_avg = ts_data.groupby("model")["value"].mean()
    top_models = []
    
    # Get unique groups
    groups = set(info['group'] for info in model_info.values())
    
    for group in groups:
        group_models = [m for m, info in model_info.items() if info['group'] == group]
        group_avg = model_avg[model_avg.index.isin(group_models)]
        
        if len(group_avg) > 0:
            if is_relative:
                # For relative scores, best models have values closest to 1
                group_top = group_avg.iloc[(group_avg - 1).abs().argsort()[:top_n]].index.tolist()
            else:
                # For absolute scores, best models have lowest values
                group_top = group_avg.nsmallest(top_n).index.tolist()
            
            top_models.extend(group_top)
    
    # Generate unique colors for each model within group color family  
    import matplotlib.colors as mcolors
    
    model_colors = {}
    for group in groups:
        group_models = [m for m in top_models if model_info.get(m, {}).get('group') == group]
        if group_models:
            base_color = group_colors.get(group, 'gray')
            if len(group_models) == 1:
                # Single model: use base color
                model_colors[group_models[0]] = base_color
            else:
                # Multiple models: create variations of base color
                base_rgb = mcolors.to_rgb(base_color)
                # Create lighter and darker variants
                for j, model in enumerate(group_models):
                    if j == 0:
                        model_colors[model] = base_color  # Original color for first model
                    else:
                        # Create variations by adjusting brightness
                        factor = 0.7 + 0.6 * (j / len(group_models))  # Range from 0.7 to 1.3
                        varied_rgb = tuple(min(1.0, c * factor) for c in base_rgb)
                        model_colors[model] = varied_rgb
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, horizon in enumerate(horizons[:4]):  # Max 4 horizons for 2x2
        ax = axes[i]
        horizon_data = ts_data[ts_data['horizon'] == horizon]
        
        if horizon_data.empty:
            ax.set_title(f"Horizon {horizon}: No data")
            continue
        
        for model_id in top_models:
            model_data = horizon_data[horizon_data["model"] == model_id].copy()
            if not model_data.empty:
                # Sort by forecast date
                model_data = model_data.sort_values("forecast_date")
                
                info = model_info.get(model_id, {"display_name": model_id, "group": "unknown"})
                color = model_colors.get(model_id, 'gray')
                missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
                
                # Create label (only show on first subplot to avoid duplication)
                if i == 0:
                    if missing_count > 0:
                        label = f"{info['display_name']} (missing:{missing_count})"
                        linestyle = '--'
                        alpha = 0.7
                    else:
                        label = info['display_name']
                        linestyle = '-'
                        alpha = 1.0
                else:
                    label = None  # No label for other subplots
                    linestyle = '--' if missing_count > 0 else '-'
                    alpha = 0.7 if missing_count > 0 else 1.0
                
                ax.plot(model_data["forecast_date"], model_data["value"], 
                       marker='o', label=label, color=color, linestyle=linestyle, 
                       linewidth=2, alpha=alpha, markersize=3)
        
        # Add reference line for relative plots
        if is_relative:
            ax.axhline(y=1, color='red', linestyle=':', alpha=0.7, 
                      label='Baseline (WIS=1)' if i == 0 else None)
        
        ax.set_xlabel('Forecast Date')
        ax.set_ylabel('Relative WIS' if is_relative else 'Absolute WIS')
        ax.set_title(f"Horizon {horizon} weeks ahead")
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for i in range(len(horizons), 4):
        axes[i].set_visible(False)
    
    # Add legend to the first subplot
    if len(top_models) > 0:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig)

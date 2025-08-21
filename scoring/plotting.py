"""
Forecast Evaluation Plotting - Visualization functions for forecast performance analysis.

This module provides specialized plotting functions for forecast evaluation:
- forecast_scores_heatmap: Scores across models and forecast dates
- forecast_components_breakdown: Component breakdown of scoring metrics  
- forecast_performance_timeseries: Performance over time with horizon subplots
- Supports flexible model grouping, coloring, and multiple scoring metrics
"""

from typing import Dict, List, Optional, Union
import os
import colorsys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from .evaluation import ForecastDataset, ScoringResults


def forecast_scores_heatmap(
    scores: Union[pd.DataFrame, ScoringResults],
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
    scoring_metric: str = "wis_total",
    aggregation: str = "mean"  # "mean" or "sum"
) -> None:
    """
    Create heatmap showing forecast scoring metrics across models and dates.
    
    Args:
        scores: Tidy forecast scores dataframe or ScoringResults object
        dataset: ForecastDataset with model display names and groups
        group_colors: Dictionary mapping model group names to colors
        title: Plot title
        filename: Output filename
        save_dir: Output directory
        missing_counts: Optional dict of missing forecast counts per model
        center, vmin, vmax: Colormap parameters
        location_filter: Location to filter ("US" or "ALL" for all locations summed)
        scoring_metric: Type of forecast scoring metric to plot
        aggregation: How to aggregate across forecast dates ("mean" or "sum")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle both ScoringResults and DataFrame inputs
    if isinstance(scores, ScoringResults):
        scores_df = scores.to_tidy()
        if missing_counts is None:
            missing_counts = scores.meta.get('missing_counts', {})
    else:
        scores_df = scores
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Filter and aggregate data
    if location_filter == "ALL":
        # Sum across all locations
        plot_data = (
            scores_df[scores_df["scoring_metric"] == scoring_metric]
            .groupby(["model", "forecast_date", "target"], as_index=False)["value"]
            .sum()
        )
    else:
        # Filter to specific location
        plot_data = scores_df[
            (scores_df["location"] == location_filter) & 
            (scores_df["scoring_metric"] == scoring_metric)
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


def forecast_components_breakdown(
    scores: Union[pd.DataFrame, ScoringResults],
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    location_filter: str = "US",
    component_order: Optional[List[str]] = None,
    sort_by: str = "wis_total"
) -> None:
    """
    Create breakdown plot showing forecast scoring metric components (e.g., WIS: sharpness, overprediction, underprediction).
    
    Args:
        component_order: Optional list specifying column order (e.g., ["wis_total", "wis_sharpness", "wis_overprediction", "wis_underprediction"])
        sort_by: Column name to sort models by (default: "wis_total")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle both ScoringResults and DataFrame inputs
    if isinstance(scores, ScoringResults):
        scores_df = scores.to_tidy()
        if missing_counts is None:
            missing_counts = scores.meta.get('missing_counts', {})
    else:
        scores_df = scores
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Filter and aggregate data
    if location_filter == "ALL":
        # Sum across all locations
        comp_data = (
            scores_df.groupby(["model", "scoring_metric"], as_index=False)["value"]
            .sum()
            .pivot(index="model", columns="scoring_metric", values="value")
            .fillna(0.0)
        )
    else:
        # Filter to specific location
        comp_data = (
            scores_df[scores_df["location"] == location_filter]
            .pivot_table(index="model", columns="scoring_metric", values="value", aggfunc="sum")
            .fillna(0.0)
        )
    
    if comp_data.empty:
        print(f"No component data: {filename}")
        return
    
    # Sort by specified column
    if sort_by in comp_data.columns:
        comp_data = comp_data.sort_values(sort_by)
    else:
        # Fallback to total column if sort_by not found
        total_cols = [c for c in comp_data.columns if "total" in c.lower()]
        if total_cols:
            comp_data = comp_data.sort_values(total_cols[0])
    
    # Select components to plot in specified order
    if component_order is not None:
        # Use specified order, but only include columns that exist
        components = [c for c in component_order if c in comp_data.columns]
    else:
        # Default order: put total first, then others
        total_cols = [c for c in comp_data.columns if "total" in c.lower()]
        other_cols = [c for c in comp_data.columns if c not in total_cols]
        components = total_cols + sorted(other_cols)
    
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
        
        # Clean up component name for title
        clean_name = component.replace("wis_", "").replace("_", " ").title()
        ax.set_title(clean_name)
    
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close(fig)


def forecast_performance_timeseries(
    scores: Union[pd.DataFrame, ScoringResults],
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    location_filter: str = "US",
    scoring_metric: str = "wis_total",
    top_n: int = 10,
    is_relative: bool = False
) -> None:
    """
    Create time series plot showing forecast model performance over time.
    Creates 2x2 subplots for different forecast horizons to avoid overlap.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle both ScoringResults and DataFrame inputs
    if isinstance(scores, ScoringResults):
        scores_df = scores.to_tidy()
        if missing_counts is None:
            missing_counts = scores.meta.get('missing_counts', {})
    else:
        scores_df = scores
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Filter data
    if location_filter == "ALL":
        # Sum across all locations for each model/date
        ts_data = (
            scores_df[scores_df["scoring_metric"] == scoring_metric]
            .groupby(["model", "forecast_date", "target", "horizon"], as_index=False)["value"]
            .sum()
        )
    else:
        ts_data = scores_df[
            (scores_df["location"] == location_filter) & 
            (scores_df["scoring_metric"] == scoring_metric)
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
    
    # Generate unique colors and markers for each model within group color family  
    model_colors = {}
    model_markers = {}
    marker_cycle = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8']
    
    for group in groups:
        group_models = [m for m in top_models if model_info.get(m, {}).get('group') == group]
        if group_models:
            base_color = group_colors.get(group, 'gray')
            
            if len(group_models) == 1:
                # Single model: use base color and circle marker
                model_colors[group_models[0]] = base_color
                model_markers[group_models[0]] = 'o'
            else:
                # Multiple models: create variations using both color tones and markers
                base_rgb = mcolors.to_rgb(base_color)
                
                for j, model in enumerate(group_models):
                    # Assign different markers first (most distinguishable)
                    model_markers[model] = marker_cycle[j % len(marker_cycle)]
                    
                    if j == 0:
                        # First model: use base color
                        model_colors[model] = base_color
                    else:
                        # Other models: create color variations
                        # Use HSV color space for better tone variations
                        hsv = colorsys.rgb_to_hsv(*base_rgb)
                        
                        # Vary saturation and value to create distinguishable tones
                        if j % 2 == 1:
                            # Odd indices: darker/more saturated
                            new_s = min(1.0, hsv[1] * 1.2)
                            new_v = max(0.3, hsv[2] * 0.8)
                        else:
                            # Even indices: lighter/less saturated
                            new_s = max(0.4, hsv[1] * 0.7)
                            new_v = min(1.0, hsv[2] * 1.1)
                        
                        varied_rgb = colorsys.hsv_to_rgb(hsv[0], new_s, new_v)
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
                marker = model_markers.get(model_id, 'o')
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
                       marker=marker, label=label, color=color, linestyle=linestyle, 
                       linewidth=2, alpha=alpha, markersize=5, markeredgewidth=1, 
                       markeredgecolor='white')
        
        # Add reference line for relative plots
        if is_relative:
            ax.axhline(y=1, color='red', linestyle=':', alpha=0.7, 
                      label='Baseline=1' if i == 0 else None)
        
        ax.set_xlabel('Forecast Date')
        ax.set_ylabel(f'Relative {scoring_metric}' if is_relative else f'Absolute {scoring_metric}')
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


def model_performance_summary(
    scores: Union[pd.DataFrame, ScoringResults],
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    metrics: Optional[List[str]] = None,
    sort_by: str = "coverage_95"
) -> None:
    """
    Create summary plot showing per-model metrics (coverage, completion rate, etc.).
    
    Args:
        scores: ScoringResults object or tidy DataFrame
        dataset: ForecastDataset with model info
        group_colors: Dictionary mapping model group names to colors
        title: Plot title
        filename: Output filename
        save_dir: Output directory
        missing_counts: Optional dict of missing forecast counts per model
        metrics: List of metric names to plot (default: all per-model metrics found)
        sort_by: Metric to sort models by (default: "coverage_95")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle both ScoringResults and DataFrame inputs
    if isinstance(scores, ScoringResults):
        if scores.model_metrics.empty:
            print(f"No per-model metrics available for: {filename}")
            return
        model_data = scores.model_metrics.copy()
        if missing_counts is None:
            missing_counts = scores.meta.get('missing_counts', {})
    else:
        # Assume it's a tidy DataFrame, filter to per-model grain
        if 'grain' in scores.columns:
            model_data = scores[scores['grain'] == 'per_model'].copy()
        else:
            print(f"Cannot identify per-model metrics in DataFrame for: {filename}")
            return
    
    if model_data.empty:
        print(f"No per-model data for: {filename}")
        return
    
    # Filter to requested metrics
    if metrics is not None:
        model_data = model_data[model_data['scoring_metric'].isin(metrics)]
    
    if model_data.empty:
        print(f"No matching metrics found for: {filename}")
        return
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Aggregate across horizons (mean)
    agg_data = (
        model_data.groupby(['model', 'scoring_metric'])['value']
        .mean()
        .reset_index()
        .pivot(index='model', columns='scoring_metric', values='value')
        .fillna(0.0)
    )
    
    if agg_data.empty:
        print(f"No aggregated data for: {filename}")
        return
    
    # Sort by specified metric
    if sort_by in agg_data.columns:
        agg_data = agg_data.sort_values(sort_by, ascending=False)  # Higher is better for coverage/completion
    
    # Select metrics to plot
    available_metrics = agg_data.columns.tolist()
    
    # Create plot
    fig, axes = plt.subplots(
        1, len(available_metrics), 
        figsize=(4 * len(available_metrics), max(8, 0.3 * len(agg_data)))
    )
    if len(available_metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        metric_data = agg_data[metric]
        ax.scatter(metric_data, np.arange(len(metric_data)), s=40, alpha=0.7)
        ax.set_yticks(np.arange(len(metric_data)))
        
        # Create labels with missing counts
        model_labels = []
        for model_id in metric_data.index:
            info = model_info.get(model_id, {"display_name": model_id, "group": "unknown"})
            missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
            
            if missing_count > 0:
                label = f"{info['display_name']}\nmissing:{missing_count}"
            else:
                label = info['display_name']
            model_labels.append(label)
        
        ax.set_yticklabels(model_labels, fontsize=8)
        
        # Apply model coloring
        for i, (model_id, label) in enumerate(zip(metric_data.index, ax.get_yticklabels())):
            info = model_info.get(model_id, {"group": "unknown"})
            missing_count = missing_counts.get(model_id, 0) if missing_counts else 0
            
            if missing_count > 0:
                label.set_color('red')
            else:
                color = group_colors.get(info['group'], 'gray')
                label.set_color(color)
        
        # Format metric name for title
        clean_name = metric.replace("_", " ").title()
        ax.set_title(clean_name)
        ax.set_xlabel("Value")
        
        # Add reference lines for coverage metrics
        if "coverage" in metric.lower():
            if "gap" in metric.lower():
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Target (0)')
            else:
                ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='Target (95%)')
            ax.legend()
        elif "completion" in metric.lower():
            ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Complete (100%)')
            ax.legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig)


def model_horizon_heatmap(
    scores: Union[pd.DataFrame, ScoringResults],
    dataset: ForecastDataset,
    group_colors: Dict[str, str],
    title: str,
    filename: str,
    save_dir: str,
    missing_counts: Optional[Dict[str, int]] = None,
    metric: str = "coverage_95"
) -> None:
    """
    Create heatmap showing per-model metric values across horizons.
    
    Args:
        scores: ScoringResults object or tidy DataFrame
        dataset: ForecastDataset with model info  
        group_colors: Dictionary mapping model group names to colors
        title: Plot title
        filename: Output filename
        save_dir: Output directory
        missing_counts: Optional dict of missing forecast counts per model
        metric: Specific metric to plot across horizons
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle both ScoringResults and DataFrame inputs
    if isinstance(scores, ScoringResults):
        if scores.model_metrics.empty:
            print(f"No per-model metrics available for: {filename}")
            return
        model_data = scores.model_metrics.copy()
        if missing_counts is None:
            missing_counts = scores.meta.get('missing_counts', {})
    else:
        # Assume it's a tidy DataFrame, filter to per-model grain
        if 'grain' in scores.columns:
            model_data = scores[scores['grain'] == 'per_model'].copy()
        else:
            print(f"Cannot identify per-model metrics in DataFrame for: {filename}")
            return
    
    if model_data.empty:
        print(f"No per-model data for: {filename}")
        return
    
    # Filter to requested metric
    metric_data = model_data[model_data['scoring_metric'] == metric]
    
    if metric_data.empty:
        print(f"No data for metric '{metric}' in: {filename}")
        return
    
    # Create model info mapping
    model_info = {r.model: {"group": r.group, "display_name": r.display_name} for r in dataset.records}
    
    # Pivot for heatmap
    heatmap_data = metric_data.pivot(index='model', columns='horizon', values='value').fillna(np.nan)
    
    if heatmap_data.empty:
        print(f"No heatmap data for metric '{metric}' in: {filename}")
        return
    
    # Sort by mean across horizons
    row_means = heatmap_data.mean(axis=1, skipna=True)
    heatmap_data = heatmap_data.loc[row_means.sort_values(ascending=False).index]
    
    # Create plot
    fig, ax = plt.subplots(
        figsize=(max(8, heatmap_data.shape[1] * 0.8), max(6, heatmap_data.shape[0] * 0.4))
    )
    
    # Configure colormap based on metric type
    if "gap" in metric.lower():
        # Gap metrics: center around 0
        vmax = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
        sns.heatmap(heatmap_data, ax=ax, cmap="RdBu_r", center=0, 
                   vmin=-vmax, vmax=vmax, annot=True, fmt=".3f", linewidths=0.5)
    else:
        # Coverage/completion metrics: 0 to 1 scale
        sns.heatmap(heatmap_data, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                   annot=True, fmt=".3f", linewidths=0.5)
    
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
    ax.set_xlabel("Forecast Horizon (weeks)")
    ax.set_ylabel("Model")
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig)
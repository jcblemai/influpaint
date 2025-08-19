# %%
# Evaluation Module - Core evaluation and plotting functionality
# This module provides a clean interface for evaluating models and creating plots
# with support for custom groupings and color schemes

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

sns.set_theme()

# %%
@dataclass
class PlotConfig:
    """Configuration for plotting appearance and behavior."""
    figsize_base: Tuple[int, int] = (12, 8)
    dpi: int = 200
    font_size: int = 8
    missing_count_color: str = 'red'
    baseline_color: str = 'red'
    default_colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.default_colors is None:
            self.default_colors = {
                'influpaint': 'green',
                'flusight': 'blue',
                'baseline': 'red',
                'missing': 'red'
            }


@dataclass 
class GroupConfig:
    """Configuration for model grouping and coloring."""
    group_fn: Callable[[str], str]  # Function to determine group from model name
    color_map: Dict[str, str]  # Map group names to colors
    group_labels: Dict[str, str] = None  # Optional: map group names to display labels
    
    def __post_init__(self):
        if self.group_labels is None:
            self.group_labels = {}


class ModelEvaluator:
    """Core evaluation and plotting functionality."""
    
    def __init__(self, plot_config: PlotConfig = None):
        """
        Initialize the evaluator.
        
        Args:
            plot_config: Configuration for plot appearance
        """
        self.plot_config = plot_config or PlotConfig()
        self.logger = logging.getLogger(__name__)
    
    def determine_model_group(self, model_name: str, group_config: GroupConfig = None) -> str:
        """
        Determine the group for a model based on configuration.
        
        Args:
            model_name: Name of the model
            group_config: Grouping configuration. If None, uses default InfluPaint/FluSight grouping
            
        Returns:
            Group identifier
        """
        if group_config is None:
            # Default grouping: InfluPaint vs FluSight
            if model_name.startswith('i') and '::' in model_name:
                return 'influpaint'
            else:
                return 'flusight'
        else:
            return group_config.group_fn(model_name)
    
    def get_model_color(self, model_name: str, group_config: GroupConfig = None) -> str:
        """
        Get the color for a model based on its group.
        
        Args:
            model_name: Name of the model
            group_config: Grouping configuration
            
        Returns:
            Color string
        """
        group = self.determine_model_group(model_name, group_config)
        
        if group_config is not None and group in group_config.color_map:
            return group_config.color_map[group]
        elif group in self.plot_config.default_colors:
            return self.plot_config.default_colors[group]
        else:
            # Fallback to default color
            return self.plot_config.default_colors.get('flusight', 'blue')
    
    def create_model_labels_with_missing(self, models: List[str], model_missing_counts: Dict[str, int]) -> List[str]:
        """Create y-tick labels with missing counts for long model names."""
        labels = []
        for model in models:
            missing_count = model_missing_counts.get(model, 0)
            
            # Handle long model names by wrapping
            if len(model) > 60:
                parts = model.split("::")
                if len(parts) >= 6:
                    line1 = "::".join(parts[:2])
                    line2 = "::".join(parts[2:5]) 
                    line3 = "::".join(parts[5:])
                    base_label = f"{line1}\\n{line2}\\n{line3}"
                else:
                    base_label = model
            else:
                base_label = model
            
            if missing_count > 0:
                labels.append(f"{base_label} missing:{missing_count}")
            else:
                labels.append(base_label)
        return labels
    
    def apply_model_colors_to_labels(self, ax, model_names: List[str], model_missing_counts: Dict[str, int], 
                                   group_config: GroupConfig = None):
        """Apply colors to y-tick labels based on model group and missing data."""
        for i, label in enumerate(ax.get_yticklabels()):
            model_name = model_names[i]
            text = label.get_text()
            
            if "missing:" in text:
                label.set_color(self.plot_config.missing_count_color)
            else:
                model_color = self.get_model_color(model_name, group_config)
                label.set_color(model_color)
    
    def add_missing_count_annotations(self, ax, model_names: List[str], model_missing_counts: Dict[str, int]):
        """Add red missing count annotations to the right of y-labels."""
        for i, label in enumerate(ax.get_yticklabels()):
            model_name = model_names[i]
            text = label.get_text()
            
            if "missing:" in text:
                # Split into lines and color the missing line red
                lines = text.split('\\n')
                clean_text = '\\n'.join([line for line in lines if not line.startswith('missing:')])
                missing_line = next((line for line in lines if line.startswith('missing:')), None)
                
                label.set_text(clean_text)
                if missing_line:
                    # Add red text for missing count
                    ax.text(-0.02, i, missing_line, color=self.plot_config.missing_count_color, 
                           va='center', fontsize=label.get_fontsize(), ha='right', 
                           transform=ax.get_yaxis_transform())
    
    def create_heatmap_plot(self, data: pd.DataFrame, title: str, filename: str, save_dir: str, 
                           model_missing_counts: Dict[str, int], group_config: GroupConfig = None,
                           cmap: str = "viridis", center: float = None, vmin: float = None, vmax: float = None):
        """
        Create a standardized heatmap plot with custom grouping support.
        
        Args:
            data: Pivot table with models as index and dates/targets as columns
            title: Plot title
            filename: Output filename
            save_dir: Directory to save plot
            model_missing_counts: Dictionary of missing counts per model
            group_config: Configuration for model grouping and colors
            cmap: Colormap name
            center: Center value for diverging colormap
            vmin, vmax: Color scale limits
        """
        if data.empty or data.shape[1] == 0:
            self.logger.warning(f"Empty data for plot {filename}, skipping")
            return
            
        # Sort by total score
        order = data.sum(axis=1, skipna=True).sort_values().index
        data_sorted = data.loc[order]
        
        # Create y-tick labels with missing counts
        ytick_labels = self.create_model_labels_with_missing(data_sorted.index.tolist(), model_missing_counts)
        
        # Create plot
        fig_width = max(self.plot_config.figsize_base[0], data_sorted.shape[1] * 0.5)
        fig_height = max(self.plot_config.figsize_base[1], data_sorted.shape[0] * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap
        if center is not None:
            cmap_obj = sns.diverging_palette(150, 300, as_cmap=True, center="light")
            sns.heatmap(data_sorted, annot=False, fmt=".2f", linewidths=0.5, ax=ax, 
                       center=center, cmap=cmap_obj, vmin=vmin, vmax=vmax)
        else:
            sns.heatmap(data_sorted, annot=False, fmt=".2f", linewidths=0.5, ax=ax, cmap=cmap)
        
        # Set custom y-tick labels
        ax.set_yticklabels([label.replace(" missing:", "\\nmissing:") for label in ytick_labels])
        
        # Apply colors and missing annotations
        self.apply_model_colors_to_labels(ax, data_sorted.index.tolist(), model_missing_counts, group_config)
        self.add_missing_count_annotations(ax, data_sorted.index.tolist(), model_missing_counts)
        
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=self.plot_config.dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved heatmap plot: {filename}")
    
    def create_component_plots(self, scores_df: pd.DataFrame, title_prefix: str, filename_prefix: str, 
                              save_dir: str, model_missing_counts: Dict[str, int], 
                              group_config: GroupConfig = None):
        """
        Create WIS component scatter plots.
        
        Args:
            scores_df: Long-format scores dataframe
            title_prefix: Prefix for plot titles
            filename_prefix: Prefix for filenames
            save_dir: Directory to save plots
            model_missing_counts: Dictionary of missing counts per model
            group_config: Configuration for model grouping and colors
        """
        # Pivot to get components by model
        comp_data = (
            scores_df.pivot_table(index=["model", "wis_type"], values="value", aggfunc="sum")
            .reset_index()
            .pivot(index="model", columns="wis_type", values="value")
            .fillna(0.0)
        )
        
        # Remove models with no data
        if "wis_total" in comp_data.columns:
            comp_data = comp_data[comp_data["wis_total"] > 0]
        
        if comp_data.empty:
            self.logger.warning(f"No component data for {filename_prefix}, skipping")
            return
        
        # Sort by total WIS
        comp_data = comp_data.sort_values("wis_total") if "wis_total" in comp_data.columns else comp_data
        
        # Select components to plot
        available_components = ["wis_total", "wis_sharpness", "wis_calibration", "wis_overprediction", "wis_underprediction"]
        to_plot = comp_data[[c for c in available_components if c in comp_data.columns]]
        
        if to_plot.empty:
            return
        
        # Create simplified labels (just model names, no missing counts in main labels)
        simple_labels = []
        for model in to_plot.index.tolist():
            # Clean up long model names - just keep the main identifier
            if len(model) > 40:
                # For InfluPaint models, keep just scenario and config
                if model.startswith('i') and '::' in model:
                    parts = model.split("::")
                    if len(parts) >= 2:
                        # Keep scenario (i806) and last part (config)
                        simple_label = f"{parts[0]}::{parts[-1]}"
                    else:
                        simple_label = model[:40] + "..."
                else:
                    simple_label = model[:40] + "..."
            else:
                simple_label = model
            simple_labels.append(simple_label)
        
        # Calculate figure dimensions with more space for y-labels
        fig_width = 6 * to_plot.shape[1]  # Increased width per subplot
        fig_height = max(10, 0.4 * len(to_plot))  # Increased height per model
        fig, axes = plt.subplots(1, to_plot.shape[1], figsize=(fig_width, fig_height))
        
        if to_plot.shape[1] == 1:
            axes = [axes]
        
        for ax, col in zip(axes, to_plot.columns):
            ax.scatter(to_plot[col], np.arange(len(to_plot)), s=12)  # Slightly larger points
            ax.set_yticks(np.arange(len(to_plot)))
            
            # Set simplified labels with larger font
            ax.set_yticklabels(simple_labels, fontsize=9)
            
            # Apply colors
            for i, label in enumerate(ax.get_yticklabels()):
                model_name = to_plot.index[i]
                missing_count = model_missing_counts.get(model_name, 0)
                
                if missing_count > 0:
                    label.set_color(self.plot_config.missing_count_color)
                else:
                    model_color = self.get_model_color(model_name, group_config)
                    label.set_color(model_color)
            
            ax.set_title(col.replace('wis_', '').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Adjust layout to give more space for y-labels
        fig.suptitle(f"{title_prefix}: WIS Components", fontsize=14)
        plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.1)  # More left margin
        fig.savefig(os.path.join(save_dir, f"{filename_prefix}_wis_components.png"), 
                   dpi=self.plot_config.dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved component plot: {filename_prefix}_wis_components.png")
    
    def create_time_series_plot(self, timeseries_data: pd.DataFrame, title: str, filename: str, 
                               save_dir: str, model_missing_counts: Dict[str, int], 
                               group_config: GroupConfig = None, is_relative: bool = False,
                               top_n_models: int = 10):
        """
        Create time series plot showing model performance over time.
        
        Args:
            timeseries_data: DataFrame with columns: model, forecast_date, value, season
            title: Plot title
            filename: Output filename  
            save_dir: Directory to save plot
            model_missing_counts: Dictionary of missing counts per model
            group_config: Configuration for model grouping and colors
            is_relative: Whether this is relative WIS (affects baseline line)
            top_n_models: Number of top models to show
        """
        if timeseries_data.empty:
            self.logger.warning(f"Empty timeseries data for {filename}, skipping")
            return
        
        # Get top models by average performance
        model_avg_performance = timeseries_data.groupby("model")["value"].mean().sort_values()
        top_models = model_avg_performance.head(top_n_models).index.tolist()
        
        if not top_models:
            return
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create complete timeline
        all_dates = sorted(timeseries_data["forecast_date"].unique())
        date_to_index = {date: i for i, date in enumerate(all_dates)}
        
        # Plot each model
        for model in top_models:
            model_data = timeseries_data[timeseries_data["model"] == model].copy()
            if model_data.empty:
                continue
            
            # Create x,y coordinates (allowing gaps for missing data)
            x_vals = [date_to_index[row["forecast_date"]] for _, row in model_data.iterrows()]
            y_vals = model_data["value"].tolist()
            
            # Determine color and style
            color = self.get_model_color(model, group_config)
            missing_count = model_missing_counts.get(model, 0)
            
            # Create label
            if missing_count > 0:
                label = f"{model[:50]}... missing:{missing_count}" if len(model) > 50 else f"{model} missing:{missing_count}"
                linestyle = '--'
                alpha = 0.7
            else:
                label = f"{model[:50]}..." if len(model) > 50 else model
                linestyle = '-'
                alpha = 1.0
            
            # Plot with gaps for missing data
            ax.plot(x_vals, y_vals, marker='o', label=label, color=color, 
                   linestyle=linestyle, linewidth=2, alpha=alpha, markersize=4)
        
        # Add season boundaries if season info available
        if "season" in timeseries_data.columns:
            self._add_season_boundaries(ax, timeseries_data, all_dates, date_to_index)
        
        # Set x-axis
        tick_indices = list(range(0, len(all_dates), max(1, len(all_dates)//20)))
        tick_labels = [str(all_dates[i]) for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # Add baseline line for relative plots
        if is_relative:
            ax.axhline(y=1, color=self.plot_config.baseline_color, linestyle=':', alpha=0.7, label='Baseline (WIS=1)')
            ax.set_ylabel('Relative WIS (vs Baseline)')
        else:
            ax.set_ylabel('Absolute WIS')
        
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=self.plot_config.font_size)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, filename), dpi=self.plot_config.dpi, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved time series plot: {filename}")
    
    def _add_season_boundaries(self, ax, timeseries_data: pd.DataFrame, all_dates: List, date_to_index: Dict):
        """Add vertical lines and labels for season boundaries."""
        # Create season boundaries
        season_boundaries = []
        season_labels = []
        current_season = None
        season_start = 0
        
        for i, date in enumerate(all_dates):
            # Find which season this date belongs to
            date_season = None
            matching_rows = timeseries_data[timeseries_data["forecast_date"] == date]
            if not matching_rows.empty:
                date_season = matching_rows["season"].iloc[0]
            
            if date_season != current_season:
                if current_season is not None:
                    # Mark end of previous season
                    season_boundaries.append(i - 0.5)
                    # Add label at midpoint of season
                    mid_point = (season_start + i - 1) / 2
                    season_labels.append((mid_point, current_season))
                current_season = date_season
                season_start = i
        
        # Add final season label
        if current_season is not None:
            mid_point = (season_start + len(all_dates) - 1) / 2
            season_labels.append((mid_point, current_season))
        
        # Add vertical lines between seasons
        for boundary in season_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
        
        # Add season labels at top
        for pos, season in season_labels:
            ax.text(pos, ax.get_ylim()[1], season, ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)


# %%
# Predefined grouping configurations for common use cases

def create_influpaint_vs_flusight_config() -> GroupConfig:
    """Create default InfluPaint vs FluSight grouping."""
    def group_fn(model_name: str) -> str:
        if model_name.startswith('i') and '::' in model_name:
            return 'influpaint'
        else:
            return 'flusight'
    
    color_map = {
        'influpaint': 'green',
        'flusight': 'blue'
    }
    
    return GroupConfig(group_fn=group_fn, color_map=color_map)


def create_scenario_based_config() -> GroupConfig:
    """Create grouping based on scenario ID ranges."""
    def group_fn(model_name: str) -> str:
        try:
            if model_name.startswith('i') and '::' in model_name:
                scenario_id = int(model_name.split("::")[0][1:])  # Remove 'i' prefix
                if scenario_id < 16:
                    return 'low_scenarios'
                else:
                    return 'high_scenarios'
            else:
                return 'flusight'
        except (ValueError, IndexError):
            return 'other'
    
    color_map = {
        'low_scenarios': 'lightgreen',
        'high_scenarios': 'darkgreen', 
        'flusight': 'blue',
        'other': 'gray'
    }
    
    return GroupConfig(group_fn=group_fn, color_map=color_map)


def create_config_based_config() -> GroupConfig:
    """Create grouping based on InfluPaint config types."""
    def group_fn(model_name: str) -> str:
        try:
            if model_name.startswith('i') and '::' in model_name:
                config = model_name.split("::")[-1]
                if 'celebahq' in config:
                    return 'celebahq'
                elif 'places' in config:
                    return 'places'
                else:
                    return 'other_influpaint'
            else:
                return 'flusight'
        except (ValueError, IndexError):
            return 'other'
    
    color_map = {
        'celebahq': 'lightgreen',
        'places': 'darkgreen',
        'other_influpaint': 'orange',
        'flusight': 'blue',
        'other': 'gray'
    }
    
    return GroupConfig(group_fn=group_fn, color_map=color_map)
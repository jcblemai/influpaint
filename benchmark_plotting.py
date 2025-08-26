#!/usr/bin/env python3
"""
Generic benchmark plotting utilities for forecast evaluation.
Direct plotting functions for heatmaps, components, and timeseries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union

# Set unified style for all plots
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'figure.dpi': 200,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})

def apply_plot_styling(ax, title=None):
    """Apply consistent styling to any plot axis."""
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    if title:
        ax.set_title(title, pad=15, fontweight='medium')

def plot_heatmap(df: pd.DataFrame,
                index_cols: List[str],
                column_cols: List[str], 
                value_col: str,
                agg_func: str = 'mean',
                sort_by: str = 'mean',
                title: str = "Heatmap",
                save_path: str = None,
                missing_info: Dict[str, str] = None,
                group_colors: Dict[str, str] = None,
                cmap: str = "viridis",
                center: float = None,
                vmin: float = None,
                vmax: float = None) -> None:
    """Plot heatmap for any combination of rows/columns/values."""
    
    if df.empty:
        print(f"Empty dataframe for {title}")
        return
    
    # Create pivot table
    pivot_data = df.pivot_table(index=index_cols, columns=column_cols, 
                               values=value_col, aggfunc=agg_func).fillna(np.nan)
    
    if pivot_data.empty:
        print(f"Empty pivot data for {title}")
        return
    
    # Sort by specified method
    if sort_by == 'mean':
        sort_values = pivot_data.mean(axis=1, skipna=True)
    elif sort_by == 'sum':
        sort_values = pivot_data.sum(axis=1, skipna=True)
    elif sort_by == 'median':
        sort_values = pivot_data.median(axis=1, skipna=True)
    else:
        sort_values = pivot_data.mean(axis=1, skipna=True)
        
    pivot_data = pivot_data.loc[sort_values.sort_values().index]
    
    # Create labels with missing info and colors
    labels = []
    colors = []
    
    for idx in pivot_data.index:
        if isinstance(idx, tuple):
            main_name = idx[0]
            display_name = main_name
            group_name = idx[1] if len(idx) > 1 else None
        else:
            main_name = idx
            display_name = main_name
            group_name = None
        
        # Add missing info if available
        if missing_info and main_name in missing_info:
            missing_data = missing_info[main_name]
            missing_text = missing_data["text"] if isinstance(missing_data, dict) else missing_data
            is_critical = missing_data.get("critical", False) if isinstance(missing_data, dict) else bool(missing_data)
            
            label = f"{display_name}\n{missing_text}"
            if is_critical:
                colors.append('red')
            elif group_name and group_colors:
                colors.append(group_colors.get(group_name, 'gray'))
            else:
                colors.append('blue')
        else:
            label = display_name
            if group_name and group_colors:
                colors.append(group_colors.get(group_name, 'gray'))
            else:
                colors.append('blue')
        labels.append(label)
    
    # Create heatmap
    fig_width = max(12, pivot_data.shape[1] * 0.4)
    fig_height = max(8, pivot_data.shape[0] * 0.4)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set color parameters
    heatmap_kwargs = {'cmap': cmap, 'yticklabels': labels}
    if center is not None:
        heatmap_kwargs.update({'center': center, 'vmin': vmin, 'vmax': vmax})
    
    sns.heatmap(pivot_data, **heatmap_kwargs)
    
    # Color y-tick labels
    ax = plt.gca()
    for label, color in zip(ax.get_yticklabels(), colors):
        label.set_color(color)
    
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_components(df: pd.DataFrame, 
                   group_by: List[str], 
                   value_cols: List[str],
                   agg_func: Dict[str, str],
                   sort_by: str = None,
                   title: str = "Component Plot",
                   save_path: str = None,
                   missing_info: Dict[str, str] = None,
                   group_colors: Dict[str, str] = None,
                   reference_lines: Dict[str, Dict[str, Union[float, str]]] = None,
                   stacked: bool = False,
                   component_colors: Dict[str, str] = None) -> None:
    """Plot component metrics as scatter plots or stacked bars with shared y-axis."""
    
    if df.empty:
        print(f"Empty dataframe for {title}")
        return
    
    # Aggregate data
    plot_data = df.groupby(group_by, as_index=False).agg(agg_func)
    
    # Sort by specified column or total
    if sort_by is None:
        plot_data['_total'] = plot_data[value_cols].sum(axis=1)
        plot_data = plot_data.sort_values('_total').drop('_total', axis=1)
    else:
        plot_data = plot_data.sort_values(sort_by)
    
    # Create labels and colors
    labels = []
    colors = []
    main_group_col = group_by[0]
    color_group_col = group_by[1] if len(group_by) > 1 else None
    
    for _, row in plot_data.iterrows():
        main_name = row[main_group_col]
        display_name = main_name
        
        # Handle missing info
        if missing_info and main_name in missing_info:
            missing_data = missing_info[main_name]
            missing_text = missing_data["text"] if isinstance(missing_data, dict) else missing_data
            is_critical = missing_data.get("critical", False) if isinstance(missing_data, dict) else bool(missing_data)
            label = f"{display_name}\n{missing_text}"
        else:
            missing_text = ""
            is_critical = False
            label = display_name
        
        # Set color
        if missing_text and is_critical:
            colors.append('red')
        elif color_group_col and group_colors:
            color_val = row[color_group_col]
            colors.append(group_colors.get(color_val, 'gray'))
        else:
            colors.append('blue')
            
        labels.append(label)
    
    if stacked:
        # Create single stacked bar plot
        fig_width = max(10, 0.4 * len(plot_data))
        fig_height = max(8, 0.4 * len(plot_data))
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        
        # Create stacked bars
        y_pos = range(len(plot_data))
        left_values = np.zeros(len(plot_data))
        
        for component in value_cols:
            color = component_colors.get(component, 'gray') if component_colors else 'gray'
            ax.barh(y_pos, plot_data[component], left=left_values, 
                   color=color, label=component.title(), alpha=0.8)
            left_values += plot_data[component]
        
        apply_plot_styling(ax, title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Models")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        for j, (label, color) in enumerate(zip(ax.get_yticklabels(), colors)):
            label.set_color(color)
        
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
        
    else:
        # Create subplots for scatter mode
        fig_width = max(12, 3 * len(value_cols))
        fig_height = max(8, 0.4 * len(plot_data))
        fig, axes = plt.subplots(1, len(value_cols), figsize=(fig_width, fig_height), sharey=True)
        
        if len(value_cols) == 1:
            axes = [axes]
        
        # Original scatter plot mode
        for i, component in enumerate(value_cols):
            ax = axes[i]
            ax.scatter(plot_data[component], range(len(plot_data)), s=40, alpha=0.7, c=colors)
            
            # Add reference line
            if reference_lines and component in reference_lines:
                ref_config = reference_lines[component]
                ref_val = ref_config['value']
                ref_label = ref_config.get('label', f'Target {ref_val}')
                ref_color = ref_config.get('color', 'red')
                ref_linestyle = ref_config.get('linestyle', '--')
                ref_alpha = ref_config.get('alpha', 0.5)
                
                ax.axvline(x=ref_val, color=ref_color, linestyle=ref_linestyle, alpha=ref_alpha, label=ref_label)
                ax.legend()
            
            apply_plot_styling(ax, f"{component.title().replace('_', ' ').title()}")
            ax.set_xlabel("Score")
            
            # Set y-axis labels (only on first subplot)
            ax.set_yticks(range(len(plot_data)))
            if i == 0:
                ax.set_yticklabels(labels, fontsize=8)
                for j, (label, color) in enumerate(zip(ax.get_yticklabels(), colors)):
                    label.set_color(color)
            else:
                ax.set_yticklabels(labels, fontsize=8)
                for label in ax.get_yticklabels():
                    label.set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_timeseries(df: pd.DataFrame,
                   x_col: str,
                   y_col: str,
                   group_col: str,
                   facet_col: str = None,
                   filter_top_n: int = None,
                   title: str = "Time Series",
                   save_path: str = None,
                   group_colors: Dict[str, str] = None,
                   relative: bool = False) -> None:
    """Plot time series for any combination of x/y/grouping columns."""
    
    if df.empty:
        print(f"Empty dataframe for {title}")
        return
    
    # Filter top models if requested
    if filter_top_n:
        if 'group' in df.columns:
            top_groups = get_top_models_per_group(df, y_col, filter_top_n, relative)
        elif facet_col:
            # Get top N per facet
            top_groups = []
            for facet_val in df[facet_col].unique():
                facet_data = df[df[facet_col] == facet_val]
                group_means = facet_data.groupby(group_col)[y_col].mean().sort_values()
                top_groups.extend(group_means.head(filter_top_n).index.tolist())
            top_groups = list(set(top_groups))
        else:
            # Get top N overall
            group_means = df.groupby(group_col)[y_col].mean().sort_values()
            top_groups = group_means.head(filter_top_n).index.tolist()
        
        df = df[df[group_col].isin(top_groups)]
    
    # Create subplots if faceting
    if facet_col:
        facet_vals = sorted(df[facet_col].unique())
        n_cols = min(2, len(facet_vals))
        n_rows = (len(facet_vals) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        
        if len(facet_vals) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax]
        facet_vals = [None]
    
    # Plot data
    for i, facet_val in enumerate(facet_vals):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        if facet_col and facet_val is not None:
            facet_data = df[df[facet_col] == facet_val]
            subplot_title = f"{facet_col.title()}: {facet_val}"
        else:
            facet_data = df
            subplot_title = title
        
        # Track styling indices for consistency
        group_indices = {}
        
        for group_val in facet_data[group_col].unique():
            group_data = facet_data[facet_data[group_col] == group_val].sort_values(x_col)
            
            # Get styling
            if group_val not in group_indices:
                group_indices[group_val] = len(group_indices)
            
            group_idx = group_indices[group_val]
            
            # Assign colors and styles
            if 'group' in facet_data.columns:
                group_type = group_data['group'].iloc[0]
                color, marker, linestyle = get_model_styling(group_val, group_type, group_idx)
            else:
                # Use cycling colors if no group type
                colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD3BA', '#E0BBE4']
                color = colors[group_idx % len(colors)]
                marker = 'o'
                linestyle = '-'
            
            # Only show legend on first subplot
            label = group_val if i == 0 else None
            
            ax.plot(group_data[x_col], group_data[y_col], 
                   color=color, label=label, marker=marker, markersize=4,
                   linewidth=1.5, linestyle=linestyle)
        
        apply_plot_styling(ax, subplot_title)
        ax.set_xlabel(x_col.replace('_', ' ').title())
        
        if relative:
            ax.set_ylabel(f"Relative {y_col.upper()}")
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
        else:
            ax.set_ylabel(y_col.upper())
        
        # Only show legend on first subplot - make it transparent and overlay on plot
        if i == 0:
            legend = ax.legend(loc='upper right', framealpha=0.8, fancybox=True, 
                             bbox_to_anchor=(0.98, 0.98), fontsize=8)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_linewidth(0.5)
    
    # Hide empty subplots
    for j in range(len(facet_vals), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_model_styling(model: str, group: str, group_idx: int):
    """Get color, marker, and linestyle for a specific model."""
    dark_colors = ['#8B0000', '#006400', '#000080', '#8B008B', '#FF8C00', '#556B2F', '#8B4513', '#2F4F4F']
    pastel_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD3BA', '#E0BBE4', '#C7CEEA', '#FFDFBA']
    
    if group == 'influpaint':
        color = dark_colors[group_idx % len(dark_colors)]
        marker = 's'
        linestyle = '-'
    else:  # flusight
        color = pastel_colors[group_idx % len(pastel_colors)]
        marker = 'o'
        linestyle = '--'
    
    return color, marker, linestyle

def get_top_models_per_group(df: pd.DataFrame, metric_col: str, top_n: int = 3, relative: bool = False):
    """Get top N models per group instead of overall top N."""
    top_models = []
    
    for group in df['group'].unique():
        group_data = df[df['group'] == group]
        if group_data.empty:
            continue
            
        group_avg = group_data.groupby('model')[metric_col].mean()
        
        if relative:
            # For relative scores, best models have values closest to 1.0
            top_n_models = group_avg.iloc[(group_avg - 1.0).abs().argsort()[:top_n]].index.tolist()
        else:
            # For absolute scores, best models have lowest values
            top_n_models = group_avg.nsmallest(top_n).index.tolist()
            
        top_models.extend(top_n_models)
    
    return top_models

def get_location_display_name(location: str, reference_location: str, name_mapper=None):
    """Generic function to get display names for locations."""
    if location == reference_location:
        return f"{reference_location} National"
    elif location == "States_Sum":
        return "Sum of States"
    elif name_mapper:
        return name_mapper(location)
    else:
        return location

def plot_wis_heatmap(df: pd.DataFrame, location_filter: str, title: str,
                     relative: bool = False, baseline_model: str = 'FluSight-baseline', 
                     original_df: pd.DataFrame = None, missing_info_fn=None, group_colors=None, 
                     valid_locations: list = None):
    """Plot WIS heatmap with location filtering and relative scoring."""
    
    # Filter data based on location
    if location_filter == "US":
        plot_data = df[df['location'] == 'US'].copy()
    elif location_filter == "sum_all_states":
        if valid_locations:
            locs = valid_locations
        else:
            locs = df['location'].unique()
        plot_data = df[df['location'].isin(locs)].copy()
        plot_data = plot_data.groupby(['model', 'group', 'season', 'target_end_date', 'horizon'], as_index=False)['wis'].sum()
    else:
        plot_data = df.copy()
    
    if plot_data.empty:
        print(f"No data for heatmap: {title}")
        return
    
    # Compute relative scores if needed
    if relative and baseline_model in plot_data['model'].unique():
        baseline_data = plot_data[plot_data['model'] == baseline_model].set_index(['target_end_date', 'horizon'])['wis']
        for idx, row in plot_data.iterrows():
            key = (row['target_end_date'], row['horizon'])
            if key in baseline_data.index and baseline_data[key] > 0:
                plot_data.at[idx, 'wis'] = row['wis'] / baseline_data[key]
        score_type = "Relative WIS"
        center, vmin, vmax = 1, 0, 2
        cmap = "RdBu_r"
    else:
        score_type = "Absolute WIS"
        center, vmin, vmax = None, None, None
        cmap = "viridis"
    
    # Create pivot table
    pivot_data = plot_data.pivot_table(index='model', columns=['target_end_date', 'horizon'], 
                                      values='wis', aggfunc='mean').fillna(np.nan)
    
    if pivot_data.empty:
        print(f"Empty pivot data for {season} {location_filter}")
        return
    
    # Sort by mean score
    pivot_data = pivot_data.loc[pivot_data.mean(axis=1, skipna=True).sort_values().index]
    
    # Get missing info if function provided
    missing_info = {}
    if missing_info_fn and original_df is not None:
        missing_info = missing_info_fn(original_df, pivot_data.index.tolist(), location_filter)
    
    # Create labels
    labels = []
    colors = []
    for model in pivot_data.index:
        group = plot_data[plot_data['model'] == model]['group'].iloc[0]
        display_name = model
        missing_data = missing_info.get(model, {"text": "", "critical": False})
        missing_text = missing_data["text"] if isinstance(missing_data, dict) else missing_data
        is_critical = missing_data.get("critical", False) if isinstance(missing_data, dict) else bool(missing_data)
        
        if missing_text:
            label = f"{display_name}\n{missing_text}"
            if is_critical and group_colors:
                colors.append('red')
            elif group_colors:
                colors.append(group_colors.get(group, 'gray'))
            else:
                colors.append('red' if is_critical else 'gray')
        else:
            label = display_name
            if group_colors:
                colors.append(group_colors.get(group, 'gray'))
            else:
                colors.append('gray')
        labels.append(label)
    
    # Create plot
    fig_width = max(12, pivot_data.shape[1] * 0.4)
    fig_height = max(8, pivot_data.shape[0] * 0.4)
    plt.figure(figsize=(fig_width, fig_height))
    
    if center is not None:
        sns.heatmap(pivot_data, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                   yticklabels=labels, cbar_kws={'label': score_type})
    else:
        sns.heatmap(pivot_data, cmap=cmap, yticklabels=labels, cbar_kws={'label': score_type})
    
    # Color y-tick labels
    ax = plt.gca()
    for i, (label, color) in enumerate(zip(ax.get_yticklabels(), colors)):
        label.set_color(color)
    
    plt.title(title)
    plt.xlabel("Forecast Date & Horizon")
    plt.ylabel("Model")
    plt.tight_layout()
    
    fig = plt.gcf()
    return fig, ax

def plot_cumulative_timeseries(plot_data: pd.DataFrame, title: str, relative: bool):
    """Plot cumulative time series with running sum over time."""
    
    # Use appropriate column based on relative flag
    value_col = 'relative_wis' if relative else 'wis'
    
    # First sum values across all horizons for each model/date
    horizon_summed = plot_data.groupby(['model', 'group', 'target_end_date'], as_index=False)[value_col].sum()
    
    # Then create true cumulative sum over time for each model
    cumulative_data = []
    for model in horizon_summed['model'].unique():
        model_data = horizon_summed[horizon_summed['model'] == model].sort_values('target_end_date')
        model_data = model_data.copy()
        model_data[value_col] = model_data[value_col].cumsum()
        cumulative_data.append(model_data)
    
    cumulative_data = pd.concat(cumulative_data, ignore_index=True)
    
    if cumulative_data.empty:
        return
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot models with consistent styling
    influpaint_idx = 0
    flusight_idx = 0
    
    for model in cumulative_data['model'].unique():
        model_data = cumulative_data[cumulative_data['model'] == model].sort_values('target_end_date')
        group = model_data['group'].iloc[0]
        
        # Get styling for this model
        if group == 'influpaint':
            color_idx = influpaint_idx
            colors = ['#8B0000', '#006400', '#000080', '#8B008B', '#FF8C00', '#556B2F', '#8B4513', '#2F4F4F']
            marker = 's'
            linestyle = '-'
        else:
            color_idx = flusight_idx  
            colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD3BA', '#E0BBE4', '#C7CEEA', '#FFDFBA']
            marker = 'o'
            linestyle = '--'
        
        color = colors[color_idx % len(colors)]
        
        if group == 'influpaint':
            influpaint_idx += 1
        else:
            flusight_idx += 1
        
        # Plot the line
        ax.plot(model_data['target_end_date'], model_data[value_col], 
               color=color, marker=marker, markersize=4, 
               linewidth=1.5, linestyle=linestyle)
        
        # Add model name at the end of the line
        last_point = model_data.iloc[-1]
        ax.annotate(model, 
                   xy=(last_point['target_end_date'], last_point[value_col]),
                   xytext=(5, 0), textcoords='offset points',
                   fontsize=8, color=color, ha='left', va='center')
    
    if relative:
        # For relative WIS, baseline should be at y=1.0 * number of horizons (since we sum across horizons)
        baseline_y = len(plot_data['horizon'].unique())
        ax.axhline(y=baseline_y, color='red', linestyle=':', alpha=0.7, 
                  label=f'Baseline ({baseline_y} horizons)')
        ax.legend()
    
    apply_plot_styling(ax, title)
    ax.set_xlabel("Forecast Date")
    ax.set_ylabel("Relative WIS (Cumulative)" if relative else "WIS (Cumulative sum over time)")
    ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    fig = plt.gcf()
    return fig, ax

def plot_multi_location_stacked(df: pd.DataFrame, locations: List[str], reference_location: str,
                               value_cols: List[str], component_colors: Dict[str, str],
                               title: str, group_colors=None, location_name_mapper=None):
    """Plot stacked components for multiple locations with models on y-axis using seaborn."""
    
    # Filter to specified locations
    plot_data = df[df['location'].isin(locations)].copy()
    
    if plot_data.empty:
        return
    
    # Aggregate by model and location
    agg_data = plot_data.groupby(['model', 'group', 'location'], as_index=False).agg({
        col: 'sum' for col in value_cols
    })
    
    # Sort models by reference location total (sum of all components)
    ref_data = agg_data[agg_data['location'] == reference_location]
    if not ref_data.empty:
        ref_data['total'] = ref_data[value_cols].sum(axis=1)
        model_order = ref_data.sort_values('total')['model'].tolist()
    else:
        model_order = agg_data['model'].unique().tolist()
    
    # Reshape data for seaborn (melt components into long format)
    melted_data = pd.melt(agg_data, 
                         id_vars=['model', 'group', 'location'],
                         value_vars=value_cols,
                         var_name='component', 
                         value_name='value')
    
    # Add location display names using generic helper
    melted_data['location_display'] = melted_data['location'].apply(
        lambda x: get_location_display_name(x, reference_location, location_name_mapper)
    )
    
    # Create ordered categorical for consistent model ordering
    melted_data['model'] = pd.Categorical(melted_data['model'], categories=model_order, ordered=True)
    
    # Create FacetGrid with 4 rows, 13 columns
    g = sns.FacetGrid(melted_data, col='location_display', col_wrap=13, 
                      height=4.8, aspect=0.8, sharex=False, sharey=True)
    
    # Plot stacked bars for each facet
    def plot_stacked_bars(data, **kwargs):
        if data.empty:
            return
        
        # Pivot to get components as columns
        pivot_data = data.pivot_table(index='model', columns='component', values='value', fill_value=0)
        
        # Ensure all components are present
        for col in value_cols:
            if col not in pivot_data.columns:
                pivot_data[col] = 0
        
        # Reorder to match value_cols order
        pivot_data = pivot_data[value_cols]
        
        # Create stacked horizontal bars
        ax = plt.gca()
        y_pos = range(len(pivot_data))
        left_values = np.zeros(len(pivot_data))
        
        for component in value_cols:
            color = component_colors.get(component, 'gray')
            ax.barh(y_pos, pivot_data[component], left=left_values,
                   color=color, alpha=0.8)
            left_values += pivot_data[component]
        
        # Set y-axis
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pivot_data.index, fontsize=8)
        
        # Color y-tick labels by group if this is reference location
        current_location = data['location'].iloc[0] if not data.empty else None
        if current_location == reference_location and group_colors:
            tick_labels = ax.get_yticklabels()
            for i, model in enumerate(pivot_data.index):
                if i < len(tick_labels):  # Ensure index is valid
                    model_group = data[data['model'] == model]['group'].iloc[0] if model in data['model'].values else None
                    if model_group:
                        color = group_colors.get(model_group, 'black')
                        tick_labels[i].set_color(color)
        
        apply_plot_styling(ax, "")
        ax.set_xlabel('Components')
    
    g.map_dataframe(plot_stacked_bars)
    
    # Add legend (only to first subplot)
    if len(g.axes.flat) > 0:
        first_ax = g.axes.flat[0]
        legend_elements = [plt.Rectangle((0,0),1,1, color=component_colors.get(comp, 'gray'), 
                                       alpha=0.8, label=comp.title()) 
                          for comp in value_cols]
        first_ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8)
    
    g.fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return g.fig, g.axes


def print_ladderboard(metric: str, aggregation: str, filtered_df: pd.DataFrame, top_n: int = 10):
    """
    Print leaderboard for specified metric using filtered DataFrame.
    
    Args:
        metric: Column name to rank by (e.g., 'wis', 'relative_wis')
        aggregation: How to aggregate ('sum', 'mean', 'median')  
        filtered_df: Already filtered DataFrame (e.g., InfluPaint models only)
        top_n: Number of top models to show
    """
    if filtered_df.empty:
        print(f"❌ No data for {metric} leaderboard")
        return
    
    if metric not in filtered_df.columns:
        print(f"❌ Metric '{metric}' not found in DataFrame")
        return
    
    # Aggregate by model across all locations/dates
    if aggregation == 'sum':
        rankings = filtered_df.groupby('model')[metric].sum().sort_values()
    elif aggregation == 'mean':
        rankings = filtered_df.groupby('model')[metric].mean().sort_values()
    elif aggregation == 'median':
        rankings = filtered_df.groupby('model')[metric].median().sort_values()
    else:
        print(f"❌ Unknown aggregation method: {aggregation}")
        return
    
    # For relative metrics, sort in reverse (closer to 1.0 is better)
    if 'relative' in metric.lower():
        # Best relative WIS is closest to 1.0
        rankings = rankings.iloc[(rankings - 1.0).abs().argsort()]
    
    top_models = rankings.head(top_n)
    
    print(f"\n🏆 TOP {top_n} LEADERBOARD: {metric.upper()} ({aggregation.upper()})")
    print("=" * 60)
    
    for rank, (model, score) in enumerate(top_models.items(), 1):
        # Format score based on metric type
        if 'relative' in metric.lower():
            score_str = f"{score:.3f}"
        else:
            score_str = f"{score:.2f}"
        
        print(f"{rank:2d}. {score_str:>8s} - {model}")
    
    print("=" * 60)


#!/usr/bin/env python3
"""
InfluPaint vs FluSight evaluation plotting from scoringutils CSV.
Uses benchmark_plotting utilities for visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import the existing SeasonAxis and benchmark plotting
from influpaint.utils.season_axis import SeasonAxis
from benchmark_plotting import plot_components, plot_timeseries, plot_wis_heatmap, plot_cumulative_timeseries, plot_multi_location_stacked, print_ladderboard


# %% Configuration
CSV_PATH = "results/scoringutils_scores.csv" 
SAVE_DIR = "results/simple_plots"
GROUP_COLORS = {'influpaint': 'green', 'flusight': 'blue'}
ALLOW_MISSING_DATES_PER_MODEL = 5  # Same threshold as evaluation_pipeline.py

def add_inclusion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns indicating which plots each model should be included in based on per-season performance."""
    print(f"🔍 ANALYZING MODEL INCLUSION:")
    print(f"   Total models: {df['model'].nunique()}")
    
    # Determine date column for filtering
    date_col = 'forecast_date' if 'forecast_date' in df.columns else 'reference_date' if 'reference_date' in df.columns else 'target_end_date'
    print(f"   Using '{date_col}' for filtering")
    
    # Debug: Show data structure
    if 'season' in df.columns:
        unique_seasons = sorted(df['season'].unique()) 
        print(f"   🗓️ Seasons: {unique_seasons}")
        
        season_info = {}
        for season in unique_seasons:
            season_data = df[df['season'] == season]
            if date_col in season_data.columns:
                season_forecast_dates = season_data[date_col].nunique()
                season_info[season] = season_forecast_dates
                print(f"   🗓️ {season}: {season_forecast_dates} forecast dates")
        
        # Create inclusion columns
        df_with_flags = df.copy()
        
        # Per-season analysis
        for season in unique_seasons:
            max_dates_in_season = season_info[season]
            min_required_in_season = max_dates_in_season - ALLOW_MISSING_DATES_PER_MODEL
            
            season_data = df[df['season'] == season]
            model_counts_in_season = season_data.groupby('model')[date_col].nunique()
            
            # Models that meet criteria for this season
            valid_models_in_season = model_counts_in_season[model_counts_in_season >= min_required_in_season].index
            
            # Add inclusion column for this season
            include_col = f'include_{season.replace("-", "_")}'
            df_with_flags[include_col] = df_with_flags['model'].isin(valid_models_in_season)
            
            print(f"   🗓️ {season}: {len(valid_models_in_season)}/{df['model'].nunique()} models meet criteria (≥{min_required_in_season}/{max_dates_in_season} dates)")
        
        # Combined inclusion (must meet criteria in ALL seasons)
        if len(unique_seasons) > 1:
            include_cols = [f'include_{season.replace("-", "_")}' for season in unique_seasons]
            df_with_flags['include_combined'] = df_with_flags[include_cols].all(axis=1)
            combined_models = df_with_flags[df_with_flags['include_combined']]['model'].unique()
            print(f"   🔄 Combined: {len(combined_models)}/{df['model'].nunique()} models meet criteria in ALL seasons")
        else:
            df_with_flags['include_combined'] = df_with_flags[f'include_{unique_seasons[0].replace("-", "_")}']
            combined_models = df_with_flags[df_with_flags['include_combined']]['model'].unique()
        
        # Show model breakdown
        print(f"   📊 MODEL BREAKDOWN:")
        all_models = sorted(df['model'].unique())
        
        for model in all_models:
            model_info = []
            for season in unique_seasons:
                season_data = df[df['season'] == season]
                model_count = season_data[season_data['model'] == model][date_col].nunique() if model in season_data['model'].values else 0
                max_dates = season_info[season]
                missing = max_dates - model_count
                status = "✅" if missing <= ALLOW_MISSING_DATES_PER_MODEL else "❌"
                model_info.append(f"{season}:{status}{model_count:02d}/{max_dates:02d}")
            
            combined_status = "✅" if model in combined_models else "❌"
            print(f"{' | '.join(model_info)} | Combined:{combined_status} {model}")
        
        return df_with_flags
    
    else:
        # No seasons, just use overall filtering
        print("   No season column found, using overall filtering")
        model_date_counts = df.groupby('model')[date_col].nunique()
        max_dates = model_date_counts.max()
        min_required_dates = max_dates - ALLOW_MISSING_DATES_PER_MODEL
        
        successful_models = model_date_counts[model_date_counts >= min_required_dates].index
        df_with_flags = df.copy()
        df_with_flags['include_combined'] = df_with_flags['model'].isin(successful_models)
        
        return df_with_flags


def count_missing_data(original_df: pd.DataFrame, models_in_plot: List[str], location_filter: str) -> Dict[str, str]:
    """Count missing data for each model and return formatted strings."""
    missing_info = {}
    
    # Get expected dimensions from authoritative sources
    # Horizons: From configuration (should always be [0, 1, 2, 3])
    expected_targets = {0, 1, 2, 3}
    
    # Dates: From jobs file for InfluPaint models, or from available data for FluSight
    jobs_file = "paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt"
    try:
        import pandas as pd
        jobs_df = pd.read_csv(jobs_file)
        # Keep dates as strings to match scores file format
        # Get unique FORECAST dates from jobs file per season
        seasons_in_data = original_df['season'].unique()
        expected_forecast_dates = set()
        for season in seasons_in_data:
            season_jobs = jobs_df[jobs_df["season"] == season]
            expected_forecast_dates.update(season_jobs["date"].unique())
    except Exception as e:
        print(f"Warning: Could not read jobs file ({e}), using dates from data")
        expected_forecast_dates = set(original_df['reference_date'].unique())
    
    
    # Get expected locations
    if location_filter == "US":
        expected_locations = {"US"}
        tolerate_missing_locs = 0
    elif location_filter == "sum_all_states":
        season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
        expected_locations = set(season_axis.locations)
        tolerate_missing_locs = 3
    else:
        expected_locations = set(original_df['location'].unique())
        tolerate_missing_locs = 0
    
    for model in models_in_plot:
        model_data = original_df[original_df['model'] == model]
        
        if model_data.empty:
            # Model has no data at all
            total_expected = len(expected_forecast_dates) * len(expected_locations) * len(expected_targets)
            missing_info[model] = f"missing ({len(expected_locations)}l,{len(expected_targets)}h,{len(expected_forecast_dates)}d, Total: {total_expected})"
        else:
            # Filter to expected locations AND expected FORECAST dates only
            model_in_expected_locs = model_data[
                (model_data['location'].isin(expected_locations)) & 
                (model_data['reference_date'].isin(expected_forecast_dates))
            ]
            total_actual = len(model_in_expected_locs)
            
            # Calculate total expected (with tolerance for ALL_SUM)
            if location_filter == "sum_all_states":
                effective_expected_locs = max(0, len(expected_locations) - tolerate_missing_locs)
            else:
                effective_expected_locs = len(expected_locations)
            
            total_expected = len(expected_forecast_dates) * effective_expected_locs * len(expected_targets)
            total_missing = max(0, total_expected - total_actual)
            
            # Count missing by dimension (for the l,h,d display) - only within expected forecast dates
            actual_forecast_dates = set(model_in_expected_locs['reference_date'].unique())
            actual_locations = set(model_in_expected_locs['location'].unique())
            actual_targets = set(model_in_expected_locs['horizon'].unique())
            
            missing_dates = len(expected_forecast_dates - actual_forecast_dates)
            missing_targets = len(expected_targets - actual_targets) 
            missing_locations = max(0, len(expected_locations - actual_locations) - (tolerate_missing_locs if location_filter == "sum_all_states" else 0))
            
            # Determine if model has critical missing data (entire dimensions missing)
            has_critical_missing = (missing_dates > 0 or missing_targets > 0 or missing_locations > 0)
            
            if total_missing > 0:
                completion_rate = (total_actual / total_expected) * 100
                
                
                # Store both the display info and the critical flag
                # Compact format: "missing 0Lx2Hx0D / 47% (missing 68/128)"
                display_text = f"missing {missing_locations}Lx{missing_targets}Hx{missing_dates}D / {completion_rate:.0f}% (missing {total_missing}/{total_expected})"
                missing_info[model] = {"text": display_text, "critical": has_critical_missing}
            else:
                missing_info[model] = {"text": "", "critical": False}
    
    return missing_info













# %% Main Script

if __name__ == "__main__":
    # Load data
    df_raw = pd.read_csv(CSV_PATH)
    df_raw['target_end_date'] = pd.to_datetime(df_raw['target_end_date']).dt.date
    
    # Filter out problematic models from analysis
    # i808 models have issues, UGuelph-CompositeCurve makes plot scale badly
    df_raw = df_raw[~df_raw['model'].str.startswith('i808')]
    df_raw = df_raw[df_raw['model'] != 'UGuelph-CompositeCurve']
    
    # Calculate relative WIS against baseline for all data
    baseline_model = 'FluSight-baseline'
    if baseline_model in df_raw['model'].unique():
        baseline_data = df_raw[df_raw['model'] == baseline_model].set_index(['location', 'target_end_date', 'horizon'])['wis']
        df_raw['relative_wis'] = df_raw.apply(
            lambda row: row['wis'] / baseline_data.get((row['location'], row['target_end_date'], row['horizon']), np.nan) 
            if baseline_data.get((row['location'], row['target_end_date'], row['horizon']), 0) > 0 else np.nan, 
            axis=1
        )
    else:
        df_raw['relative_wis'] = np.nan
    
    # Add inclusion columns based on per-season performance
    df_with_flags = add_inclusion_columns(df_raw)
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Plot order: Combined, then individual seasons
    available_seasons = [s for s in df_with_flags['season'].unique() if s != "Combined"]
    seasons_to_plot = ["Combined"] + sorted(available_seasons)
    
    for season in seasons_to_plot:
        print(f"\n{'='*50}")
        print(f"PLOTTING: {season.upper()}")
        print('='*50)
        
        # Create season-specific folder
        season_dir = os.path.join(SAVE_DIR, season)
        os.makedirs(season_dir, exist_ok=True)
        
        # Get season data with appropriate filtering
        if season == "Combined":
            # For combined, only include models that meet criteria in ALL seasons
            season_df = df_with_flags[df_with_flags['include_combined']].copy()
            season_raw_df = df_with_flags.copy()  # Keep all for missing data counting
        else:
            # For individual seasons, include models that meet criteria in THIS season
            include_col = f'include_{season.replace("-", "_")}'
            season_data = df_with_flags[df_with_flags['season'] == season]
            season_df = season_data[season_data[include_col]].copy()
            season_raw_df = df_with_flags[df_with_flags['season'] == season].copy()
        
        if season_df.empty:
            print(f"No data for {season}")
            continue
        
        print(f"Models in {season}: {len(season_df['model'].unique())} (after per-season filtering)")
        
        # INFLUPAINT LEADERBOARDS
        influpaint_df = season_df[season_df['group'] == 'influpaint'].copy()
        if not influpaint_df.empty:
            # Total WIS across all locations
            print_ladderboard('wis', 'sum', influpaint_df, top_n=10)
            
            # Relative WIS (mean across all locations)
            print_ladderboard('relative_wis', 'mean', influpaint_df, top_n=10)
        
        # 1. WIS Heatmaps
        season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="US", 
            title=f"{season}: Absolute WIS Heatmap (US National)",
            relative=False, 
            original_df=season_raw_df, 
            missing_info_fn=count_missing_data, 
            group_colors=GROUP_COLORS
        )
        fig.savefig(os.path.join(season_dir, "absolute_wis_heatmap_US.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="US",
            title=f"{season}: Relative WIS Heatmap (US National)", 
            relative=True,
            original_df=season_raw_df,
            missing_info_fn=count_missing_data,
            group_colors=GROUP_COLORS
        )
        fig.savefig(os.path.join(season_dir, "relative_wis_heatmap_US.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="sum_all_states",
            title=f"{season}: Absolute WIS Heatmap (Sum Over Locations)",
            relative=False,
            original_df=season_raw_df,
            missing_info_fn=count_missing_data,
            group_colors=GROUP_COLORS,
            valid_locations=season_axis.locations
        )
        fig.savefig(os.path.join(season_dir, "absolute_wis_heatmap_sum_all_states.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plot_wis_heatmap(
            df=season_df,
            location_filter="sum_all_states",
            title=f"{season}: Relative WIS Heatmap (Sum Over Locations)",
            relative=True,
            original_df=season_raw_df,
            missing_info_fn=count_missing_data,
            group_colors=GROUP_COLORS,
            valid_locations=season_axis.locations
        )
        fig.savefig(os.path.join(season_dir, "relative_wis_heatmap_sum_all_states.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Full Performance Plot
        
        # US National full performance plot
        us_data = season_df[season_df['location'] == 'US'].copy()
        if not us_data.empty:
            
            us_missing_info = count_missing_data(season_raw_df, us_data['model'].unique().tolist(), "US")
            plot_components(
                df=us_data,
                group_by=['model', 'group'],
                value_cols=['wis', 'dispersion', 'overprediction', 'underprediction', 
                           'interval_coverage_50', 'interval_coverage_90', 'relative_wis'],
                agg_func={'wis': 'sum', 'dispersion': 'sum', 'overprediction': 'sum', 'underprediction': 'sum',
                         'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean', 'relative_wis': 'mean'},
                sort_by='wis',
                title=f"{season}: Full Performance (US National)",
                save_path=os.path.join(season_dir, "full_plot_US.png"),
                missing_info=us_missing_info,
                group_colors=GROUP_COLORS,
                reference_lines={
                    'interval_coverage_50': {'value': 0.5, 'label': 'Target 50%', 'color': 'red'},
                    'interval_coverage_90': {'value': 0.9, 'label': 'Target 90%', 'color': 'red'},
                    'relative_wis': {'value': 1.0, 'label': 'Baseline', 'color': 'black', 'linestyle': ':'}
                }
            )
        
        # Sum over locations full performance plot
        season_axis = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
        valid_locs = season_axis.locations
        allsum_data = season_df[season_df['location'].isin(valid_locs)].copy()
        if not allsum_data.empty:
            allsum_missing_info = count_missing_data(season_raw_df, allsum_data['model'].unique().tolist(), "sum_all_states")
            
            plot_components(
                df=allsum_data,
                group_by=['model', 'group'],
                value_cols=['wis', 'dispersion', 'overprediction', 'underprediction', 
                           'interval_coverage_50', 'interval_coverage_90', 'relative_wis'],
                agg_func={'wis': 'sum', 'dispersion': 'sum', 'overprediction': 'sum', 'underprediction': 'sum',
                         'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean', 'relative_wis': 'mean'},
                sort_by='wis',
                title=f"{season}: Full Performance (Sum Over Locations)",
                save_path=os.path.join(season_dir, "full_plot_sum_all_states.png"),
                missing_info=allsum_missing_info,
                group_colors=GROUP_COLORS,
                reference_lines={
                    'interval_coverage_50': {'value': 0.5, 'label': 'Target 50%', 'color': 'red'},
                    'interval_coverage_90': {'value': 0.9, 'label': 'Target 90%', 'color': 'red'},
                    'relative_wis': {'value': 1.0, 'label': 'Baseline', 'color': 'black', 'linestyle': ':'}
                }
            )
        
        # 5. Time Series
        # US National Time Series (Absolute WIS)
        us_ts_data = season_df[season_df['location'] == 'US'].copy()
        if not us_ts_data.empty:
            plot_timeseries(
                df=us_ts_data,
                x_col='target_end_date',
                y_col='wis',
                group_col='model',
                facet_col='horizon',
                filter_top_n=3,
                title=f"{season}: Absolute WIS Time Series (US National - Top 3 per Group)",
                save_path=os.path.join(season_dir, "absolute_timeseries_US.png"),
                relative=False
            )
            
            # Absolute WIS Cumulative Time Series
            # Filter to top 10 models per group for better readability
            if 'group' in us_ts_data.columns:
                from benchmark_plotting import get_top_models_per_group
                top_models = get_top_models_per_group(us_ts_data, 'wis', top_n=10, relative=False)
                us_ts_filtered = us_ts_data[us_ts_data['model'].isin(top_models)]
            else:
                # Fallback: top 10 overall
                model_avg = us_ts_data.groupby('model')['wis'].mean().nsmallest(10)
                us_ts_filtered = us_ts_data[us_ts_data['model'].isin(model_avg.index)]
            
            fig, ax = plot_cumulative_timeseries(
                plot_data=us_ts_filtered,
                title=f"{season}: Cumulative WIS (US National - Top 10 per Group)",
                relative=False
            )
            fig.savefig(os.path.join(season_dir, "cumulative_wis_US.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # Relative WIS Time Series
            plot_timeseries(
                df=us_ts_data,
                x_col='target_end_date',
                y_col='relative_wis',
                group_col='model',
                facet_col='horizon',
                filter_top_n=3,
                title=f"{season}: Relative WIS Time Series (US National - Top 3 per Group)",
                save_path=os.path.join(season_dir, "relative_timeseries_US.png"),
                relative=True
            )
            
            # Relative WIS Cumulative Time Series
            # Filter to top 10 models per group for better readability
            if 'group' in us_ts_data.columns:
                top_models_rel = get_top_models_per_group(us_ts_data, 'relative_wis', top_n=10, relative=True)
                us_ts_filtered_rel = us_ts_data[us_ts_data['model'].isin(top_models_rel)]
            else:
                # Fallback: top 10 overall (closest to 1.0 for relative)
                model_avg_rel = us_ts_data.groupby('model')['relative_wis'].mean()
                closest_to_one = model_avg_rel.iloc[(model_avg_rel - 1.0).abs().argsort()[:10]]
                us_ts_filtered_rel = us_ts_data[us_ts_data['model'].isin(closest_to_one.index)]
            
            fig, ax = plot_cumulative_timeseries(
                plot_data=us_ts_filtered_rel,
                title=f"{season}: Relative Cumulative WIS (US National - Top 10 per Group)",
                relative=True
            )
            fig.savefig(os.path.join(season_dir, "relative_cumulative_wis_US.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
        
        # 6. States Stacked WIS Components
        
        # Create states sum data
        states_data = season_df[season_df['location'].isin(season_axis.locations)]
        states_sum = states_data.groupby(['model', 'group'], as_index=False).agg({
            'underprediction': 'sum', 'overprediction': 'sum', 'dispersion': 'sum'
        })
        states_sum['location'] = 'States_Sum'
        
        # Combine with original data
        states_plot_data = pd.concat([season_df, states_sum], ignore_index=True)
        
        
        all_locations = ['US', 'States_Sum'] + season_axis.locations
        fig, axes = plot_multi_location_stacked(
            df=states_plot_data,
            locations=all_locations,
            reference_location='US',
            value_cols=['underprediction', 'overprediction', 'dispersion'],
            component_colors={'underprediction': 'red', 'overprediction': 'green', 'dispersion': 'blue'},
            title=f"{season}: WIS Components by State (Sorted by US Total WIS)",
            group_colors=GROUP_COLORS,
            location_name_mapper=season_axis.get_location_name
        )
        fig.savefig(os.path.join(season_dir, "wis_components_states_stacked.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        # 7. Scatter Plots using plot_components
        
        # WIS vs Relative WIS scatter - US National
        us_data = season_df[season_df['location'] == 'US']
        if not us_data.empty:
            plot_components(
                df=us_data,
                group_by=['model', 'group'],
                value_cols=['wis', 'relative_wis'], 
                agg_func={'wis': 'mean', 'relative_wis': 'mean'},
                title=f"{season}: WIS vs Relative WIS (US National)",
                save_path=os.path.join(season_dir, "wis_scatter_US.png"),
                group_colors=GROUP_COLORS,
                stacked=False
            )
        
        # WIS vs Relative WIS scatter - All Locations
        plot_components(
            df=season_df,
            group_by=['model', 'group'],
            value_cols=['wis', 'relative_wis'], 
            agg_func={'wis': 'sum', 'relative_wis': 'mean'},
            title=f"{season}: WIS vs Relative WIS (All Locations)",
            save_path=os.path.join(season_dir, "wis_scatter_all_locations.png"),
            group_colors=GROUP_COLORS,
            stacked=False
        )
        
        # Coverage scatter - US National
        us_data = season_df[season_df['location'] == 'US']
        if not us_data.empty:
            plot_components(
                df=us_data,
                group_by=['model', 'group'],
                value_cols=['interval_coverage_50', 'interval_coverage_90'],
                agg_func={'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean'},
                title=f"{season}: Coverage Comparison (US National)",
                save_path=os.path.join(season_dir, "coverage_scatter_US.png"),
                group_colors=GROUP_COLORS,
                stacked=False
            )
        
        # Coverage scatter - All Locations
        plot_components(
            df=season_df,
            group_by=['model', 'group'],
            value_cols=['interval_coverage_50', 'interval_coverage_90'],
            agg_func={'interval_coverage_50': 'mean', 'interval_coverage_90': 'mean'},
            title=f"{season}: Coverage Comparison (All Locations)",
            save_path=os.path.join(season_dir, "coverage_scatter_all_locations.png"),
            group_colors=GROUP_COLORS,
            stacked=False
        )
        
        print(f"Completed plots for {season}")
    
    print(f"\n{'='*50}")
    print(f"ALL PLOTS SAVED TO: {SAVE_DIR}")
    print('='*50)
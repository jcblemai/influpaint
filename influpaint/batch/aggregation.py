#!/usr/bin/env python3
"""
Post-processing script to aggregate inpainting results and compute WIS scores.

Reads results from MLflow experiment and computes:
- WIS scores across scenarios/dates/configs
- Summary tables and plots
- Ensemble forecasts

Usage:
    python aggregate_results.py -e "paper-2025-06_inpainting" --output_dir "./results"
"""

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from tabulate import tabulate
import json


@click.command()
@click.option("-e", "--experiment_name", required=True, help="MLflow inpainting experiment name")
@click.option("--output_dir", default="./results", help="Output directory for aggregated results")
@click.option("--compute_wis", is_flag=True, help="Compute WIS scores (requires ground truth)")
@click.option("--create_ensemble", is_flag=True, help="Create ensemble forecasts across configs")
@click.option("--plot_results", is_flag=True, help="Generate summary plots")
def main(experiment_name, output_dir, compute_wis, create_ensemble, plot_results):
    """Aggregate inpainting results from MLflow experiment"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Aggregating results from experiment: {experiment_name}")
    print(f"📁 Output directory: {output_path}")
    
    # Load results from MLflow
    print("\n📊 Loading results from MLflow...")
    results_df = load_mlflow_results(experiment_name)
    
    if results_df.empty:
        print("❌ No results found in MLflow experiment")
        return
    
    print(f"✅ Loaded {len(results_df)} inpainting runs")
    
    # Save raw results
    results_file = output_path / "raw_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"💾 Raw results saved to: {results_file}")
    
    # Generate summary statistics
    print("\n📈 Generating summary statistics...")
    summary_stats = generate_summary_stats(results_df)
    
    # Save summary
    summary_file = output_path / "summary_stats.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"💾 Summary stats saved to: {summary_file}")
    
    # Print summary table
    print("\n📋 Summary by Scenario and Config:")
    print_summary_table(results_df)
    
    # Compute WIS if requested
    if compute_wis:
        print("\n🎯 Computing WIS scores...")
        try:
            wis_results = compute_wis_scores(results_df)
            wis_file = output_path / "wis_scores.csv"
            wis_results.to_csv(wis_file, index=False)
            print(f"💾 WIS scores saved to: {wis_file}")
            
            print("\n🏆 Best performing configs by WIS:")
            print_wis_rankings(wis_results)
            
        except Exception as e:
            print(f"⚠️  WIS computation failed: {e}")
    
    # Create ensemble forecasts
    if create_ensemble:
        print("\n🤝 Creating ensemble forecasts...")
        try:
            ensemble_results = create_ensemble_forecasts(results_df)
            ensemble_file = output_path / "ensemble_forecasts.csv"
            ensemble_results.to_csv(ensemble_file, index=False)
            print(f"💾 Ensemble forecasts saved to: {ensemble_file}")
        except Exception as e:
            print(f"⚠️  Ensemble creation failed: {e}")
    
    # Generate plots
    if plot_results:
        print("\n📊 Generating plots...")
        try:
            create_summary_plots(results_df, output_path)
            print(f"💾 Plots saved to: {output_path}")
        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")
    
    print(f"\n✅ Aggregation complete! Results in: {output_path}")


def load_mlflow_results(experiment_name):
    """Load inpainting results from MLflow experiment"""
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experiment '{experiment_name}' not found")
            return pd.DataFrame()
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"]
        )
        
        if not runs:
            print(f"❌ No finished runs found in experiment")
            return pd.DataFrame()
        
        # Extract data from runs
        data = []
        for run in runs:
            params = run.data.params
            metrics = run.data.metrics
            
            data.append({
                'run_id': run.info.run_id,
                'scenario_id': int(params.get('scenario_id', -1)),
                'forecast_date': params.get('forecast_date'),
                'config_name': params.get('config_name'),
                'unet_name': params.get('unet_name'),
                'dataset_name': params.get('dataset_name'),
                'transform_name': params.get('transform_name'),
                'enrich_name': params.get('enrich_name'),
                'training_run_id': params.get('training_run_id'),
                'forecast_mean': metrics.get('forecast_mean'),
                'forecast_std': metrics.get('forecast_std'),
                'forecast_min': metrics.get('forecast_min'),
                'forecast_max': metrics.get('forecast_max'),
                'num_samples': metrics.get('num_samples'),
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'status': run.info.status
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"❌ Error loading MLflow results: {e}")
        return pd.DataFrame()


def generate_summary_stats(df):
    """Generate summary statistics from results"""
    stats = {
        'total_runs': len(df),
        'unique_scenarios': df['scenario_id'].nunique(),
        'unique_dates': df['forecast_date'].nunique(),
        'unique_configs': df['config_name'].nunique(),
        'scenario_range': f"{df['scenario_id'].min()}-{df['scenario_id'].max()}",
        'date_range': f"{df['forecast_date'].min()} to {df['forecast_date'].max()}",
        'configs': sorted(df['config_name'].unique().tolist()),
        'completion_rate': f"{len(df[df['status'] == 'FINISHED'])} / {len(df)} ({100*len(df[df['status'] == 'FINISHED'])/len(df):.1f}%)",
        'avg_forecast_mean': df['forecast_mean'].mean(),
        'avg_forecast_std': df['forecast_std'].mean(),
    }
    return stats


def print_summary_table(df):
    """Print summary table of results"""
    summary = df.groupby(['scenario_id', 'config_name']).agg({
        'forecast_mean': ['count', 'mean', 'std'],
        'forecast_std': 'mean',
        'forecast_date': 'nunique'
    }).round(2)
    
    summary.columns = ['num_dates', 'mean_forecast', 'std_forecast', 'mean_uncertainty', 'unique_dates']
    summary = summary.reset_index()
    
    print(tabulate(summary.head(20), headers='keys', tablefmt='simple', showindex=False))
    if len(summary) > 20:
        print(f"... and {len(summary) - 20} more rows")


def compute_wis_scores(df):
    """Compute WIS scores (placeholder - needs ground truth data)"""
    # This is a placeholder - you'd need to implement actual WIS computation
    # based on your ground truth data and forecast quantiles
    
    print("⚠️  WIS computation not implemented yet - need ground truth data")
    print("   This would require:")
    print("   1. Loading ground truth observations for each date")
    print("   2. Loading forecast quantiles from saved files")  
    print("   3. Computing WIS using proper scoring rules")
    
    # Return placeholder results
    wis_data = []
    for scenario in df['scenario_id'].unique():
        for config in df['config_name'].unique():
            wis_data.append({
                'scenario_id': scenario,
                'config_name': config,
                'wis_score': np.random.random(),  # Placeholder
                'num_dates': len(df[(df['scenario_id'] == scenario) & (df['config_name'] == config)])
            })
    
    return pd.DataFrame(wis_data)


def print_wis_rankings(wis_df):
    """Print WIS rankings by config"""
    avg_wis = wis_df.groupby('config_name')['wis_score'].mean().sort_values()
    
    print("Average WIS by config (lower is better):")
    for i, (config, wis) in enumerate(avg_wis.items(), 1):
        print(f"  {i}. {config}: {wis:.3f}")


def create_ensemble_forecasts(df):
    """Create ensemble forecasts by averaging across configs"""
    ensemble = df.groupby(['scenario_id', 'forecast_date']).agg({
        'forecast_mean': 'mean',
        'forecast_std': 'mean',  # Simple average - could use more sophisticated methods
        'num_samples': 'first'
    }).reset_index()
    
    ensemble['config_name'] = 'ensemble'
    return ensemble


def create_summary_plots(df, output_path):
    """Create summary plots"""
    
    # Plot 1: Forecast means by scenario and config
    plt.figure(figsize=(12, 8))
    
    # Pivot for heatmap
    pivot_data = df.pivot_table(
        values='forecast_mean', 
        index='scenario_id', 
        columns='config_name',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis')
    plt.title('Average Forecast Mean by Scenario and Config')
    plt.tight_layout()
    plt.savefig(output_path / 'forecast_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Forecast distribution by config
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='config_name', y='forecast_mean')
    plt.xticks(rotation=45)
    plt.title('Forecast Distribution by Config')
    plt.tight_layout()
    plt.savefig(output_path / 'forecast_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Timeline of forecasts
    if 'forecast_date' in df.columns:
        df_timeline = df.copy()
        df_timeline['forecast_date'] = pd.to_datetime(df_timeline['forecast_date'])
        
        plt.figure(figsize=(14, 8))
        for config in df['config_name'].unique():
            config_data = df_timeline[df_timeline['config_name'] == config]
            daily_avg = config_data.groupby('forecast_date')['forecast_mean'].mean()
            plt.plot(daily_avg.index, daily_avg.values, marker='o', label=config, alpha=0.7)
        
        plt.xlabel('Forecast Date')
        plt.ylabel('Average Forecast Mean')
        plt.title('Forecast Timeline by Config')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'forecast_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Extract full loss time series per model from MLflow experiment.

Usage:
    python extract_mlflow_losses.py --experiment_name "paper-2025-07-22_training"
"""

import click
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient


@click.command()
@click.option("--experiment_name", required=True, help="MLflow experiment name")
@click.option("--output_file", default="mlflow_losses.csv", help="Output CSV file for summary")
@click.option("--output_timeseries", default="mlflow_loss_timeseries.csv", help="Output CSV file for full time series")
def main(experiment_name, output_file, output_timeseries):
    """Extract loss time series from MLflow experiment"""
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Get experiment by name
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return
    except Exception as e:
        print(f"Error finding experiment: {e}")
        return
    
    print(f"Found experiment: {experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    
    # Get all runs from the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["start_time DESC"]
    )
    
    print(f"Found {len(runs)} runs in experiment")
    
    # Extract summary data and full time series
    summary_data = []
    timeseries_data = []
    
    for run in runs:
        run_id = run.info.run_id
        scenario_id = run.data.params.get('scenario_id')
        scenario_string = run.data.params.get('scenario_string')
        
        # Summary data
        run_summary = {
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', ''),
            'scenario_id': scenario_id,
            'scenario_string': scenario_string,
            'final_loss': run.data.metrics.get('final_loss'),
            'avg_loss_last_100': run.data.metrics.get('avg_loss_last_100'),
            'ddpm_name': run.data.params.get('ddpm_name'),
            'dataset_name': run.data.params.get('dataset_name'),
            'transform_name': run.data.params.get('transform_name'),
            'enrich_name': run.data.params.get('enrich_name'),
            'epochs': run.data.params.get('epochs'),
            'batch_size': run.data.params.get('batch_size'),
            'dataset_size': run.data.params.get('dataset_size'),
            'training_completed': run.data.metrics.get('training_completed'),
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'status': run.info.status
        }
        summary_data.append(run_summary)
        
        # Get full loss time series for this run
        try:
            # Try to get step-by-step loss metrics
            loss_history = client.get_metric_history(run_id, "loss")
            if loss_history:
                print(f"Found {len(loss_history)} loss steps for run {scenario_id}")
                for metric in loss_history:
                    timeseries_data.append({
                        'run_id': run_id,
                        'scenario_id': scenario_id,
                        'scenario_string': scenario_string,
                        'step': metric.step,
                        'timestamp': metric.timestamp,
                        'loss': metric.value
                    })
            else:
                # Try alternative metric names
                for metric_name in ["train_loss", "training_loss", "epoch_loss"]:
                    loss_history = client.get_metric_history(run_id, metric_name)
                    if loss_history:
                        print(f"Found {len(loss_history)} {metric_name} steps for run {scenario_id}")
                        for metric in loss_history:
                            timeseries_data.append({
                                'run_id': run_id,
                                'scenario_id': scenario_id,
                                'scenario_string': scenario_string,
                                'step': metric.step,
                                'timestamp': metric.timestamp,
                                'loss': metric.value
                            })
                        break
                else:
                    print(f"No loss time series found for run {scenario_id}")
                    
        except Exception as e:
            print(f"Error getting loss history for run {scenario_id}: {e}")
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_data)
    timeseries_df = pd.DataFrame(timeseries_data)
    
    # Filter completed runs
    completed_summary = summary_df[
        (summary_df['training_completed'] == 1.0) & 
        (summary_df['final_loss'].notna())
    ].copy()
    
    print(f"\nCompleted runs with final_loss: {len(completed_summary)}")
    
    if len(completed_summary) > 0:
        # Sort by scenario_id
        completed_summary = completed_summary.sort_values('scenario_id')
        completed_summary.to_csv(output_file, index=False)
        print(f"Summary saved to: {output_file}")
        
        # Display summary
        print(f"\nSummary:")
        print(f"Total completed runs: {len(completed_summary)}")
        print(f"Unique scenarios: {completed_summary['scenario_string'].nunique()}")
        print(f"Final loss range: {completed_summary['final_loss'].min():.6f} - {completed_summary['final_loss'].max():.6f}")
        print(f"Mean final loss: {completed_summary['final_loss'].mean():.6f}")
        
        # Show first few rows
        print("\nFirst few rows:")
        display_cols = ['scenario_id', 'scenario_string', 'final_loss', 'avg_loss_last_100']
        print(completed_summary[display_cols].head(10).to_string(index=False))
    
    # Save time series data
    if len(timeseries_df) > 0:
        timeseries_df = timeseries_df.sort_values(['scenario_id', 'step'])
        timeseries_df.to_csv(output_timeseries, index=False)
        print(f"\nTime series data saved to: {output_timeseries}")
        print(f"Total time series points: {len(timeseries_df)}")
        print(f"Runs with time series: {timeseries_df['run_id'].nunique()}")
        
        # Show time series summary
        if len(timeseries_df) > 0:
            print(f"Steps per run range: {timeseries_df.groupby('run_id')['step'].count().min()} - {timeseries_df.groupby('run_id')['step'].count().max()}")
            print("\nTime series sample:")
            print(timeseries_df[['scenario_id', 'step', 'loss']].head(10).to_string(index=False))
    else:
        print("\nNo time series data found. Loss may be logged under different metric names or not logged step-by-step.")
    
    return completed_summary, timeseries_df


if __name__ == '__main__':
    main()
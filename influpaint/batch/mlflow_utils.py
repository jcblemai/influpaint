#!/usr/bin/env python3
"""
Helper script to get MLflow run ID for a specific scenario from training experiment.

Usage:
    python get_mlflow_run_id.py -e "experiment_name_training" -s 5
"""

import click
import mlflow
from mlflow.tracking import MlflowClient


@click.command()
@click.option("-e", "--experiment_name", required=True, help="MLflow experiment name (training)")
@click.option("-s", "--scenario_id", required=True, type=int, help="Scenario ID")
def main(experiment_name, scenario_id):
    """Get MLflow run ID for a trained scenario"""
    
    client = MlflowClient()
    
    try:
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"ERROR: Experiment '{experiment_name}' not found")
            exit(1)
        
        # Search for runs with this scenario
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.scenario_id = '{scenario_id}'",
            order_by=["start_time DESC"]
        )
        
        if not runs:
            print(f"ERROR: No runs found for scenario {scenario_id} in experiment '{experiment_name}'")
            exit(1)
        
        # Get most recent successful run
        successful_runs = [r for r in runs if r.info.status == "FINISHED"]
        if not successful_runs:
            print(f"ERROR: No successful runs found for scenario {scenario_id}")
            print(f"Available runs: {[r.info.status for r in runs]}")
            exit(1)
            
        # Print just the run ID (for use in shell scripts)
        print(successful_runs[0].info.run_id)
        
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)


if __name__ == '__main__':
    main()
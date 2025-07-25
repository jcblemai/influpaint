#!/usr/bin/env python3
"""
Generate SLURM array jobs for parallel inpainting across dates and configs.

This creates a job array where each job runs one atomic inpainting task:
- Single scenario (from training experiment)
- Single forecast date
- Single config

Usage:
    python generate_inpaint_jobs.py -e "paper-2025-06" -s 5 --start_date "2022-10-12" --end_date "2023-05-15"
"""

import click
import pandas as pd
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def get_finished_models(experiment_name):
    """Get all finished training runs from MLflow experiment"""
    client = MlflowClient()
    
    try:
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"ERROR: Experiment '{experiment_name}' not found")
            return []
        
        # Search for all finished runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"]
        )
        
        finished_models = []
        for run in runs:
            params = run.data.params
            finished_models.append({
                'run_id': run.info.run_id,
                'scenario_id': int(params['scenario_id']),
                'scenario_string': params.get('scenario_string', 'unknown'),
                'start_time': run.info.start_time
            })
        
        # Sort by scenario_id for consistent ordering
        finished_models.sort(key=lambda x: x['scenario_id'])
        
        return finished_models
        
    except Exception as e:
        print(f"ERROR querying MLflow: {e}")
        return []


def filter_completed_jobs(jobs, training_experiment_name):
    """Filter out jobs that already completed successfully in the inpainting experiment"""
    client = MlflowClient()
    
    # Get inpainting experiment name
    inpainting_experiment_name = training_experiment_name.replace('_training', '_inpainting')
    
    # Get inpainting experiment
    experiment = client.get_experiment_by_name(inpainting_experiment_name)
    if experiment is None:
        print(f"No inpainting experiment '{inpainting_experiment_name}' found - keeping all jobs")
        return jobs
    
    # Get all successful inpainting runs
    completed_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"]
    )
    
    # Build set of completed (scenario_id, forecast_date, config_name) combinations
    completed_jobs = set()
    for run in completed_runs:
        params = run.data.params
        if all(key in params for key in ['scenario_id', 'forecast_date', 'config_name']):
            completed_jobs.add((
                int(params['scenario_id']),
                params['forecast_date'], 
                params['config_name']
            ))
    
    print(f"Found {len(completed_jobs)} completed inpainting jobs in '{inpainting_experiment_name}'")
    
    # Filter out completed jobs
    remaining_jobs = []
    for job in jobs:
        job_key = (job['scenario_id'], job['date'], job['config'])
        if job_key not in completed_jobs:
            remaining_jobs.append(job)
    
    return remaining_jobs


@click.command()
@click.option("-e", "--experiment_name", required=True, help="MLflow training experiment name (e.g. 'paper-2025-06_training')")
@click.option("--configs", default=None, help="Comma-separated config names (default: all available configs)")
@click.option("--output_dir", default=".", help="Where to write job files")
@click.option("--skip_completed", is_flag=True, help="Skip jobs that already completed successfully in inpainting experiment")
def main(experiment_name, configs, output_dir, skip_completed):
    """Generate SLURM job files for inpainting all finished models from MLflow experiment on last 2 flu seasons"""
    
    # Get finished models from MLflow
    print(f"Querying MLflow experiment: {experiment_name}")
    finished_models = get_finished_models(experiment_name)
    
    if not finished_models:
        print(f"No finished models found in experiment '{experiment_name}'")
        return
    
    print(f"Found {len(finished_models)} finished models")
    
    # Define 3 flu seasons for scoring
    flu_seasons = [
        # {
        #     'name': '2022-2023',
        #     'start': '2022-10-17', 
        #     'end': '2023-05-15',
        #     'freq': 'W-SAT'
        # },
        {
            'name': '2023-2024', 
            'start': '2023-10-14',
            'end': '2024-05-04', 
            'freq': 'W-SAT'
        },
        {
            'name': '2024-2025', 
            'start': '2024-11-23',
            'end': '2025-05-31', 
            'freq': 'W-SAT'
        }
    ]
    
    # Use all available configs if none specified
    if configs is None:
        from .config import AVAILABLE_COPAINT_CONFIGS
        configs = ",".join(AVAILABLE_COPAINT_CONFIGS)
        print(f"Using all available configs: {configs}")
    
    config_names = [c.strip() for c in configs.split(",")]
    
    print(f"Flu seasons: {[s['name'] for s in flu_seasons]}")
    print(f"Configs: {config_names}")
    
    # Generate all combinations: model × season × date × config
    jobs = []
    job_id = 0
    
    for model in finished_models:
        scenario_id = model['scenario_id']
        run_id = model['run_id']
        
        for season in flu_seasons:
            forecast_dates = pd.date_range(season['start'], season['end'], freq=season['freq'])
            
            for date in forecast_dates:
                for config_name in config_names:
                    jobs.append({
                        'job_id': job_id,
                        'scenario_id': scenario_id,
                        'run_id': run_id,
                        'season': season['name'],
                        'date': date.strftime('%Y-%m-%d'),
                        'config': config_name
                    })
                    job_id += 1
    
    print(f"Total jobs: {len(jobs)} = {len(finished_models)} models × {len(flu_seasons)} seasons × ~{len(forecast_dates)} dates × {len(config_names)} configs")
    
    # Filter out completed jobs if requested
    if skip_completed:
        jobs = filter_completed_jobs(jobs, experiment_name)
        print(f"After filtering completed jobs: {len(jobs)} remaining")
        
        if not jobs:
            print("No jobs remaining after filtering - all appear to be completed!")
            return
    
    # Write job list with run_id included
    output_path = Path(output_dir)
    experiment_basename = experiment_name.replace('_training', '')
    job_list_file = output_path / f"inpaint_jobs_{experiment_basename}.txt"
    
    with open(job_list_file, 'w') as f:
        f.write("job_id,scenario_id,run_id,season,date,config\n")
        for job in jobs:
            f.write(f"{job['job_id']},{job['scenario_id']},{job['run_id']},{job['season']},{job['date']},{job['config']}\n")
    
    print(f"Job list written to: {job_list_file}")
    
    # Generate SLURM script
    inpaint_exp = f"{experiment_basename}_inpainting"
    
    slurm_script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --qos gpu_access
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --mem=32G
#SBATCH -t 00-04:00:00
#SBATCH --array=0-{len(jobs)-1}
#SBATCH --gres=gpu:1

module purge

# Experiment configuration
TRAINING_EXP="{experiment_name}"
INPAINT_EXP="{inpaint_exp}"
JOB_LIST="{job_list_file}"

echo "Inpainting job ${{SLURM_ARRAY_TASK_ID}}"

# Get job parameters from job list
JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $JOB_LIST)  # +2 to skip header
SCENARIO_ID=$(echo $JOB_LINE | cut -d',' -f2)
RUN_ID=$(echo $JOB_LINE | cut -d',' -f3)
SEASON=$(echo $JOB_LINE | cut -d',' -f4)
DATE=$(echo $JOB_LINE | cut -d',' -f5)
CONFIG=$(echo $JOB_LINE | cut -d',' -f6)

echo "Scenario: ${{SCENARIO_ID}}, Run: ${{RUN_ID}}, Season: ${{SEASON}}, Date: ${{DATE}}, Config: ${{CONFIG}}"

# Run atomic inpainting with known run_id
/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python -u -m influpaint.batch.inpainting \\
    -s ${{SCENARIO_ID}} \\
    -r "${{RUN_ID}}" \\
    -e "${{INPAINT_EXP}}" \\
    --forecast_date "${{DATE}}" \\
    --config_name "${{CONFIG}}" \\
    > out_inpaint_s${{SCENARIO_ID}}_${{SEASON}}_${{DATE}}_${{CONFIG}}.out 2>&1

echo "Completed: Scenario ${{SCENARIO_ID}}, Season ${{SEASON}}, Date ${{DATE}}, Config ${{CONFIG}}"
"""
    
    slurm_file = output_path / f"inpaint_array_{experiment_basename}.run"
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)
    
    # Make executable
    slurm_file.chmod(0o755)
    
    print(f"SLURM script written to: {slurm_file}")
    print()
    print("To submit:")
    print(f"  sbatch {slurm_file}")
    print()
    print("To monitor:")
    print(f"  squeue -u $USER")
    print(f"  ls out_inpaint_s*_*_*_*.out")
    print()
    print(f"Expected output files: out_inpaint_s<SCENARIO>_<SEASON>_<DATE>_<CONFIG>.out")
    print(f"Example: out_inpaint_s5_2022-2023_2022-11-14_celebahq_try1.out")
    print()
    print("Model summary:")
    for model in finished_models[:5]:  # Show first 5
        print(f"  Scenario {model['scenario_id']}: {model['scenario_string']} (run: {model['run_id'][:8]}...)")
    if len(finished_models) > 5:
        print(f"  ... and {len(finished_models) - 5} more models")


if __name__ == '__main__':
    main()
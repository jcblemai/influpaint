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
from .scenarios import ScenarioLibrary


@click.command()
@click.option("-e", "--experiment_name", required=True, help="Base experiment name (e.g. 'paper-2025-06')")
@click.option("--scenarios", default="0-31", help="Scenario range (e.g. '0-31' or '5,8,12')")
@click.option("--start_date", default="2022-10-12", help="Start date (YYYY-MM-DD)")
@click.option("--end_date", default="2023-05-15", help="End date (YYYY-MM-DD)")
@click.option("--freq", default="2W-MON", help="Date frequency")
@click.option("--configs", default="celebahq_try1,celebahq_try3,celebahq", help="Comma-separated config names")
@click.option("--output_dir", default=".", help="Where to write job files")
def main(experiment_name, scenarios, start_date, end_date, freq, configs, output_dir):
    """Generate SLURM job files for parallel inpainting across all scenarios"""
    
    # Parse scenarios
    if '-' in scenarios:
        start_scn, end_scn = map(int, scenarios.split('-'))
        scenario_ids = list(range(start_scn, end_scn + 1))
    else:
        scenario_ids = [int(s.strip()) for s in scenarios.split(',')]
    
    # Parse inputs
    forecast_dates = pd.date_range(start_date, end_date, freq=freq)
    config_names = [c.strip() for c in configs.split(",")]
    
    print(f"Experiment: {experiment_name}")
    print(f"Scenarios: {len(scenario_ids)} ({min(scenario_ids)}-{max(scenario_ids)})")
    print(f"Dates: {len(forecast_dates)} ({start_date} to {end_date})")
    print(f"Configs: {len(config_names)} ({config_names})")
    
    # Generate all combinations: scenario × date × config
    jobs = []
    job_id = 0
    
    for scenario_id in scenario_ids:
        for date in forecast_dates:
            for config_name in config_names:
                jobs.append({
                    'job_id': job_id,
                    'scenario_id': scenario_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'config': config_name
                })
                job_id += 1
    
    print(f"Total jobs: {len(jobs)} = {len(scenario_ids)} scenarios × {len(forecast_dates)} dates × {len(config_names)} configs")
    
    # Write job list
    output_path = Path(output_dir)
    job_list_file = output_path / f"inpaint_jobs_{experiment_name}_all_scenarios.txt"
    
    with open(job_list_file, 'w') as f:
        f.write("job_id,scenario_id,date,config\n")
        for job in jobs:
            f.write(f"{job['job_id']},{job['scenario_id']},{job['date']},{job['config']}\n")
    
    print(f"Job list written to: {job_list_file}")
    
    # Generate SLURM script
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
EXPERIMENT_NAME="{experiment_name}"
TRAINING_EXP="${{EXPERIMENT_NAME}}_training"
INPAINT_EXP="${{EXPERIMENT_NAME}}_inpainting"
JOB_LIST="{job_list_file}"

echo "Inpainting job ${{SLURM_ARRAY_TASK_ID}}"

# Get job parameters from job list
JOB_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $JOB_LIST)  # +2 to skip header
SCENARIO_ID=$(echo $JOB_LINE | cut -d',' -f2)
DATE=$(echo $JOB_LINE | cut -d',' -f3)
CONFIG=$(echo $JOB_LINE | cut -d',' -f4)

echo "Scenario: ${{SCENARIO_ID}}, Date: ${{DATE}}, Config: ${{CONFIG}}"

# Get the MLflow run ID for this scenario
RUN_ID=$(/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python influpaint/batch/mlflow_utils.py \\
    -e "${{TRAINING_EXP}}" \\
    -s ${{SCENARIO_ID}})

if [ $? -ne 0 ]; then
    echo "ERROR: No trained model found for scenario ${{SCENARIO_ID}} in ${{TRAINING_EXP}}"
    exit 1
fi

echo "Found trained model: ${{RUN_ID}}"

# Run atomic inpainting
/nas/longleaf/home/chadi/.conda/envs/diffusion_torch6/bin/python -u influpaint/batch/inpainting.py \\
    -s ${{SCENARIO_ID}} \\
    -r "${{RUN_ID}}" \\
    -e "${{INPAINT_EXP}}" \\
    --forecast_date "${{DATE}}" \\
    --config_name "${{CONFIG}}" \\
    > out_inpaint_s${{SCENARIO_ID}}_${{DATE}}_${{CONFIG}}.out 2>&1

echo "Completed: Scenario ${{SCENARIO_ID}}, Date ${{DATE}}, Config ${{CONFIG}}"
"""
    
    slurm_file = output_path / f"inpaint_array_{experiment_name}_all_scenarios.run"
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
    print(f"  ls out_inpaint_s*_*.out")
    print()
    print(f"Expected output files: out_inpaint_s<SCENARIO>_<DATE>_<CONFIG>.out")
    print(f"Example: out_inpaint_s5_2022-11-14_celebahq_try1.out")


if __name__ == '__main__':
    main()
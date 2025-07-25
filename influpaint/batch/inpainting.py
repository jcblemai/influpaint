#!/usr/bin/env python3
"""
Atomic inpainting script for generating forecasts using trained diffusion models.
Runs a single scenario, for a single date, with a single config (fully parallelizable).

Usage:
    # Single atomic run
    python inpaint.py -s 5 -r "abc123def456" -e "exp" --forecast_date "2022-11-14" --config_name "celebahq_try1"
    
    # With filesystem fallback
    python inpaint.py -s 5 -m "/path/to/model.pth" -e "exp" --forecast_date "2022-11-14" --config_name "celebahq_try1"
"""

import datetime
import sys
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import mlflow

from .scenarios import get_training_scenario
from .config import copaint_config_library, create_folders, get_git_revision_short_hash
from ..utils import ground_truth

sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler


from influpaint.utils import SeasonAxis

season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)


@click.command()
@click.option("-s", "--scn_id", "scn_id", required=True, type=int, help="ID of the scenario to run inpainting for")
@click.option("-r", "--run_id", "run_id", type=str, help="MLflow run ID of the training run to use")
@click.option("-m", "--model_path", "model_path", type=str, help="Path to model checkpoint (.pth file) - alternative to --run_id")
@click.option("-e", "--experiment_name", "experiment_name", envvar="experiment_name", type=str, required=True,
              help="MLflow experiment name (e.g. 'paper-2025-06_inpainting')")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, 
              default='/users/c/h/chadi/influpaint_res/', show_default=True,
              help="Where to write forecast results")
@click.option("--forecast_date", required=True, type=str, help="Single forecast date (YYYY-MM-DD)")
@click.option("--config_name", required=True, type=str, help="CoPaint config name (e.g. 'celebahq_try1')")
@click.option("--image_size", default=64, type=int, help="Image size")
@click.option("--channels", default=1, type=int, help="Number of channels")
@click.option("--batch_size", default=512, type=int, help="Batch size for inference")
def main(scn_id, run_id, model_path, experiment_name, outdir, forecast_date, config_name, image_size, channels, batch_size):
    """Run inpainting/forecasting for a specific scenario using a trained model"""
    
    # Validate inputs
    if not run_id and not model_path:
        raise click.ClickException("Must provide either --run_id (MLflow) or --model_path (filesystem)")
    if run_id and model_path:
        raise click.ClickException("Cannot provide both --run_id and --model_path. Choose one.")
    
    # Get scenario specification
    scenario_spec = get_training_scenario(scn_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running inpainting for scenario {scn_id}: {scenario_spec.scenario_string}")
    print(f"Device: {device}")
    print(f"Forecast date: {forecast_date}")
    print(f"Config: {config_name}")
    
    # Experiment setup  
    # Create output directory structure with experiment name
    output_folder = f"{outdir}{get_git_revision_short_hash()}_{experiment_name}_{datetime.date.today()}"
    create_folders(output_folder)
    mlflow.set_experiment(experiment_name)
    inpaint_folder = f"{output_folder}/{scenario_spec.scenario_string}::inpaint_CoPaint::conf_{config_name}::{forecast_date}"
    create_folders(inpaint_folder)
    
    with mlflow.start_run(run_name=f"inpaint_{scn_id}_{config_name}_{forecast_date}"):
        # Log scenario and run parameters
        params = {
            "scenario_id": scn_id,
            "ddpm_name": scenario_spec.ddpm_name,
            "unet_name": scenario_spec.unet_name,
            "dataset_name": scenario_spec.dataset_name,
            "transform_name": scenario_spec.transform_name,
            "enrich_name": scenario_spec.enrich_name,
            "scenario_string": scenario_spec.scenario_string,
            "forecast_date": forecast_date,
            "config_name": config_name,
            "image_size": image_size,
            "channels": channels,
            "batch_size": batch_size,
            "device": device,
            "output_folder": inpaint_folder
        }
        
        if run_id:
            params["training_run_id"] = run_id
            print(f"Loading model from MLflow run: {run_id}")
        else:
            params["model_path"] = model_path
            print(f"Loading model from filesystem: {model_path}")
            
        mlflow.log_params(params)
        
        # Create objects using simple helper
        print("Creating model, dataset, and transforms...")
        from .scenarios import create_scenario_objects
        ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_sd = create_scenario_objects(
            scenario_spec, season_setup, image_size, channels, batch_size, 1, device
        )
        
        # Load trained model
        model_source = load_model(ddpm, run_id, model_path)
        mlflow.log_param("model_source", model_source)
        
        # Log additional parameters
        mlflow.log_param("scaling_per_channel", scaling_per_channel.tolist())
        mlflow.log_param("data_mean", float(data_mean))
        mlflow.log_param("data_std", float(data_sd))
        mlflow.log_param("dataset_size", len(dataset))
        
        # Run inpainting
        run_inpainting(scenario_spec, ddpm, dataset, image_size, channels, batch_size, device, inpaint_folder, forecast_date, config_name)
        
        print(f"Inpainting completed for scenario {scn_id}, date {forecast_date}, config {config_name}")


def load_model(ddpm, run_id=None, model_path=None):
    """Load model from MLflow or filesystem. Note that mlflow run_id are unique and thus
    we don't need to know the training run experiment name here."""
    if run_id:
        try:
            # Load model from MLflow run
            # model_uri = f"runs:/{run_id}/model"
            # print(f"Loading PyTorch model from MLflow: {model_uri}")
            # loaded_model = mlflow.pytorch.load_model(model_uri)
            # 
            # # Replace the model in ddpm
            # ddpm.model = loaded_model
            
            # Instead try to load checkpoint from pth file instead of mlflow
            # model for additional state
            checkpoint_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="checkpoints"
            )
            # Find .pth file in downloaded artifacts
            import os
            for file in os.listdir(checkpoint_path):
                if file.endswith('.pth'):
                    full_checkpoint_path = os.path.join(checkpoint_path, file)
                    print(f"Loading additional checkpoint state from: {full_checkpoint_path}")
                    ddpm.load_model_checkpoint(full_checkpoint_path)
                    break
        
            return f"mlflow_run:{run_id}"
            
        except Exception as e:
            raise click.ClickException(f"Failed to load model from MLflow run {run_id}: {e}")
    
    elif model_path:
        try:
            print(f"Loading model checkpoint from filesystem: {model_path}")
            ddpm.load_model_checkpoint(model_path)
            return f"filesystem:{model_path}"
        except Exception as e:
            raise click.ClickException(f"Failed to load model from {model_path}: {e}")
    
    else:
        raise ValueError("Must provide either run_id or model_path")


def run_inpainting(scenario_spec, ddpm, dataset, image_size, channels, batch_size, device, inpaint_folder, forecast_date, config_name):
    """Run inpainting for a single scenario, date, and config"""
    
    # Parse forecast date
    try:
        forecast_dt = pd.to_datetime(forecast_date)
    except Exception as e:
        raise ValueError(f"Invalid forecast_date format: {forecast_date}. Use YYYY-MM-DD")
    
    # Validate config exists
    available_configs = copaint_config_library(ddpm.timesteps)
    if config_name not in available_configs:
        available = list(available_configs.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    
    conf = available_configs[config_name]
    
    print(f">>> Creating ground truth for {forecast_dt.date()}")
    
    # Determine flu season year dynamically based on forecast date
    season_first_year = str(season_setup.get_fluseason_year(forecast_dt))
    print(f">>> Detected flu season year: {season_first_year}")
    
    # Create ground truth for this date
    gt1 = ground_truth.GroundTruth(
        season_first_year=season_first_year,
        data_date=datetime.datetime.today(),
        mask_date=forecast_dt,
        channels=channels,
        image_size=image_size,
        nogit=True
    )
    
    # Prepare ground truth tensors
    gt = dataset.apply_transform(np.nan_to_num(gt1.gt_xarr.data, nan=0.0))
    gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(device)
    gt = torch.from_numpy(gt).type(torch.FloatTensor).to(device)
    
    print(f">>> Running CoPaint with config: {config_name}")
    
    try:
        # Create sampler
        sampler = O_DDIMSampler(
            use_timesteps=np.arange(ddpm.timesteps),
            conf=conf,
            betas=ddpm.betas,
            model_mean_type=None,
            model_var_type=None,
            loss_type=None
        )
        
        # Run sampling
        result = sampler.p_sample_loop(
            model_fn=ddpm.model,
            shape=(batch_size, channels, image_size, image_size),
            conf=conf,
            model_kwargs={
                "gt": gt.repeat(batch_size, 1, 1, 1),
                "gt_keep_mask": gt_keep_mask.repeat(batch_size, 1, 1, 1),
                "mymodel": True,
            }
        )
        
        # Process results
        fluforecasts = np.array(result['sample'].cpu())
        fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
        forecasts_national = fluforecasts_ti.sum(axis=-1)
    
        
        # Log forecast artifacts to MLflow and filesystem  
        log_forecast_artifacts(fluforecasts, fluforecasts_ti, forecasts_national, inpaint_folder)
        
        gt1.export_forecasts(
            fluforecasts_ti=fluforecasts_ti,
            forecasts_national=forecasts_national,
            directory=inpaint_folder,
            prefix="copaint",
            forecast_date=forecast_dt.date(),
            save_plot=True,
            nochecks=True
        )
        
        # Log metrics to MLflow
        metrics = {
            'forecast_mean': float(forecasts_national.mean()),
            'forecast_std': float(forecasts_national.std()),
            'forecast_min': float(forecasts_national.min()),
            'forecast_max': float(forecasts_national.max()),
            'num_samples': len(forecasts_national),
            'timesteps': ddpm.timesteps
        }
        
        mlflow.log_metrics(metrics)
        
        print(f">>> Generated {len(forecasts_national)} forecasts")
        print(f">>> Results saved to: {inpaint_folder}")
        
    except Exception as e:
        error_msg = f"Error with config {config_name}: {e}"
        print(f">>> {error_msg}")
        mlflow.log_param("error", error_msg)
        raise


def log_forecast_artifacts(fluforecasts, fluforecasts_ti, forecasts_national, inpaint_folder):
    """Log forecast results as MLflow artifacts"""
    import os
    
    # Save and log raw forecasts
    fluforecasts_path = os.path.join(inpaint_folder, "fluforecasts.npy")
    np.save(fluforecasts_path, fluforecasts)
    mlflow.log_artifact(fluforecasts_path, "forecasts")
    
    # Save and log inverse-transformed forecasts  
    fluforecasts_ti_path = os.path.join(inpaint_folder, "fluforecasts_ti.npy")
    np.save(fluforecasts_ti_path, fluforecasts_ti)
    mlflow.log_artifact(fluforecasts_ti_path, "forecasts")
    
    print(f">>> Logged forecast artifacts: {fluforecasts.shape[0]} samples")


if __name__ == '__main__':
    main()
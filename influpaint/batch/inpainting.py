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
import numpy as np
import pandas as pd
import torch
import mlflow

import epiframework
import ground_truth

sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler


# Need to initialize season_setup - this might need to be passed from elsewhere
season_setup = None  # This needs to be set based on your specific setup


@click.command()
@click.option("-s", "--scn_id", "scn_id", required=True, type=int, help="ID of the scenario to run inpainting for")
@click.option("-r", "--run_id", "run_id", type=str, help="MLflow run ID of the training run to use")
@click.option("-m", "--model_path", "model_path", type=str, help="Path to model checkpoint (.pth file) - alternative to --run_id")
@click.option("-e", "--experiment_name", "experiment_name", envvar="experiment_name", type=str, required=True,
              help="MLflow experiment name")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, 
              default='/work/users/c/h/chadi/influpaint_res/', show_default=True,
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
    
    # Configuration
    run_config = epiframework.TrainingRunConfig(
        image_size=image_size,
        channels=channels,
        batch_size=batch_size,
        epochs=800,  # Not used for inference, but needed for compatibility
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Get scenario specification
    scenario_spec = epiframework.ScenarioLibrary.get_training_scenario(scn_id)
    
    print(f"Running inpainting for scenario {scn_id}: {scenario_spec.scenario_string}")
    print(f"Device: {run_config.device}")
    print(f"Forecast date: {forecast_date}")
    print(f"Config: {config_name}")
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"inpaint_scenario_{scn_id}"):
        # Log scenario and run parameters
        params = {
            "scenario_id": scn_id,
            "unet_name": scenario_spec.unet_name,
            "dataset_name": scenario_spec.dataset_name,
            "transform_name": scenario_spec.transform_name,
            "enrich_name": scenario_spec.enrich_name,
            "scenario_string": scenario_spec.scenario_string,
            "forecast_date": forecast_date,
            "config_name": config_name,
            **run_config.to_dict()
        }
        
        if run_id:
            params["training_run_id"] = run_id
            print(f"Loading model from MLflow run: {run_id}")
        else:
            params["model_path"] = model_path
            print(f"Loading model from filesystem: {model_path}")
            
        mlflow.log_params(params)
        
        # Create heavy objects using factory
        print("Creating model, dataset, and transforms...")
        unet = epiframework.ObjectFactory.create_unet(scenario_spec, run_config)
        dataset = epiframework.ObjectFactory.create_dataset(scenario_spec, season_setup)
        transform, enrich, scaling_per_channel = epiframework.ObjectFactory.create_transforms(scenario_spec, dataset)
        
        # Configure dataset with transforms
        dataset.add_transform(
            transform=transform["reg"], 
            transform_inv=transform["inv"], 
            transform_enrich=enrich, 
            bypass_test=False
        )
        
        # Load trained model
        model_source = load_model(unet, run_id, model_path)
        mlflow.log_param("model_source", model_source)
        
        # Log additional parameters
        mlflow.log_param("scaling_per_channel", scaling_per_channel.tolist())
        
        # Run inpainting
        run_inpainting(scenario_spec, unet, dataset, run_config, outdir, forecast_date, config_name)
        
        print(f"Inpainting completed for scenario {scn_id}, date {forecast_date}, config {config_name}")


def load_model(unet, run_id=None, model_path=None):
    """Load model from MLflow or filesystem"""
    if run_id:
        try:
            # Load model from MLflow run
            model_uri = f"runs:/{run_id}/model"
            print(f"Loading PyTorch model from MLflow: {model_uri}")
            loaded_model = mlflow.pytorch.load_model(model_uri)
            
            # Replace the model in unet
            unet.model = loaded_model
            
            # Also try to load checkpoint for additional state
            try:
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
                        unet.load_model_checkpoint(full_checkpoint_path)
                        break
            except Exception as e:
                print(f"Warning: Could not load checkpoint artifacts: {e}")
                print("Continuing with PyTorch model only...")
            
            return f"mlflow_run:{run_id}"
            
        except Exception as e:
            raise click.ClickException(f"Failed to load model from MLflow run {run_id}: {e}")
    
    elif model_path:
        try:
            print(f"Loading model checkpoint from filesystem: {model_path}")
            unet.load_model_checkpoint(model_path)
            return f"filesystem:{model_path}"
        except Exception as e:
            raise click.ClickException(f"Failed to load model from {model_path}: {e}")
    
    else:
        raise ValueError("Must provide either run_id or model_path")


def run_inpainting(scenario_spec, unet, dataset, run_config, outdir, forecast_date, config_name):
    """Run inpainting for a single scenario, date, and config"""
    model_id = scenario_spec.scenario_string
    print(f">>> Running inpainting for {model_id}")
    print(f">>> Date: {forecast_date}, Config: {config_name}")
    
    # Parse forecast date
    try:
        forecast_dt = pd.to_datetime(forecast_date)
    except Exception as e:
        raise ValueError(f"Invalid forecast_date format: {forecast_date}. Use YYYY-MM-DD")
    
    # Validate config exists
    available_configs = epiframework.copaint_config_library(unet.timesteps)
    if config_name not in available_configs:
        available = list(available_configs.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    
    conf = available_configs[config_name]
    
    # Create output directory structure
    output_folder = f"{outdir}/forecasts_{datetime.date.today()}"
    epiframework.create_folders(output_folder)
    
    print(f">>> Creating ground truth for {forecast_dt.date()}")
    
    # Create ground truth for this date
    gt1 = ground_truth.GroundTruth(
        season_first_year="2022",
        data_date=datetime.datetime.today(),
        mask_date=forecast_dt,
        channels=run_config.channels,
        image_size=run_config.image_size,
        nogit=True
    )
    
    # Prepare ground truth tensors
    gt = dataset.apply_transform(gt1.gt_xarr.data)
    gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(run_config.device)
    gt = torch.from_numpy(gt).type(torch.FloatTensor).to(run_config.device)
    
    print(f">>> Running CoPaint with config: {config_name}")
    
    try:
        # Create sampler
        sampler = O_DDIMSampler(
            use_timesteps=np.arange(unet.timesteps),
            conf=conf,
            betas=unet.betas,
            model_mean_type=None,
            model_var_type=None,
            loss_type=None
        )
        
        # Run sampling
        result = sampler.p_sample_loop(
            model_fn=unet.model,
            shape=(run_config.batch_size, run_config.channels, run_config.image_size, run_config.image_size),
            conf=conf,
            model_kwargs={
                "gt": gt.repeat(run_config.batch_size, 1, 1, 1),
                "gt_keep_mask": gt_keep_mask.repeat(run_config.batch_size, 1, 1, 1),
                "mymodel": True,
            }
        )
        
        # Process results
        fluforecasts = np.array(result['sample'].cpu())
        fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
        forecasts_national = fluforecasts_ti.sum(axis=-1)
        
        # Save results
        forecast_fn = f"{model_id}::inpaint_CoPaint::conf_{config_name}"
        inpaint_folder = f"{output_folder}/{forecast_fn}"
        epiframework.create_folders(inpaint_folder)
        
        gt1.export_forecasts(
            fluforecasts_ti=fluforecasts_ti,
            forecasts_national=forecasts_national,
            directory=inpaint_folder,
            prefix=forecast_fn,
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
            'timesteps': unet.timesteps
        }
        
        mlflow.log_metrics(metrics)
        
        print(f">>> Generated {len(forecasts_national)} forecasts")
        print(f">>> Results saved to: {inpaint_folder}")
        
    except Exception as e:
        error_msg = f"Error with config {config_name}: {e}"
        print(f">>> {error_msg}")
        mlflow.log_param("error", error_msg)
        raise


if __name__ == '__main__':
    main()
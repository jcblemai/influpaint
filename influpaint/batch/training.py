#!/usr/bin/env python3
"""
Training script for diffusion models using the scenarios and config created.

Usage:
    python train.py -s 5 -e "my_training_experiment" -d "/path/to/output"
"""

import datetime
import click
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from .scenarios import get_training_scenario, create_scenario_objects
from .config import get_git_revision_short_hash, create_folders
from ..utils import plotting as idplots
from influpaint.utils import SeasonAxis

season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)

@click.command()
@click.option("-s", "--scn_id", "scn_id", required=True, type=int, help="ID of the scenario to train")
@click.option("-e", "--experiment_name", "experiment_name", envvar="experiment_name", type=str, required=True,
            help="MLflow experiment name (e.g. 'paper-2025-06_training')")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, 
            default='/users/c/h/chadi/influpaint_res/', show_default=True, 
            help="Where to write model checkpoints")
@click.option("--image_size", default=64, type=int, help="Image size")
@click.option("--channels", default=1, type=int, help="Number of channels")
@click.option("--batch_size", default=512, type=int, help="Batch size")
@click.option("--epochs", default=3000, type=int, help="Number of epochs")
def main(scn_id, experiment_name, outdir, image_size, channels, batch_size, epochs):
    """Train a diffusion model for a specific scenario"""
    
    # Get scenario specification
    scenario_spec = get_training_scenario(scn_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training scenario {scn_id}: {scenario_spec.scenario_string}")
    print(f"Device: {device}")

    # Experiment setup
    model_folder = f"{outdir}{get_git_revision_short_hash()}_{experiment_name}_{datetime.date.today()}"
    create_folders(model_folder)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"train_{scn_id}"):
        # Log scenario and run parameters
        mlflow.log_params({
            "scenario_id": scn_id,
            "ddpm_name": scenario_spec.unet_name,
            "dataset_name": scenario_spec.dataset_name,
            "transform_name": scenario_spec.transform_name,
            "enrich_name": scenario_spec.enrich_name,
            "scenario_string": scenario_spec.scenario_string,
            "image_size": image_size,
            "channels": channels,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": device
        })
        
        # Create objects using simple helper
        print("Creating model, dataset, and transforms...")
        ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_sd = create_scenario_objects(
            scenario_spec, season_setup, image_size, channels, batch_size, epochs, device
        )
        
        # Log additional parameters
        mlflow.log_param("scaling_per_channel", scaling_per_channel.tolist())
        mlflow.log_param("data_mean", float(data_mean))
        mlflow.log_param("data_std", float(data_sd))
        mlflow.log_param("dataset_size", len(dataset))
        
        # Run training
        run_training(scenario_spec, ddpm, dataset, image_size, channels, batch_size, epochs, device, model_folder)
        
        print(f"Training completed for scenario {scn_id}")


def run_training(scenario_spec, ddpm, dataset, image_size, channels, batch_size, epochs, device, model_folder):
    """Run training for a scenario"""
    # Create output directory

    mlflow.log_param("output_folder", model_folder)
    
    model_id = scenario_spec.scenario_string
    print(f">>> Training {model_id}")
    print(f">>> Saving to {model_folder}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f">>> Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    
    # Log training start
    mlflow.log_metric("training_started", 1)
    
    print(">>> Starting training...")
    losses = ddpm.train(dataloader, mlflow_logging=True)
    
    # Save model checkpoint
    checkpoint_path = f"{model_folder}/{model_id}::{epochs}.pth"
    ddpm.write_train_checkpoint(save_path=checkpoint_path)
    print(f">>> Model saved to {checkpoint_path}")

    mlflow.log_params({"output_folder":model_folder,
                    "model_path": checkpoint_path})
    # Log model to MLflow
    mlflow.pytorch.log_model(ddpm.model, name="model")
    mlflow.log_artifact(checkpoint_path, "checkpoints")
    
    # Generate and log sample images
    print(">>> Generating samples...")
    samples = ddpm.sample()

    log_samples_as_artifacts(samples, dataset, scenario_spec.scenario_string, model_folder)
    
    # Create and log sample plot
    fig, axes = plot_sample(samples, dataset, idplots)
    plt.savefig(f"{model_folder}/samples_{model_id}.png", bbox_inches='tight', dpi=100)
    mlflow.log_figure(fig, "generated_samples.png")
    
    # Log loss plots to MLflow
    log_loss_plots_to_mlflow(losses)
    
    # Log training completion metrics
    mlflow.log_metrics({
        "training_completed": 1,
        "final_epoch": epochs,
        "samples_generated": samples[-1].shape[0],
        "final_loss": losses[-1] if losses else 0,
        "avg_loss_last_100": sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else sum(losses) / len(losses)
    })
    
    print(f">>> Training completed for {model_id}")


def plot_sample(samples, dataset, idplots):
    """Create sample visualization plot"""
    import matplotlib.pyplot as plt
    
    # Create sample visualization
    fig, axes = plt.subplots(8, 7, figsize=(16, 16), dpi=100)
    axes = axes.flatten()
    
    for ipl in range(min(51, len(axes))):
        ax = axes[ipl]
        if ipl < samples[-1].shape[0]:
            # Show transformed sample
            sample_image = dataset.apply_transform_inv(samples[-1][ipl])
            idplots.show_tensor_image(sample_image, ax=ax, place=ipl, multi=True)
        ax.axis('off')
    
    # Hide unused axes
    for ipl in range(51, len(axes)):
        axes[ipl].axis('off')
    
    plt.tight_layout()
    return fig, axes


def log_loss_plots_to_mlflow(losses):
    """Log loss plots directly to MLflow"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not losses:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Full loss curve
    axes[0].plot(np.arange(len(losses)), np.array(losses))
    axes[0].set_title('Full Training Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Last 1000 steps
    last_1000 = losses[-1000:] if len(losses) > 1000 else losses
    axes[1].plot(np.arange(len(last_1000)), np.array(last_1000))
    axes[1].set_title('Last 1000 Steps')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Last 100 steps
    last_100 = losses[-100:] if len(losses) > 100 else losses
    axes[2].plot(np.arange(len(last_100)), np.array(last_100))
    axes[2].set_title('Last 100 Steps')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Log figure directly to MLflow
    mlflow.log_figure(fig, "training_loss_curves.png")
    plt.close()


def log_samples_as_artifacts(samples, dataset, scenario_string, model_folder):
    """Log generated samples as artifacts in multiple formats"""
    import numpy as np
    import os
    
    raw_samples_path = os.path.join(model_folder, "raw_samples.npy")
    np.save(raw_samples_path, samples)
    mlflow.log_artifact(raw_samples_path, "samples")
    # 2. Log inverse-transformed samples (original scale) 
    inv_samples_path = os.path.join(model_folder, "inverse_transformed_samples.npy")
    inv_samples = []
    for i in range(samples[-1].shape[0]):
        inv_sample = dataset.apply_transform_inv(samples[-1][i])
        inv_samples.append(inv_sample)
    inv_samples = np.array(inv_samples)
    np.save(inv_samples_path, inv_samples)
    mlflow.log_artifact(inv_samples_path, "samples")

    full_payload_array = season_setup.add_axis_to_numpy_array(np.array(inv_samples), truncate=True)
    fig, ax = idplots.plot_us_grid(
        data=full_payload_array,
        season_axis=season_setup,
        sample_idx=list(np.arange(0, 100, step=5)),
        multi_line=True,
        sharey=False,
        past_ground_truth=True,
    )
    plt.savefig(f"{model_folder}/us_map_{scenario_string}.png", bbox_inches='tight', dpi=300)
    mlflow.log_figure(fig, "sample_us_map.png")

    # 4. Log sample statistics as metrics
    mlflow.log_metrics({
        "sample_count": samples[-1].shape[0],
        "sample_min": float(samples[-1].min()),
        "sample_max": float(samples[-1].max()),
        "sample_mean": float(samples[-1].mean()),
        "sample_std": float(samples[-1].std()),
        "inv_sample_min": inv_samples.min(),
        "inv_sample_max": inv_samples.max(),
        "inv_sample_mean": inv_samples.mean(),
        "inv_sample_std": inv_samples.std()
    })


if __name__ == '__main__':
    main()
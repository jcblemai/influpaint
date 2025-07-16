#!/usr/bin/env python3
"""
Training script for diffusion models using the epiframework.

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


# Need to initialize season_setup - this might need to be passed from elsewhere
# For now, keeping the same structure but will need to be set properly
season_setup = None  # This needs to be set based on your specific setup


@click.command()
@click.option("-s", "--scn_id", "scn_id", required=True, type=int, help="ID of the scenario to train")
@click.option("-e", "--experiment_name", "experiment_name", envvar="experiment_name", type=str, required=True,
              help="MLflow experiment name")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, 
              default='/work/users/c/h/chadi/influpaint_res/', show_default=True, 
              help="Where to write model checkpoints")
@click.option("--image_size", default=64, type=int, help="Image size")
@click.option("--channels", default=1, type=int, help="Number of channels")
@click.option("--batch_size", default=512, type=int, help="Batch size")
@click.option("--epochs", default=800, type=int, help="Number of epochs")
def main(scn_id, experiment_name, outdir, image_size, channels, batch_size, epochs):
    """Train a diffusion model for a specific scenario"""
    
    # Get scenario specification
    scenario_spec = get_training_scenario(scn_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training scenario {scn_id}: {scenario_spec.scenario_string}")
    print(f"Device: {device}")
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"train_scenario_{scn_id}"):
        # Log scenario and run parameters
        mlflow.log_params({
            "scenario_id": scn_id,
            "unet_name": scenario_spec.unet_name,
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
        unet, dataset, transform, enrich, scaling_per_channel = create_scenario_objects(
            scenario_spec, season_setup, image_size, channels, batch_size, epochs, device
        )
        
        # Log additional parameters
        mlflow.log_param("scaling_per_channel", scaling_per_channel.tolist())
        mlflow.log_param("dataset_size", len(dataset))
        
        # Run training
        run_training(scenario_spec, unet, dataset, image_size, channels, batch_size, epochs, device, outdir)
        
        print(f"Training completed for scenario {scn_id}")


def run_training(scenario_spec, unet, dataset, image_size, channels, batch_size, epochs, device, outdir):
    """Run training for a scenario"""
    # Create output directory
    model_folder = f"{outdir}{get_git_revision_short_hash()}_{datetime.date.today()}"
    create_folders(model_folder)
    
    model_id = scenario_spec.scenario_string
    print(f">>> Training {model_id}")
    print(f">>> Saving to {model_folder}")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f">>> Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    
    # Log training start
    mlflow.log_metric("training_started", 1)
    
    # Training
    print(">>> Starting training...")
    unet.train(dataloader=dataloader)
    
    # Save model checkpoint
    checkpoint_path = f"{model_folder}/{model_id}::{epochs}.pth"
    unet.write_train_checkpoint(save_path=checkpoint_path)
    print(f">>> Model saved to {checkpoint_path}")
    
    # Log model to MLflow
    mlflow.pytorch.log_model(unet.model, "model")
    mlflow.log_artifact(checkpoint_path, "checkpoints")
    
    # Generate and save sample images
    print(">>> Generating samples...")
    samples = unet.sample()
    
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
    
    samples_path = f"{model_folder}/{model_id}-{epochs}::samples.pdf"
    plt.tight_layout()
    plt.savefig(samples_path, bbox_inches='tight')
    mlflow.log_artifact(samples_path, "samples")
    plt.close()
    
    print(f">>> Samples saved to {samples_path}")
    
    # Log training completion metrics
    mlflow.log_metrics({
        "training_completed": 1,
        "final_epoch": epochs,
        "samples_generated": samples[-1].shape[0]
    })
    
    print(f">>> Training completed for {model_id}")


if __name__ == '__main__':
    main()
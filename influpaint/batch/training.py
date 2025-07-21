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


# Need to initialize season_setup - this might need to be passed from elsewhere
# For now, keeping the same structure but will need to be set properly
season_setup = None  # This needs to be set based on your specific setup


@click.command()
@click.option("-s", "--scn_id", "scn_id", required=True, type=int, help="ID of the scenario to train")
@click.option("-e", "--experiment_name", "experiment_name", envvar="experiment_name", type=str, required=True,
            help="MLflow experiment name")
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
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"train_scenario_{scn_id}"):
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
        run_training(scenario_spec, ddpm, dataset, image_size, channels, batch_size, epochs, device, outdir)
        
        print(f"Training completed for scenario {scn_id}")


def run_training(scenario_spec, ddpm, dataset, image_size, channels, batch_size, epochs, device, outdir):
    """Run training for a scenario"""
    # Create output directory
    model_folder = f"{outdir}{get_git_revision_short_hash()}_{datetime.date.today()}"
    create_folders(model_folder)
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
    mlflow.pytorch.log_model(ddpm.model, "model")
    mlflow.log_artifact(checkpoint_path, "checkpoints")
    
    # Generate and log sample images
    print(">>> Generating samples...")
    samples = ddpm.sample()

    log_samples_as_artifacts(samples, dataset, scenario_spec.scenario_string)
    
    # Create and log sample plot
    fig, axes = plot_sample(samples, dataset, idplots)
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


def log_samples_as_artifacts(samples, dataset, scenario_string):
    """Log generated samples as artifacts in multiple formats"""
    import numpy as np
    import tempfile
    import os
    
    # Create temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Log raw samples as numpy array
        raw_samples_path = os.path.join(temp_dir, "raw_samples.npy")
        np.save(raw_samples_path, samples[-1])
        mlflow.log_artifact(raw_samples_path, "samples")
        
        # 2. Log inverse-transformed samples (original scale) 
        inv_samples_path = os.path.join(temp_dir, "inverse_transformed_samples.npy")
        inv_samples = []
        for i in range(min(samples[-1].shape[0], 100)):  # Log first 100 samples
            inv_sample = dataset.apply_transform_inv(samples[-1][i])
            inv_samples.append(inv_sample)
        np.save(inv_samples_path, np.array(inv_samples))
        mlflow.log_artifact(inv_samples_path, "samples")
        
        # 3. Log sample metadata
        metadata_path = os.path.join(temp_dir, "sample_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Scenario: {scenario_string}\n")
            f.write(f"Number of samples: {samples[-1].shape[0]}\n")
            f.write(f"Sample shape: {samples[-1].shape}\n")
            f.write(f"Sample dtype: {samples[-1].dtype}\n")
            f.write(f"Sample device: {samples[-1].device}\n")
            f.write(f"Sample min: {samples[-1].min().item():.6f}\n")
            f.write(f"Sample max: {samples[-1].max().item():.6f}\n")
            f.write(f"Sample mean: {samples[-1].mean().item():.6f}\n")
            f.write(f"Sample std: {samples[-1].std().item():.6f}\n")
            f.write(f"Inverse transformed samples shape: {np.array(inv_samples).shape}\n")
            f.write(f"Inverse transformed min: {np.array(inv_samples).min():.6f}\n")
            f.write(f"Inverse transformed max: {np.array(inv_samples).max():.6f}\n")
            f.write(f"Inverse transformed mean: {np.array(inv_samples).mean():.6f}\n")
        mlflow.log_artifact(metadata_path, "samples")
        
        # 4. Log sample statistics as metrics
        mlflow.log_metrics({
            "sample_count": samples[-1].shape[0],
            "sample_min": samples[-1].min().item(),
            "sample_max": samples[-1].max().item(),
            "sample_mean": samples[-1].mean().item(),
            "sample_std": samples[-1].std().item(),
            "inv_sample_min": np.array(inv_samples).min(),
            "inv_sample_max": np.array(inv_samples).max(),
            "inv_sample_mean": np.array(inv_samples).mean(),
            "inv_sample_std": np.array(inv_samples).std()
        })
        
        # 5. Create spatially-aware samples with season axis for US map plotting
        create_us_map_visualization(inv_samples, dataset, scenario_string, temp_dir)
        
        print(f">>> Logged {len(inv_samples)} samples as artifacts")


def create_us_map_visualization(inv_samples, dataset, scenario_string, temp_dir):
    """Create US map visualization for generated samples with proper padding handling"""
    try:
        import numpy as np
        import xarray as xr
        import os
        from ..utils.season_axis import SeasonAxis
        
        # Create season axis for US states (same as used in dataset creation)
        season_axis = SeasonAxis.for_flusight(
            remove_us=True,           # Remove US national total
            remove_territories=True   # Remove territories for cleaner visualization
        )
        
        # Take first 20 samples for US map visualization
        n_viz_samples = min(20, len(inv_samples))
        viz_samples = np.array(inv_samples)[:n_viz_samples]
        
        # Create xarray with proper spatial coordinates including padding
        # Expected format: (sample, feature, season_week, place)
        n_features = viz_samples.shape[1]  # Usually 1 for single feature
        n_weeks = viz_samples.shape[2]     # Number of time steps
        n_places_total = viz_samples.shape[3]  # Total including padding
        n_places_real = len(season_axis.locations)  # Real locations
        
        # Create padded location coordinates (same as build_dataset_from_framelist)
        place_coords = season_axis.locations + [""] * (n_places_total - n_places_real)
        
        # Create xarray with proper coordinates including padding
        samples_xarray = xr.DataArray(
            viz_samples,
            coords={
                'sample': np.arange(n_viz_samples),
                'feature': np.arange(n_features),
                'season_week': np.arange(1, n_weeks + 1),  # Season weeks 1-53
                'place': place_coords  # US state codes + padding
            },
            dims=['sample', 'feature', 'season_week', 'place']
        )
        
        # Remove padding for visualization (select only real locations)
        samples_xarray_clean = samples_xarray.sel(place=season_axis.locations)
        
        # 1. Create US map visualization
        fig_us_map, _ = idplots.plot_us_grid(
            data=samples_xarray_clean,
            season_axis=season_axis,
            sample_idx=list(range(n_viz_samples)),  # Plot all samples
            multi_line=True,                        # Multiple lines per state
            show_us_summary=True,                   # Include US summary
            sharey=False,                          # Independent y-axes
            title_suffix=f' - {scenario_string}',  # Add scenario info
            alpha_fill=0.1,                        # Light fill for readability
            line_width=1.0                         # Thinner lines for multi-line
        )
        
        # Log the US map visualization
        mlflow.log_figure(fig_us_map, "generated_samples_us_map.png")
        plt.close(fig_us_map)
        
        # 2. Create detailed few-sample visualization
        # Convert samples back to the format expected by plot_few_sample
        # plot_few_sample expects samples in format: samples[-1] shape (batch_size, channels, height, width)
        # We need to reconstruct this from our inv_samples
        
        # Take first 5 samples for detailed visualization
        n_detail_samples = min(5, len(inv_samples))
        
        # Create a mock samples structure for plot_few_sample
        # inv_samples has shape (n_samples, n_features, n_weeks, n_places_total)
        # We need to reshape to (n_samples, n_features, height, width) format
        
        # Reshape for plot_few_sample: (n_samples, n_features, n_weeks, n_places_real)
        detail_samples = viz_samples[:n_detail_samples, :, :, :n_places_real]
        
        # Create mock samples list (plot_few_sample expects samples[-1])
        class MockSamples:
            def __init__(self, final_samples):
                self.final_samples = final_samples
            
            def __getitem__(self, idx):
                if idx == -1:
                    return self.final_samples
                return self.final_samples[idx]
        
        mock_samples = MockSamples(detail_samples)
        
        # Generate the detailed few-sample visualization
        fig_detail = idplots.plot_few_sample(
            samples=mock_samples,
            dataset=dataset,
            indices=list(range(n_detail_samples)),
            season_labels=[f'({chr(97+i)}) Sample {i+1}' for i in range(n_detail_samples)],
            title=f'Generated Samples - {scenario_string}'
        )
        
        # Log the detailed few-sample visualization
        mlflow.log_figure(fig_detail, "generated_samples_detailed.png")
        plt.close(fig_detail)
        
        # Save the spatially-aware xarray (with padding for consistency)
        samples_xarray_path = os.path.join(temp_dir, "samples_with_spatial_coords.nc")
        samples_xarray.to_netcdf(samples_xarray_path)
        mlflow.log_artifact(samples_xarray_path, "samples")
        
        print(f">>> Created US map visualization with {n_viz_samples} samples")
        print(f">>> Created detailed few-sample visualization with {n_detail_samples} samples")
        print(f">>> Spatial dimensions: {n_features} features × {n_weeks} weeks × {n_places_real} locations (+ {n_places_total - n_places_real} padding)")
        
    except Exception as e:
        print(f">>> Warning: Could not create US map visualization: {e}")
        print(f">>> Continuing with standard sample logging...")


if __name__ == '__main__':
    main()
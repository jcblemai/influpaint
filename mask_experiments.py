# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (diffusion_torch65)
#     language: python
#     name: diffusion_torch6
# ---

# %%
import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import mlflow

sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler

from influpaint.batch.scenarios import get_training_scenario, create_scenario_objects
from influpaint.batch.config import copaint_config_library, create_folders, get_git_revision_short_hash
from influpaint.utils import ground_truth, SeasonAxis
import influpaint.utils.plotting as ip_plot

# %%
# Configuration
CONFIG_NAME = "celebahq_noTTJ5"
IMAGE_SIZE = 64
CHANNELS = 1
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT_NAME = "paper-2025-07-22_training"
SCENARIO_ID = 868
# Season toggle - set to desired flu season year
SEASON_YEAR = "2024"  # Change this to switch seasons


print(f"Running mask experiments for scenario {SCENARIO_ID}")
print(f"Device: {DEVICE}")

# %%
# Setup season and scenario
season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
scenario_spec = get_training_scenario(SCENARIO_ID)

print(f"Scenario: {scenario_spec.scenario_string}")
print(f"DDPM: {scenario_spec.ddpm_name}, UNet: {scenario_spec.unet_name}")
print(f"Dataset: {scenario_spec.dataset_name}, Transform: {scenario_spec.transform_name}")

# %%
# Create model objects
print("Creating model, dataset, and transforms...")
ddpm, dataset, transform, enrich, scaling_per_channel, data_mean, data_sd = create_scenario_objects(
    scenario_spec, season_setup, IMAGE_SIZE, CHANNELS, BATCH_SIZE, 1, DEVICE
)

print(f"Dataset size: {len(dataset)}")
print(f"Data mean: {data_mean}, Data std: {data_sd}")

# %%
# Import functions from existing files
from influpaint.batch.inpainting import load_model
from influpaint.batch.generate_inpainting_jobs import get_finished_models


print(f"Finding run ID for scenario {SCENARIO_ID} in {EXPERIMENT_NAME}...")
finished_models = get_finished_models(EXPERIMENT_NAME)

# Find the specific model for i804
target_model = None
for model in finished_models:
    if model['scenario_id'] == SCENARIO_ID:
        target_model = model
        break

if target_model is None:
    raise ValueError(f"No finished model found for scenario {SCENARIO_ID} in {EXPERIMENT_NAME}")

RUN_ID = target_model['run_id']
print(f"Found run ID: {RUN_ID}")
print(f"Model scenario string: {target_model['scenario_string']}")

# Load the model
print(f"Loading i804 model from MLflow run: {RUN_ID}")
model_source = load_model(ddpm, run_id=RUN_ID)
print(f"Model loaded successfully: {model_source}")

# %%

results = {}
output_dir = f"mask_experiments_{SCENARIO_ID}_{CONFIG_NAME}"
create_folders(output_dir)

# %%
# Setup CoPaint configuration
available_configs = copaint_config_library(ddpm.timesteps)
if CONFIG_NAME not in available_configs:
    available = list(available_configs.keys())
    raise ValueError(f"Config '{CONFIG_NAME}' not found. Available: {available}")

conf = available_configs[CONFIG_NAME]
print(f"Using CoPaint config: {CONFIG_NAME}")

# %%
# Mask generation functions. 1 = mask out, 0 = keep

def mask_dates(mask, dates, season_setup):
    """
    Helper function to mask specific dates
    
    Args:
        ground_truth: GroundTruth object with season_setup
        dates: List of dates (strings 'YYYY-MM-DD' or datetime objects) or date range tuples
    """
    new_mask = mask.copy()
    for date_item in dates:
        if isinstance(date_item, tuple):
            # Date range (start_date, end_date)
            start_date, end_date = date_item
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            start_week = season_setup.get_season_week(start_date)
            end_week = season_setup.get_season_week(end_date)
            
            new_mask[:, start_week-1:end_week, :] = 0
        else:
            # Single date
            if isinstance(date_item, str):
                date_item = pd.to_datetime(date_item)
            
            week = season_setup.get_season_week(date_item)
            
            new_mask[:, week-1, :] = 0
    
    return new_mask

def mask_subpop(mask, location_codes, season_setup):
    """
    Helper function to mask specific locations
    
    Args:
        ground_truth: GroundTruth object with season_setup
        location_codes: List of location codes to mask out
    """
    new_mask = mask.copy()
    
    for location_code in location_codes:
        assert location_code in season_setup.locations, f"Location {location_code} not found in season_setup.locations"
        location_idx = season_setup.locations.index(location_code)
        new_mask[:, :, location_idx] = 0
    
    return new_mask



# %%
# Setup ground truth for the forecast date
for season_first_year in ["2023", "2024"]:
    gt1 = ground_truth.GroundTruth(
        season_first_year=season_first_year,
        data_date=datetime.datetime.today(),
        mask_date=pd.to_datetime(f"{int(season_first_year)+1}-07-29"),
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        nogit=True
    )

    # Prepare ground truth tensors
    gt_original = dataset.apply_transform(np.nan_to_num(gt1.gt_xarr.data, nan=0.0))
    gt_keep_mask_original = gt1.gt_keep_mask
    gt_tensor = torch.from_numpy(gt_original).type(torch.FloatTensor).to(DEVICE)

    print(f"Ground truth shape: {gt_original.shape}")
    print(f"Original mask shape: {gt_keep_mask_original.shape}")

    masks = {}

    # date stuff:
    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_dates(gt_keep_mask, [(f"{season_first_year}-12-07", f"{int(season_first_year)+1}-01-07")], gt1.season_setup)
    masks['missing_midseason'] = gt_keep_mask

    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_dates(gt_keep_mask, [("{int(season_first_year)+1}-02-01", "{int(season_first_year)+1}-02-15")], gt1.season_setup)
    masks['missing_midseason_peak'] = gt_keep_mask


    # location stuff
    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_subpop(gt_keep_mask, gt1.season_setup.locations[:len(gt1.season_setup.locations)//2], gt1.season_setup)
    masks['missing_half_subpop'] = gt_keep_mask

    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_subpop(gt_keep_mask, ["37"], gt1.season_setup)
    masks['missing_nc'] = gt_keep_mask

    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_subpop(gt_keep_mask, ["06"], gt1.season_setup)
    masks['missing_ca'] = gt_keep_mask

    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_subpop(gt_keep_mask, ["17"], gt1.season_setup)
    masks['missing_ca'] = gt_keep_mask

    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_subpop(gt_keep_mask, ["06", "37", "58", "42", "53"], gt1.season_setup)
    masks['missing_five'] = gt_keep_mask

    gt_keep_mask = np.ones((CHANNELS,IMAGE_SIZE,IMAGE_SIZE))
    gt_keep_mask = mask_subpop(gt_keep_mask, ["13", "49", "18" "36", "24"], gt1.season_setup)
    masks['missing_five'] = gt_keep_mask
    

    for i, (name, mask) in enumerate(masks.items()):
        ip_plot.plot_mask(gt_xarr=gt1.gt_xarr, gt_keep_mask=mask, channel=0)

    for mask_name, mask in masks.items():
        # Convert mask to tensor
        gt_keep_mask_tensor = torch.from_numpy(mask).type(torch.FloatTensor).to(DEVICE)
        
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
            shape=(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            conf=conf,
            model_kwargs={
                "gt": gt_tensor.repeat(BATCH_SIZE, 1, 1, 1),
                "gt_keep_mask": gt_keep_mask_tensor.repeat(BATCH_SIZE, 1, 1, 1),
                "mymodel": True,
            }
        )
        
        # Process results
        fluforecasts = np.array(result['sample'].cpu())
        fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
        
        mask_dir = f"{output_dir}/{mask_name}_season{season_first_year}"
        create_folders(mask_dir)
        
        np.save(f"{mask_dir}/fluforecasts.npy", fluforecasts)
        np.save(f"{mask_dir}/fluforecasts_ti.npy", fluforecasts_ti)
        np.save(f"{mask_dir}/mask.npy", mask)
        np.save(f"{mask_dir}/ground_truth.npy", gt1.gt_xarr.data)

print(f"\nResults saved to: {output_dir}")
print("Mask experiment completed!")

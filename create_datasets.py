# -*- coding: utf-8 -*-
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
#     display_name: diffusion_torch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Create dataset
# create calibration datasets, requires flu_dataset_explorerNB to have run

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import epiweeks
import warnings
import importlib
import tqdm
import datetime

# InfluPaint modular imports
from influpaint.utils import SeasonAxis
from influpaint.datasets import build_frames
from influpaint.utils import plotting as idplots
from influpaint.utils import converters
from influpaint.datasets import mixer as dataset_mixer
from influpaint.datasets import read_datasources
season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)

today = datetime.datetime.now().strftime("%Y-%m-%d")
# create the folder if exists:
Path("training_datasets").mkdir(parents=True, exist_ok=True)

def build_dataset_from_framelist(frame_list):
    main_origins = []
    for i, frame in enumerate(frame_list):
        df = frame_list[i]
        df["fluseason"] = i
        frame_list[i] = df
        assert df.season_week.max() == 53 and df.season_week.min() == 1, f"Frame {i} has invalid season_week range: {df.season_week.min()} to {df.season_week.max()}"
        assert set(df["location_code"].unique()) == set(season_setup.locations), f"Frame {i} has invalid locations: {set(df['location_code'].unique())} vs {set(season_setup.locations)}"
        # Add the most occurring origin for this frame
        main_origins.append(df["origin"].mode()[0] if not df["origin"].mode().empty else None)
    assert set(pd.concat(frame_list).fluseason.unique()) == set(range(len(frame_list)))

    all_frames_df = pd.concat(frame_list).reset_index(drop=True)
    array_list = converters.dataframe_to_arraylist(df=all_frames_df, season_setup=season_setup)

    array = np.array(array_list)

    flu_payload_array = xr.DataArray(array, 
                    coords={'sample': np.arange(array.shape[0]),
                        'feature': np.arange(array.shape[1]),
                        'season_week': np.arange(1, array.shape[2]+1),
                        'place': season_setup.locations + [""]*(array.shape[3] - len(season_setup.locations))}, 
                    dims=["sample", "feature", "season_week", "place"])
    return flu_payload_array, main_origins


# %%
all_datasets_df = pd.read_parquet("Flusight/flu-datasets/all_datasets.parquet")
for dH1 in all_datasets_df['datasetH1'].unique():
    h1df= all_datasets_df[all_datasets_df['datasetH1'] == dH1]
    print(f"datasetH1: {dH1}, nH2= {len(h1df['datasetH2'].unique())}")
    for dH2 in h1df['datasetH2'].unique():
        h2df = h1df[h1df['datasetH2'] == dH2]
        print(f" -  datasetH2: {dH2}, shape: {h2df.shape}, years: {len(h2df['fluseason'].unique())}, samples: {len(h2df['sample'].unique())} ===> n_frames={len(h2df['fluseason'].unique())* len(h2df['sample'].unique())}")


# %% [markdown]
# * 1080 Frame SMH
# * 160 FlepiR1
# * 1240 total synthetic
# * 20 for the sum of all surveilalnce dataset
#
#

# %%
all_datasets_df = pd.read_parquet("Flusight/flu-datasets/all_datasets.parquet")
season_identifiers = ['datasetH1', 'datasetH2', 'fluseason', 'sample']
time_identifier = 'season_week'

# Step 1: Sum across all locations for each season-time combination
# (assuming locations are represented by rows not explicitly grouped)
location_sums = all_datasets_df.groupby(season_identifiers + [time_identifier])['value'].sum().reset_index()
# Step 2: Find the peak (max over time) for each season
season_peaks = location_sums.groupby(season_identifiers)['value'].max().reset_index()
# Get all unique datasetH1 values
unique_dH1 = all_datasets_df['datasetH1'].unique()
fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=False)
axs = axs.flatten()
for i, dH1 in enumerate(unique_dH1[:4]):
    h1_peaks = season_peaks[season_peaks['datasetH1'] == dH1]
    n_h2 = len(all_datasets_df[all_datasets_df['datasetH1'] == dH1]['datasetH2'].unique())
    axs[i].hist(h1_peaks['value'], bins=100, alpha=0.7, edgecolor='black')
    axs[i].set_title(f"Season Peaks: {dH1}")
    axs[i].set_xlabel('Peak Value')
    axs[i].set_ylabel('Frequency')
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
scaling_distribution = season_peaks[season_peaks['datasetH1'] == 'SMH_R4-R5'].value

# %%

# %%
DATASET_GRIDS = {
    "100S": {
        "fluview":     {"multiplier": 26, "to_scale": True},
        "flusurv": {"multiplier": 26}
    },
    # 2. Hybrid 70 % surveillance / 30 % modeling 
    "70S30M": {
        "fluview":     {"proportion": 0.37, "total": 3000, "to_scale": True},
        "flusurv":     {"proportion": 0.33, "total": 3000},
        "flepiR1":     {"proportion": 0.05, "total": 3000},
        "SMH_R4-R5":   {"proportion": 0.25, "total": 3000}
    },
    # 3. Half-half (uncertainty stress-test)
    "30S70M": {
        "fluview":     {"proportion": 0.15, "total": 3000, "to_scale": True},
        "flusurv":     {"proportion": 0.15, "total": 3000},
        "flepiR1":     {"proportion": 0.05, "total": 3000},
        "SMH_R4-R5":   {"proportion": 0.65, "total": 3000}
    },
    "100M": {
        "flepiR1":  {"multiplier": 1},
        "SMH_R4-R5": {"multiplier": 1}
    },
}


# %%
frame_list = dataset_mixer.build_frames(all_datasets_df, {
    "fluview":     {"multiplier": 1, "to_scale":True},
    "flusurv": {"multiplier": 1}
            }, 
            season_axis=season_setup, 
            fill_missing_locations="random",
            scaling_distribution=scaling_distribution)

# %%
all_frames_df = pd.concat(frame_list)
flu_payload_array, main_origins = build_dataset_from_framelist(frame_list)
from importlib import reload
import influpaint.utils.plotting as idplots
reload(idplots)


idx = 5  # Example index for plotting
print(f"Plotting dataset main origins: {main_origins[idx]}")
  # Plot all seasons as light lines
fig, ax = idplots.plot_us_grid(
      data=flu_payload_array,
      season_axis=season_setup,
      sample_idx=list(np.arange(13)),
      multi_line=True,
      sharey=False,

  )

fig, ax = idplots.plot_us_grid(
      data=flu_payload_array,
      season_axis=season_setup,
      sample_idx=list(np.arange(14, 20)),
      multi_line=True,
      sharey=False,

  )

# %%
for ds_name, mix_cfg in DATASET_GRIDS.items():
    frame_list = dataset_mixer.build_frames(all_datasets_df, mix_cfg, 
                    season_axis=season_setup, 
                    fill_missing_locations="random",
                    scaling_distribution=scaling_distribution)
    flu_payload_array, main_origins = build_dataset_from_framelist(frame_list)
    flu_payload_array = flu_payload_array.assign_attrs(
                main_origins=list(main_origins),
                mix_cfg=mix_cfg.__str__()
    )
    flu_payload_array.to_netcdf(f"training_datasets/TS_{ds_name}_{today}.nc")

# %%
flu_payload_array

# %% [markdown]
# ## Let's check manually these dataset

# %% [markdown]
# ### 30S70M

# %%
to_read = "30S70M"
ds = xr.open_dataarray(f"training_datasets/TS_{to_read}_{today}.nc")
#ds = ds[list(ds.data_vars)[0]]
all_origin = ds.attrs.get("main_origins", None)

# %%

# %%
# Extract top-level source (e.g., 'fluview' from 'fluview/fluview/...')
prefixes = [entry.split('/')[0] for entry in all_origin]
segments = []
start = 0
current_prefix = prefixes[0]
for i in range(1, len(prefixes)):
    if prefixes[i] != current_prefix:
        segments.append((start, i - 1, current_prefix))
        start = i
        current_prefix = prefixes[i]
# Add the last segment
segments.append((start, len(prefixes) - 1, current_prefix))
# Output: list of (start_idx, end_idx, prefix)
for seg in segments:
    print(f"{seg[2]}: from index {seg[0]} to {seg[1]} (n={seg[1] - seg[0] + 1})")

# %%
idx_tp = 3000
fig, ax = idplots.plot_us_grid(
    data=ds,
    season_axis=season_setup,
    sample_idx=list(np.arange(idx_tp, idx_tp+5)),
    multi_line=True,
    sharey=False,
)

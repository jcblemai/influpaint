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

# InfluPaint modular imports
from influpaint.utils import SeasonAxis
from influpaint.datasets import build_frames
from influpaint.utils import plotting as idplots
from influpaint.utils import converters
from influpaint.datasets import mixer as dataset_mixer
from influpaint.datasets import read_datasources
season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)

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
DATASET_GRIDS = {
    "SURV_ONLY": {
        "fluview":     {"multiplier": 26},
        "flusurv": {"multiplier": 26}
    },
    # 2. Hybrid 70 % surveillance / 30 % modeling 
    "HYBRID_70S_30M": {
        "fluview":     {"proportion": 0.37, "total": 3000},
        "flusurv":     {"proportion": 0.33, "total": 3000},
        "flepiR1":     {"proportion": 0.05, "total": 3000},
        "SMH_R4-R5":   {"proportion": 0.25, "total": 3000}
    },
    # 3. Half-half (uncertainty stress-test)
    "HYBRID_30S_70M": {
        "fluview":     {"proportion": 0.15, "total": 3000},
        "flusurv":     {"proportion": 0.15, "total": 3000},
        "flepiR1":     {"proportion": 0.05, "total": 3000},
        "SMH_R4-R5":   {"proportion": 0.65, "total": 3000}
    },
    "MOD_ONLY": {
        "flepiR1":  {"multiplier": 1},
        "SMH_R4-R5": {"multiplier": 1}
    },
}


# %%

# %%
#mix_cfg = DATASET_GRIDS["SURV_ONLY"]
#frame_list = dataset_mixer.build_frames(all_datasets_df, mix_cfg, season_axis=season_setup, fill_missing_locations="random")



# %%
mix_cfg = DATASET_GRIDS["MOD_ONLY"]
frame_list = dataset_mixer.build_frames(all_datasets_df, mix_cfg, season_axis=season_setup, fill_missing_locations="random")

# %%
for i, frame in enumerate(frame_list):
    df = frame_list[i]
    df["fluseason"] = i
    frame_list[i] = df
    assert df.season_week.max() == 53 and df.season_week.min() == 1, f"Frame {i} has invalid season_week range: {df.season_week.min()} to {df.season_week.max()}"
    assert set(df["location_code"].unique()) == set(season_setup.locations), f"Frame {i} has invalid locations: {set(df['location_code'].unique())} vs {set(season_setup.locations)}"
assert set(pd.concat(frame_list).fluseason.unique()) == set(range(len(frame_list)))

all_frames_df = pd.concat(frame_list).reset_index(drop=True)
array_list = converters.dataframe_to_arraylist(df=all_frames_df, season_setup=season_setup)

# %%
a = frame_list[880]
a[a['location_code'] == '11'].sort_values(by='season_week')

# %%
all_frames_df.iloc[2379069:2379077]

# %%
all_frames_df[(all_frames_df['fluseason'] == 880) & (all_frames_df['location_code'] == '11')].sort_values(by='season_week')

# %%

# Check for duplicates in the index columns before pivoting
dupes = all_frames_df[all_frames_df.duplicated(subset=["fluseason", "season_week", "location_code"], keep=False)]
if not dupes.empty:
    print("Duplicate entries found:")
    print(dupes)
else:
    df_piv = all_frames_df.pivot(
        columns="location_code",
        values="value",
        index=["fluseason", "season_week"],
    )

# %%
a["origin"].unique()

# %%

# %%
mix_cfg = DATASET_GRIDS["HYBRID_70S_30M"]
frame_list = dataset_mixer.build_frames(all_datasets_df, mix_cfg, season_axis=season_setup, fill_missing_locations="random")

# %%

# %%

# %%
# save as an netcdf file
array = np.array(array_list)

flu_payload_array = xr.DataArray(array, 
                coords={'sample': np.arange(array.shape[0]),
                    'feature': np.arange(array.shape[1]),
                    'season_week': np.arange(1, array.shape[2]+1),
                    'place': season_setup.locations + [""]*(array.shape[3] - len(season_setup.locations))}, 
                dims=["sample", "feature", "season_week", "place"])

# ge today's date
import datetime
today = datetime.datetime.now().strftime("%Y-%m-%d")
# create the folder if exists:
Path("training_datasets").mkdir(parents=True, exist_ok=True)


flu_payload_array.to_netcdf(f"training_datasets/NC_Flusight_{today}.nc")



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
from season_axis import SeasonAxis
import read_datasources
import epiweeks
import warnings
import importlib
import tqdm
import dataset_mixer
import idplots
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
mix_cfg = DATASET_GRIDS["SURV_ONLY"]
loader = dataset_mixer.build_frames(all_datasets_df, mix_cfg, season_axis=season_setup, fill_missing_locations="random")

# %%
a = loader[0]
a[a["location_code"] == "22"]

# %%
955	2016-10-08 00:00:00	22	0.0		2					fluview/fluview/2010/1[filled_same_location_year_2016_sample_1]

# %%
a["origin"].unique()

# %%
mix_cfg = DATASET_GRIDS["MOD_ONLY"]
loader = dataset_mixer.build_frames(all_datasets_df, mix_cfg, season_axis=season_setup, fill_missing_locations="random")

# %%
mix_cfg = DATASET_GRIDS["HYBRID_70S_30M"]
loader = dataset_mixer.build_frames(all_datasets_df, mix_cfg, season_axis=season_setup, fill_missing_locations="random")

# %%
a = loader[0]
a #a[a["location_code"] == "03"]

# %%
# The goal is to build a dataset as an arraydf
# (n_samples, n_features, n_dates, n_places)
# the multiplier is used to create multiple datasets from the same data,
# which increases the weight of a particular dataset
dict_of_dfs = {
   # "nc_payload": {"df":nc_payload, "multiplier":1}, 
    "fluview": {"df":fluview_df, "multiplier":30},
    #"flepiR1_df": {"df":flepiR1_df, "multiplier":1}
    }

import dataset_mixer

final_frames, combined_df = dataset_mixer.build_frames(dict_of_dfs)
seasons = sorted(combined_df['fluseason'].unique())
location_codes = combined_df.location_code.unique()
print(f"generated {len(final_frames)} frames from {len(seasons)} seasons and {len(location_codes)} locations in datasets {dict_of_dfs.keys()}")

# %%
for i, frame in enumerate(final_frames):
    df = final_frames[i]
    df["fluseason"] = i
    final_frames[i] = df

assert set(pd.concat(final_frames).fluseason.unique()) == set(range(len(final_frames)))
# TODO: cette function assume que chaque frame commence à la semaine 1. Il faudrait la rendre plus robuste
array_list = read_datasources.dataframe_to_arraylist(df=pd.concat(final_frames), season_setup=season_setup)

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



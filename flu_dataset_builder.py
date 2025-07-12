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
# ## Dataset format for Influpaint
# The dataset is a xarray object stored as netcdf on disk. It has dimensions `(sample, feature, date, place)` where date and place are padded to have dimension 64.
# - dates are Saturdays
# - places are location from Flusight data locations. The sum of all places (whole U.S) is not included at that stage
# - samples are integers
#

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from season_setup import SeasonSetup
import build_dataset
import epiweeks
import warnings
import importlib
import tqdm
import dataset_mixer
season_setup = SeasonSetup.for_flusight(remove_us=True, remove_territories=True)

download=False
if download:
    write=True
else:
    write=False

# %% [markdown]
# ## A. Surveillance Datasets

# %% [markdown]
# ### A.1. Delphi-epidata FluSurv

# %%
flusurv = build_dataset.get_from_epidata(dataset="flusurv",
                                        season_setup=season_setup, 
                                        download=download, 
                                        write=write,
                                        clean=False)
flusurv_clean = build_dataset.clean_dataset(flusurv, season_setup=season_setup)
flusurv_clean = season_setup.add_season_columns(flusurv_clean, do_fluseason_year=False)


# %%
flusurv

# %%
fig, axes = plt.subplots(5, 4, sharex=True, figsize=(15,15))
flusurv_piv  = flusurv_clean.pivot(columns='location_code', values='value', index="week_enddate")
for idx, pl in enumerate(flusurv_piv.columns):
    ax = axes.flat[idx]
    ax.plot(flusurv_piv[pl])
    ax.set_title(season_setup.get_location_name(pl))
    ax.grid()
fig.tight_layout()
fig.autofmt_xdate()

# %%
import seaborn as sns
fig, axes = plt.subplots(5, 4, sharex=True, figsize=(18,15))
flusurv_piv = flusurv_clean.pivot(columns='location_code', values='value', index=["fluseason", "fluseason_fraction"])
for idx, pl in enumerate(flusurv_piv.columns):
    for year in flusurv_piv.index.unique(level='fluseason'):
        ax = axes.flat[idx]
        ax.plot(flusurv_piv.loc[year, pl], c='k', lw=1.2)
        ax.set_title(pl)
    #ax.grid()
    sns.despine(ax=ax)
fig.tight_layout()
fig.autofmt_xdate()

# %% [markdown]
# ### A.2 Read FluView from Delphi Epidata
# There is also [fluview clinical](https://cmu-delphi.github.io/delphi-epidata/api/fluview_clinical.html) for FluA, FluB, and tested specimen. Which quantity should I look for in this dataset ?

# %%
fluview = build_dataset.get_from_epidata(
    dataset="fluview", season_setup=season_setup, download=download, write=write, clean=False
)
fluview_clean = build_dataset.clean_dataset(fluview, season_setup=season_setup)
fluview_clean = season_setup.add_season_columns(fluview_clean, do_fluseason_year=False)
fluview_clean

# %%

# %%
fig, axes = plt.subplots(9, 9, sharex=True, figsize=(15,15))
# remove NA locations
fluview_piv  = fluview[~fluview["region"].isna()].pivot(columns='region', values='ili', index="week_enddate")
for idx, pl in enumerate(fluview_piv.columns):
    ax = axes.flat[idx]
    ax.plot(fluview_piv[pl])
    ax.set_title(pl)#get_location_name(pl))
    ax.grid()
fig.tight_layout()
fig.autofmt_xdate()

# %%
fig, axes = plt.subplots(9, 9, sharex=True, figsize=(15,15))
fluview_piv = fluview[~fluview["region"].isna()].pivot(columns='region', values='ili', index=["fluseason", "fluseason_fraction"])
for idx, pl in enumerate(fluview_piv.columns):
    for year in fluview_piv.index.unique(level='fluseason'):
        ax = axes.flat[idx]
        ax.plot(fluview_piv.loc[year, pl])
        ax.set_title(pl)
    ax.grid()
fig.tight_layout()
fig.autofmt_xdate()

# %% [markdown]
# ### A.3. Check what is on FluSurv and/or Fluview

# %%

for locations_code in season_setup.locations_df.location_code:
    in_fluview, in_flusruv = False, False
    if not flusurv_clean[flusurv_clean['location_code'] == locations_code].empty:
        in_flusruv = True
    if not fluview_clean[fluview_clean['location_code'] == locations_code].empty:
        in_fluview = True
    if in_fluview and in_flusruv:
        suffix = "in both fluview and flusurv"
    elif in_fluview:
        suffix = "in fluview"
    elif in_flusruv:
        suffix = " in flusurv"
    else:
        suffix = "NOT in fluview NOR flusurv"
    print(f"{locations_code}, {season_setup.get_location_name(locations_code):<22} {suffix}")

# %% [markdown]
# ### A.4. FluSurv from Claire P. Smith and FlepiMoP Team
# (min and max from Flu SMH comes from these):
# the numbers come from FluSurvNet -- hosp rates taken and multiplied by population
# ```
# [9:49 AM] Prior Peaks: are based on the minimum and maximum observed in national data in the 8 pre-pandemic seasons 2012-2020, converted to state-level estimates.
# The estimates for incident are based on weekly peaks, and for cumulative based on cumulative hospitalizations at the end of the season.
# ```
#
# So on R you do:
# ```R
# source("~/Documents/phd/COVIDScenarioPipeline/Flu_USA/R/groundtruth_functions.R")
# flus_surv <- get_flu_groundtruth(source="flusurv_hosp", "2015-10-10", "2022-06-11", age_strat = FALSE, daily = FALSE)
# flus_surv %>% write_csv("flu_surv_cspGT.csv")
# ```
#
# This is the only dataset where the scaling is good

# %%
csp_flusurv = pd.read_csv("Flusight/flu-datasets/flu_surv_cspGT.csv", parse_dates=["date"])
csp_flusurv = pd.merge(csp_flusurv, season_setup.locations_df, left_on="FIPS", right_on="abbreviation", how='left')
csp_flusurv["value"] = csp_flusurv["incidH"]
csp_flusurv["week_enddate"] = csp_flusurv["date"]
csp_flusurv = csp_flusurv.drop(columns=["FIPS", "abbreviation", "incidH"])
csp_flusurv = build_dataset.clean_dataset(csp_flusurv, season_setup=season_setup)
csp_flusurv = season_setup.add_season_columns(csp_flusurv, do_fluseason_year=True)
print(f"available for years {csp_flusurv.fluseason.unique()}")

# %%
csp_flusurv

# %%
fig, axes = plt.subplots(7, 8, sharex=True, figsize=(15,15))
csp_flusurv_piv = csp_flusurv.pivot(columns='location_code', values='value', index=["fluseason", "fluseason_fraction"])
for idx, pl in enumerate(csp_flusurv_piv.columns):
    for year in csp_flusurv_piv.index.unique(level='fluseason'):
        ax = axes.flat[idx]
        if len(csp_flusurv_piv.loc[year, pl]) > 0:
            ax.plot(csp_flusurv_piv.loc[year, pl])
        else:
            print(f"Empty data for {pl} and year {year}")
        ax.set_title(season_setup.get_location_name(pl))
    ax.grid()
fig.tight_layout()

# %% [markdown]
# ## B. Modeling Datasets

# %% [markdown]
# ### B.1. Read FluSMH Round 4 and 5
# Prepared using:
# ```bash
# cd Flusight
# git clone https://github.com/midas-network/flu-scenario-modeling-hub_archive.git flu-scenario-modeling-hub_archive-round4
# cd flu-scenario-modeling-hub_archive-round4
# git checkout a67f53fd696ee1b47596ba67b108f6dcba01a1d3
# cd ..
# git clone https://github.com/midas-network/flu-scenario-modeling-hub_archive.git flu-scenario-modeling-hub_archive-round5
# ```
#

# %%
importlib.reload(build_dataset)
smh_traj = build_dataset.extract_flu_scenario_hub_trajectories(min_locations=45)

# %%
smh_traj['round5_ACCIDDA-FlepiMoP']['A-2024-08-01']['sample'].unique()

# %% [markdown]
# ## C. Addition Payload datasets

# %% [markdown]
# ### C.1. Read NC data
# > The hospital admission data are a subset of the ILI data as those admitted will present with ILI in the ED first and then counted again when admitted. I’ve also been told the admission date generally occurs on the same date as the ED visit.
# > ve also added the only PHE-positive test data available. They provide the last 52 weeks on a rolling basis. The historical data is unavailable at this time, and further discussions may be needed to gain access. Again, these data are confirmed (positive test) infections conducted by the hospital-based Public Health Epidemiologist (PHE) program.
#  
# > *Public Health Epidemiologists Program*
# > In 2003, DPH created a hospital-based Public Health Epidemiologist (PHE) program to strengthen coordination and communication between hospitals, health departments and the state. The PHE program covers approximately 38 percent of general/acute care beds and 40 percent of ED visits in the state. PHEs play a critical role in assuring routine and urgent communicable disease control, hospital reporting of communicable diseases, outbreak management and case finding during community wide outbreaks.

# %%
hosp_now = pd.read_csv("custom_datasets/weekly_hosps_2010_24.csv", parse_dates=["Week Date"])
hosp_now["date"] = hosp_now["Week Date"]
hosp_now = hosp_now.set_index("date").drop("Week Date", axis=1)

nc_ed =  pd.read_csv("custom_datasets/weekly_ED_Visits_2010_24.csv", parse_dates=["Week Date"])
nc_ed["date"] = nc_ed["Week Date"]
nc_ed = nc_ed.set_index("date").drop("Week Date", axis=1)

nc_payload = pd.merge(hosp_now, nc_ed, on="date", how="outer", suffixes=("_hosp", "_ed"))


# drop the column whose name contains covid
nc_payload = nc_payload[[col for col in nc_payload.columns if "covid" not in col.lower()]]
#nc_payload.plot(subplots=True)

# tidy the dataframe by putting the data in long format
nc_payload = nc_payload.reset_index().melt(id_vars=["date"], var_name="location_code", value_name="value")
nc_payload = nc_payload.rename(columns={"date":"week_enddate"})

# add the fluseason column and the fluseason_fraction column
nc_payload = season_setup.add_season_columns(nc_payload, season_setup)
# add NC to what is already in the location column
nc_payload["location_code"] = "NC_" + nc_payload["location_code"]

# remoove the 2024 season
nc_payload[["week_enddate","location_code", "value", "fluseason", "fluseason_fraction", "season_week"]].to_csv("custom_datasets/nc_payload_gt.csv", index=False)
nc_payload = nc_payload[nc_payload["fluseason"] != 2024]
#nc_payload = nc_payload[nc_payload["fluseason"] != 2023] # remove the 2023 season because we test on that for the paper

# Filter only ed data for now.
#nc_payload = nc_payload[nc_payload["location_code"].str.contains("ed")]
# rename columns to standardize names
nc_payload = nc_payload.replace({
    'NC_Influenza_hosp': 'NC_flu_hosp',
    'NC_RSV-like Illness_hosp': 'NC_rsv_hosp',
    'NC_Influenza_ed': 'NC_flu_ED',
    'NC_RSV-like Illness_ed': 'NC_rsv_ED'
}, regex=False)
# remove the row where location contains covid
nc_payload = nc_payload[~nc_payload["location_code"].str.contains("covid")]
nc_payload.pivot(index="week_enddate", columns="location_code", values="value").plot(subplots=True)

# from 
nc_payload

# %%
netcdf_file = (
    "Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc"
)
channels = 1
flu_dyn = xr.open_dataarray(netcdf_file)
flu_dyn = flu_dyn.sel(feature="incidH_FluA") + flu_dyn.sel(
    feature="incidH_FluB"
)

smh_df = []
for i, s in enumerate(flu_dyn.sample[:200]):
    df = flu_dyn.sel(sample=s).to_dataframe(name='value').reset_index()
    df["fluseason"] = i
    smh_df.append(df)

smh_df = pd.concat(smh_df)
# remove NaT values from "date" and empty location_code
smh_df = smh_df.dropna(subset=["date"])
# remove empty string from location_code
smh_df = smh_df[smh_df["place"] != ""]

smh_df = smh_df.rename(columns={"place": "location_code", "date": "week_enddate"})
# keep just the first two letter of location_code
smh_df["location_code"] = smh_df["location_code"].apply(lambda x: x[:2])


smh_df = season_setup.add_season_columns(smh_df, season_setup, do_fluseason_year=False)
smh_df

# %%
# The goal is to build a dataset as an array
# (n_samples, n_features, n_dates, n_places)
# the multiplier is used to create multiple datasets from the same data,
# which increases the weight of a particular dataset
dict_of_dfs = {
    "nc_payload": {"df":nc_payload, "multiplier":1}, 
    "fluview": {"df":fluview, "multiplier":30},
    "smh_df": {"df":smh_df, "multiplier":1}
    }

import dataset_mixer

final_frames, combined_df = dataset_mixer.build_frames(dict_of_dfs)
seasons = sorted(combined_df['fluseason'].unique())
location_codes = combined_df.location_code.unique()
print(f"generated {len(final_frames)} frames from {len(seasons)} seasons and {len(location_codes)} locations in datasets {dict_of_dfs.keys()}")

# %%
smh_df.columns

# %%
combined_df

# %%

# %%
new_locations = pd.DataFrame({
    "location_code": sorted(location_codes),
})

# Ensure location_code is of type string
new_locations['location_code'] = new_locations['location_code'].astype(str)

# Merge with season_setup.locations_df to get the location names
new_locations = new_locations.merge(season_setup.locations_df, 
                                    on='location_code',
                                    how='left')

# Fill missing location names with the location code
new_locations['location_name'] = new_locations['location_name'].fillna(new_locations['location_code'])

new_locations = new_locations[['location_code', 'location_name']]
new_locations


# %%
season_setup.update_locations(new_locations)

# %%


# %%
a = pd.concat(final_frames).sort_values(["location_code"])
a[a["season_week"] == 1].shape

# %%
for i, frame in enumerate(final_frames):
    df = final_frames[i]
    df["fluseason"] = i
    final_frames[i] = df

assert set(pd.concat(final_frames).fluseason.unique()) == set(range(len(final_frames)))
# TODO: cette function assume que chaque frame commence à la semaine 1. Il faudrait la rendre plus robuste
array_list = build_dataset.dataframe_to_arraylist(df=pd.concat(final_frames), season_setup=season_setup)

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

# %% [markdown]
# Commands to sync Flu SMH R1 from s3 bucket
# ```bash
# aws s3 sync s3://idd-inference-runs/USA-20220923T154311/model_output/ datasets/SMH_R1/SMH_R1_lowVac_optImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T155228/model_output/ datasets/SMH_R1/SMH_R1_lowVac_pesImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T160106/model_output/ datasets/SMH_R1/SMH_R1_highVac_optImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T161418/model_output/ datasets/SMH_R1/SMH_R1_highVac_pesImm_2022 --exclude "*" --include "hosp*/final/*"
# ```
# and take a humidity file from the config
#

# %%


# %%


# %%
import seaborn as sns
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(5,5))
nc_payload_piv = nc_payload.pivot(columns='location_code', values='value', index=["fluseason", "fluseason_fraction"])
for idx, pl in enumerate(nc_payload_piv.columns):
    for year in nc_payload_piv.index.unique(level='fluseason'):
        ax = axes.flat[idx]
        ax.plot(nc_payload_piv.loc[year, pl], c='k', lw=1.2)
        ax.set_title(pl)
    #ax.grid()
    sns.despine(ax=ax)
fig.tight_layout()
fig.autofmt_xdate()

# %%
build_dataset.dataframe_to_arraylist(nc_payload, season_setup = season_setup, value_column='value')

# %% [markdown]
# ## 1. FluSight dataset (= Ground-truth data for inpainting)

# %% [markdown]
# Make sure my fork is synced, then
# ```bash
# ./update_data.sh
# ```
# Locations are always ordered like this dataframe:

# %%
flusight22 = build_dataset.get_from_epidata(dataset="flusight2022_23", 
                                            season_setup=season_setup, 
                                            write=False,
                                            clean=False)
flusight22

# %%
# Flusight 2023 does not have data for the Virgin Islands (78) so the clean=True flag will remove it.
flusight23 = build_dataset.get_from_epidata(dataset="flusight2023_24", 
                                            season_setup=season_setup, 
                                            write=False,
                                            clean=True)
flusight23

# %%
fig, axes = plt.subplots(9, 6, sharex=True, figsize=(15,15))
flusight22_piv  = flusight22.pivot(index = "week_enddate", columns='location_code', values='value')
flusight23_piv  = flusight23.pivot(index = "week_enddate", columns='location_code', values='value')
for idx, pl in enumerate(flusight22_piv.columns):
    ax = axes.flat[idx]
    ax.plot(flusight22_piv[pl], color='blue', label='2022/23')
    if pl in flusight23_piv.columns:
        ax.plot(flusight23_piv[pl], color='red', label='2023/24', alpha=0.5)
    ax.set_title(season_setup.get_location_name(pl))
    ax.grid()
ax.legend()
fig.tight_layout()
fig.autofmt_xdate()

# %%
gt_xarr = build_dataset.dataframe_to_xarray(df, season_setup=season_setup, 
            xarray_name = "gt_flusight_incidHosp", 
            xarrax_features = "incidHosp")
print(gt_xarr.shape)
gt_xarr.to_netcdf(f"Flusight/flu-datasets/{gt_xarr.name}_padded.nc")

# %% [markdown]
# This data is on saturday for 53 locations + US

# %%


# %%


# %% [markdown]
#

# %% [markdown]
# ## Synthetic dataset from CSP

# %%
assert False  # stop here when "Run All" is used in this notebook
import gempyor
folder = 'datasets/SMH_R1/'
col2keep = ['incidH_FluA', 'incidH_FluB']

# %% [markdown]
# Commands to sync Flu SMH R1 from s3 bucket
# ```bash
# aws s3 sync s3://idd-inference-runs/USA-20220923T154311/model_output/ datasets/SMH_R1/SMH_R1_lowVac_optImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T155228/model_output/ datasets/SMH_R1/SMH_R1_lowVac_pesImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T160106/model_output/ datasets/SMH_R1/SMH_R1_highVac_optImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T161418/model_output/ datasets/SMH_R1/SMH_R1_highVac_pesImm_2022 --exclude "*" --include "hosp*/final/*"
# ```
# and take a humidity file from the config
#

# %%
humid = pd.read_csv('datasets/SMH_R1/SMH_R1_lowVac_optImm_2022/r0s_ts_2022-2023.csv', index_col='date', parse_dates=True)

# %%
maxfiles = -1
hosp_files = list(Path(str(folder)).rglob('*.parquet'))[:maxfiles]
df = gempyor.read_df(str(hosp_files[0]))

# To be pasted later
indexes = df[['date', 'geoid']]
full_df = df[['date', 'geoid']] # to
geoids = list(pd.concat([df[col2keep[0]], indexes], axis=1).pivot(values=col2keep[0], index='date', columns='geoid').columns)
dates = list(pd.concat([df[col2keep[0]], indexes], axis=1).pivot(values=col2keep[0], index='date', columns='geoid').index)

# %%


# %%
df.columns

# %%
incid_xarr = xr.DataArray(-1 * np.ones((len(hosp_files), 
                           len(col2keep),
                           len(full_df.date.unique()),
                           len(full_df.geoid.unique())
                          )), 
                         coords={'sample': np.arange(len(hosp_files)),'feature': col2keep, 'date': dates, 'place': geoids}, 
                         dims=["sample", "feature", "date", "place"])


for i, path_str in enumerate(hosp_files):
    df = gempyor.read_df(str(path_str))
    data = df[col2keep]
    for k, c in enumerate(col2keep):
        incid_xarr.loc[dict(sample=i, feature=c)] = pd.concat([data[c], indexes], axis=1).pivot(values=c, index='date', columns='geoid').to_numpy()
        

    data.columns = [n+f'_{i}' for n in col2keep]   
    full_df = pd.concat([full_df, data], axis=1)
    

print(int((incid_xarr<0).sum()), f' errors on {i} files')

# %%
humid_st = np.dstack([humid.to_numpy()]*len(hosp_files))
#humid_st = humid_st[:, np.newaxis, :]
print(humid_st.shape)
covar_xarr = xr.DataArray(humid_st, 
                          coords={
                                  #'feature': ['R0Humidity'],
                                  'date': humid.index,
                                  'place': geoids,
                                  'sample': np.arange(len(hosp_files)),}, 
                          dims=[ "date", "place", "sample"]) #"feature",
covar_xarr = covar_xarr.expand_dims({"feature":['R0Humidity']})

# %% [markdown]
# ### makes the dates of r0 and humidity match

# %%
print(type(incid_xarr), incid_xarr.date[0], incid_xarr.date[-1] )
print(type(covar_xarr), covar_xarr.date[0], covar_xarr.date[-1])

# %%
full_xarr = xr.concat([incid_xarr,covar_xarr], dim="feature", join="inner")

# %%
grid = (1,4)
fig, axes = plt.subplots(grid[0], grid[1], sharex=True, sharey=True, figsize=(grid[1]*2,grid[0]*2))
for i, ax in enumerate(axes.flat):
    c = ['red', 'green', 'blue']
    place = full_xarr.get_index('place')[i]
    tp = full_xarr.sel(place=place)
    for k, val in enumerate(full_xarr.feature):
        ax.plot(tp.date, tp.sel(feature=val).T, c = c[k], lw = .1, alpha=.5)
        ax.plot(tp.date, tp.sel(feature=val).T.median(axis=1), 
                c = 'k',#'dark'+c[k], 
                lw = .5, 
                alpha=1)
    ax.grid()
    ax.set_title(place)
fig.autofmt_xdate()
fig.tight_layout()

# %%
full_xarr_w = full_xarr.resample(date="W").sum()
full_xarr_w

# %%
full_xarr_w_padded = full_xarr_w.pad({'date': (0, 17), 'place':(0,13)}, mode='constant', constant_values=0)
print(full_xarr_w_padded.shape)
full_xarr_w_padded.to_netcdf("datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc")

# %% [markdown]
# ## Data from the trajectories submitted to SMH

# %%
maxfiles = -1
folder = "Flusight/Flu-SMH/data-processed/"
# Just the last round, submitted on 2023-09-03, contains trajectories
hosp_files = list(Path(str(folder)).rglob('*2023-09-03*.parquet'))[:maxfiles]
print(len(hosp_files), [str(m).split("-")[-1].split(".")[0] for m in hosp_files])

import gempyor

df = gempyor.read_df(str(hosp_files[0]))


# %%
big_df = []
for m in hosp_files:
    df = gempyor.read_df(str(m))
    df["model"] = str(m).split("-")[-1].split(".")[0]
    big_df.append(df)
big_df = pd.concat(big_df)

big_df = big_df[(big_df["output_type"] == "sample") & (big_df["target"] == "inc hosp") &  (big_df["age_group"] == "0-130")].reset_index(drop=True)

# %%
big_df["week_enddate"] = pd.to_datetime(big_df["origin_date"]) + pd.to_timedelta(big_df["horizon"], unit='W')
big_df["location_code"] = big_df["location"]
big_df = big_df.drop(columns=["origin_date", "horizon", "target", "output_type", "age_group", "location"])
big_df

# %%
this_traj

# %%
a = this_df[this_df["output_type_id"] == traj]
a = a.assign(season_fraction=a["week_enddate"].apply(season_setup.get_fluseason_fraction))
a = a.assign(season=a["week_enddate"].apply(season_setup.get_fluseason_year))
a = a.assign(epiweek=a["week_enddate"].apply(lambda x: epiweeks.Week.fromdate(x).week))

# %%
# There are either 52 [1 to 53] epiweeks or 51 epiweeks [1 to 51] in a year
index = [epiweeks.Week.fromdate(d).week for d in season_setup.get_dates()] # has 53 weeks
assert len(index) == 52  # the index year has 52 weeks (1 to 53)

# %%
index

# %%
index

# %%
if max(a.epiweek) < 53:
    # drop the week number 53 in the middle of the INDEX
    index_mod = index.drop(53, axis=0)
else:
    index_mod = index
index_mod

# %%
import pandas as pd
a_piv = a.pivot(columns='location_code', values='value', index="epiweek")
a_piv

# %%
# reindex a_piv with a_piv.index = index, using the closest value and zeros for missing values
a_piv = a_piv.reindex(index, fill_value=0)
a_piv

# %%
array_list = []

gleam = big_df[(big_df["model"]=="GLEAM_FLU")& (big_df["location_code"] == "11")]

for m in big_df.model.unique():
    if len(big_df[(big_df["model"]==m)].location_code.unique()) > 45: # a lot have smaller than 51 locations (just US, just CA), and one has 57 location...
        print(len(big_df[(big_df["model"]==m)].location_code.unique()), m)
        for scn in big_df.scenario_id.unique():
            this_df =  big_df[(big_df["scenario_id"] == scn) & (big_df["model"]==m)]
            for traj in this_df.output_type_id.unique():
                this_traj = this_df[this_df["output_type_id"] == traj].pivot(index="week_enddate", columns='location_code', values='value')
                
                if m == "ImmunoSEIRS": # this model does not have washington DC (11)
                    # replacet it with GLEAM_FLU
                    gleam11 = gleam[(gleam["scenario_id"] == scn) 
                                    & (gleam["output_type_id"] == traj)].pivot(index="week_enddate", columns='location_code', values='value')
                    this_traj["11"] = gleam11
                
                # re-order the locations and keep these in season_setup
                this_traj = this_traj[season_setup.locations]
        
                # order the dates
                this_traj = this_traj.sort_index()
                
                array_list.append(this_traj.to_numpy())

# %%
flu_smh = np.array(array_list)
# add the feature dimention
flu_smh = flu_smh[:, np.newaxis, :]

# %%
flu_smh_xarr = xr.DataArray(flu_smh, 
                coords={'sample': np.arange(flu_smh.shape[0]),
                    'feature': np.arange(flu_smh.shape[1]),
                    'date': this_traj.index,
                    'place': season_setup.locations}, 
                dims=["sample", "feature", "date", "place"])




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
#
# Create first dataframes with:
# * frame identifier:
#   * datasetH1
#   * datasetH2
#   * sample
#   * fluseason
# * Frame axis and value
#   * location_code'
#   * 'season_week'
#   * 'value'
# * available is also week_enddate
#
#

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
from influpaint.datasets import mixer as dataset_mixer
from influpaint.datasets import read_datasources


season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)

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
flusurv_raw = read_datasources.get_from_epidata(dataset="flusurv",
                                        season_setup=season_setup, 
                                        download=download, 
                                        write=write,
                                        clean=False)
flusurv_df = read_datasources.clean_dataset(flusurv_raw, season_setup=season_setup)
flusurv_df = season_setup.add_season_columns(flusurv_df, do_fluseason_year=True)


# %%
fig, axes = idplots.plot_timeseries_grid(flusurv_df, season_setup)

# %%
flusurv_df

# %%
import seaborn as sns
fig, axes = idplots.plot_season_overlap_grid(flusurv_df, season_setup)

# %% [markdown]
# ### A.2 Read FluView from Delphi Epidata
# There is also [fluview clinical](https://cmu-delphi.github.io/delphi-epidata/api/fluview_clinical.html) for FluA, FluB, and tested specimen. Which quantity should I look for in this dataset ?

# %%
fluview_raw = read_datasources.get_from_epidata(
    dataset="fluview", season_setup=season_setup, download=download, write=write, clean=False
)

fluview_raw = fluview_raw[~fluview_raw["region"].isna()]
fluview_df = read_datasources.clean_dataset(fluview_raw, season_setup=season_setup)
fluview_df = season_setup.add_season_columns(fluview_df, do_fluseason_year=True)
fluview_df["datasetH1"] = "fluview"
fluview_df["datasetH2"] = "fluview"
fluview_df["sample"] = 1

# remove NA locations

fig, axes = idplots.plot_timeseries_grid(fluview_raw, season_setup, 
                                    location_col='region', value_col='ili',
                                    title_func=lambda x: x)


# %%
fig, axes = idplots.plot_season_overlap_grid(fluview_df, season_setup,despine=False)

# %%
fluview_df

# %% [markdown]
# ### A.3. Check what is on FluSurv and/or Fluview

# %%
for locations_code in season_setup.locations_df.location_code:
    in_fluview, in_flusruv = False, False
    if not flusurv_df[flusurv_df['location_code'] == locations_code].empty:
        in_flusruv = True
    if not fluview_df[fluview_df['location_code'] == locations_code].empty:
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
csp_flusurv = read_datasources.clean_dataset(csp_flusurv, season_setup=season_setup)
csp_flusurv = season_setup.add_season_columns(csp_flusurv, do_fluseason_year=True)
csp_flusurv["datasetH1"] = "flusurv"
csp_flusurv["datasetH2"] = "csp_flusurv"
csp_flusurv["sample"] = 1
print(f"available for years {csp_flusurv.fluseason.unique()}")

# %%
fig, axes = idplots.plot_season_overlap_grid(csp_flusurv, season_setup, despine=True)

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

# %%
smp

# %%
importlib.reload(read_datasources)
smh_traj = read_datasources.extract_FluSMH_trajectories(min_locations=50)

# remove the one with PSI-M2, as they numbered their samples differently each location and I don't want to deal with that
# at the moment
smh_traj = {k: v for k, v in smh_traj.items() if "PSI-M2" not in k}

# most of them has 100 samples per scenario. Maybe let's pick 20 using random.choice to not degenerate the dataset too much
for model, all_scn in smh_traj.items():
    for scn, df in all_scn.items():
        smp = np.random.choice(df['sample'].unique(), size=20, replace=False)
        smh_traj[model][scn] = df[df['sample'].isin(smp)]


# %%
all_smh_traj = []
for round_model, all_scn in smh_traj.items():
    for scn, scn_df in all_scn.items():
        scn_df["datasetH2"] = f"{round_model}_{scn}"
        all_smh_traj.append(scn_df)
all_smh_traj = pd.concat(all_smh_traj, ignore_index=True)
all_smh_traj["datasetH1"] = "SMH_R4-R5"
all_smh_traj = all_smh_traj.drop(columns=["output_type_id", "run_grouping", "stochastic_run"])

# %%
all_smh_traj

# %%
all_smh_traj = season_setup.add_season_columns(all_smh_traj, do_fluseason_year=True) 

# %% [markdown]
# ### B.2. FlepiMoP/CSP Flu SMH R1 from 2022
# Commands to sync Flu SMH R1 from s3 bucket
# ```bash
# aws s3 sync s3://idd-inference-runs/USA-20220923T154311/model_output/ datasets/SMH_R1/SMH_R1_lowVac_optImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T155228/model_output/ datasets/SMH_R1/SMH_R1_lowVac_pesImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T160106/model_output/ datasets/SMH_R1/SMH_R1_highVac_optImm_2022 --exclude "*" --include "hosp*/final/*"
# aws s3 sync s3://idd-inference-runs/USA-20220923T161418/model_output/ datasets/SMH_R1/SMH_R1_highVac_pesImm_2022 --exclude "*" --include "hosp*/final/*"
# ```
# and take a humidity file from the config

# %%
if False:  # stop here when "Run All" is used in this notebook
    import gempyor
    folder = 'datasets/SMH_R1/'
    col2keep = ['incidH_FluA', 'incidH_FluB']
    humid = pd.read_csv('datasets/SMH_R1/SMH_R1_lowVac_optImm_2022/r0s_ts_2022-2023.csv', index_col='date', parse_dates=True)

    maxfiles = -1
    hosp_files = list(Path(str(folder)).rglob('*.parquet'))[:maxfiles]
    df = gempyor.read_df(str(hosp_files[0]))

    # To be pasted later
    indexes = df[['date', 'geoid']]
    full_df = df[['date', 'geoid']] # to
    geoids = list(pd.concat([df[col2keep[0]], indexes], axis=1).pivot(values=col2keep[0], index='date', columns='geoid').columns)
    dates = list(pd.concat([df[col2keep[0]], indexes], axis=1).pivot(values=col2keep[0], index='date', columns='geoid').index)


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

    #### makes the dates of r0 and humidity match
    print(type(incid_xarr), incid_xarr.date[0], incid_xarr.date[-1] )
    print(type(covar_xarr), covar_xarr.date[0], covar_xarr.date[-1])
    full_xarr = xr.concat([incid_xarr,covar_xarr], dim="feature", join="inner")
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


    full_xarr_w = full_xarr.resample(date="W").sum()

    full_xarr_w_padded = full_xarr_w.pad({'date': (0, 17), 'place':(0,13)}, mode='constant', constant_values=0)
    print(full_xarr_w_padded.shape)
    full_xarr_w_padded.to_netcdf("datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc")



# %%
full_xarr_w_padded = xr.open_dataarray("Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc")
flepiR1_df = (full_xarr_w_padded
        .to_dataframe(name='value')
        .reset_index()
        .pivot_table(index=['sample', 'date', 'place'], 
                columns='feature', 
                values='value')
        .reset_index())
flepiR1_df["value"] = flepiR1_df["incidH_FluA"] + flepiR1_df["incidH_FluB"] 
flepiR1_df = flepiR1_df.rename(columns={"date": "week_enddate", "place": "location_code"}).drop(columns=["incidH_FluA", "incidH_FluB", "R0Humidity"])

flepiR1_df = flepiR1_df.dropna(subset=["week_enddate"])
# remove empty string from location_code
flepiR1_df = flepiR1_df[flepiR1_df["location_code"] != ""]
flepiR1_df["location_code"] = flepiR1_df["location_code"].apply(lambda x: x[:2])
flepiR1_df["datasetH1"] = "flepiR1"
flepiR1_df["datasetH2"] = "flepiR1"
flepiR1_df = season_setup.add_season_columns(flepiR1_df, do_fluseason_year=True)

# also sample 20*4 scn random sample
smp = np.random.choice(flepiR1_df['sample'].unique(), size=80, replace=False)
flepiR1_df = flepiR1_df[flepiR1_df['sample'].isin(smp)]


# %% [markdown]
# ## C. Generate dataset for fitting

# %%
sorted(fluview_df[fluview_df["fluseason"] == 2010]["location_code"].unique())

# %%
all_datasets = {"fluview": fluview_df, 
                "csp_flusurv": csp_flusurv, 
                "flepiR1": flepiR1_df, 
                "smh_traj": all_smh_traj}
for source, df in all_datasets.items():
    print(f"Source: {source}, shape: {df.shape} > years: {len(df['fluseason'].unique())}, datasetH2: {len(df['datasetH2'].unique())}, sample: {len(df['sample'].unique())}")

all_datasets_df = pd.concat(all_datasets.values(), ignore_index=True)
print(f"All datasets combined shape: {all_datasets_df.shape}")
print(sorted(all_datasets_df.columns))

# %%
all_datasets_df

# %%
all_datasets_df['sample'] = all_datasets_df['sample'].astype(str)
all_datasets_df.to_parquet("Flusight/flu-datasets/all_datasets.parquet", index=False)

# %%
assert False, "This is a temporary assert to stop the notebook from running further. Remove it when you want to run the rest of the notebook."

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
# The goal is to build a dataset as an array
# (n_samples, n_features, n_dates, n_places)
# the multiplier is used to create multiple datasets from the same data,
# which increases the weight of a particular dataset
dict_of_dfs = {
    "nc_payload": {"df":nc_payload, "multiplier":1}, 
    "fluview": {"df":fluview, "multiplier":30},
    "flepiR1_df": {"df":flepiR1_df, "multiplier":1}
    }

# dataset_mixer already imported above as: from influpaint.datasets import mixer as dataset_mixer

final_frames, combined_df = dataset_mixer.build_frames(dict_of_dfs)
seasons = sorted(combined_df['fluseason'].unique())
location_codes = combined_df.location_code.unique()
print(f"generated {len(final_frames)} frames from {len(seasons)} seasons and {len(location_codes)} locations in datasets {dict_of_dfs.keys()}")

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

# %%
import seaborn as sns
fig, axes = idplots.plot_season_overlap_grid(nc_payload, season_setup,
                                           title_func=lambda x: x)

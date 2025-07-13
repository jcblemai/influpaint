import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
from season_axis import SeasonAxis
import xarray as xr

def padto64x64(x: np.ndarray) -> np.ndarray:
    return np.pad(
        x,
        ((0, 64 - x.shape[0]), (0, 64 - x.shape[1])),
        mode="constant",
        constant_values=0,
    )

def extract_FluSMH_trajectories(base_path="/Users/chadi/Research/influpaint/Flusight", 
                                        target="inc hosp", 
                                        age_group="0-130",
                                        min_locations=10,
                                        season_setup=None):
    """
    Extract trajectories from flu scenario modeling hub archive data.
    
    Parameters:
    -----------
    base_path : str
        Base path to the Flusight directory
    target : str
        Target to extract (e.g., "inc hosp", "inc death")
    age_group : str
        Age group to extract (default: "0-130")
    min_locations : int
        Minimum number of locations required (to filter out state-only models)
        
    Returns:
    --------
    dict
        Dictionary with keys like "round4_CADPH-FluCAT" and "round5_CADPH-FluCAT"
        Each value is a dict with scenario_id keys containing arrays of trajectories
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    trajectory_data = {}
    
    # Process both rounds
    for round_num in [4, 5]:
        round_path = Path(base_path) / f"flu-scenario-modeling-hub_archive-round{round_num}" / "data-processed"
        
        if not round_path.exists():
            print(f"Warning: Round {round_num} path does not exist: {round_path}")
            continue
            
        print(f"\n=== Processing Round {round_num} ===")
        
        # Find all team-model directories
        team_model_dirs = [d for d in round_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for team_model_dir in team_model_dirs:
            team_model_name = team_model_dir.name
            
            # Skip example models and non-team directories
            if team_model_name in ['MyTeam-MyModel']:
                continue
                
            print(f"Processing {team_model_name}...", end=" ")
            
            # Find parquet files in this directory
            parquet_files = list(team_model_dir.glob("*.parquet")) + list(team_model_dir.glob("*.gz.parquet"))
            
            if not parquet_files:
                print(f"❌ No parquet files found")
                continue
            
            # Try to find a file with trajectories (sample output type)
            trajectory_file = None
            for parquet_file in parquet_files:
                try:
                    # Quick check if this file has trajectories
                    df_sample = pd.read_parquet(parquet_file, columns=['output_type'])
                    if 'sample' in df_sample['output_type'].values:
                        trajectory_file = parquet_file
                        break
                except Exception:
                    continue
                    
            if trajectory_file is None:
                print(f"❌ No trajectory data found")
                continue
            
            try:
                # Read the parquet file
                df = pd.read_parquet(trajectory_file)
                
                # Filter for trajectories (sample output type)
                trajectory_df = df[
                    (df['output_type'] == 'sample') & 
                    (df['target'] == target) & 
                    (df['age_group'] == age_group)
                ].copy()
                
                if trajectory_df.empty:
                    print(f"❌ No trajectories found for target='{target}', age_group='{age_group}'")
                    continue
                
                # Check if model covers enough locations (filter out state-only models)
                unique_locations = trajectory_df['location'].unique()
                if len(unique_locations) < min_locations:
                    print(f"❌ Only {len(unique_locations)} locations (need ≥{min_locations})")
                    continue
                
                # Create composite trajectory ID from available columns
                trajectory_id_parts = []
                
                # Add run_grouping if available
                if 'run_grouping' in trajectory_df.columns:
                    trajectory_id_parts.append(trajectory_df['run_grouping'].fillna('NA').astype(str))
                
                # Add output_type_id if available and not all null
                if 'output_type_id' in trajectory_df.columns and trajectory_df['output_type_id'].notna().any():
                    trajectory_id_parts.append(trajectory_df['output_type_id'].fillna('NA').astype(str))
                elif 'output_type_id' in trajectory_df.columns:
                    # If output_type_id exists but is all null, add 'NA'
                    trajectory_id_parts.append('NA')
                
                # Add stochastic_run if available
                if 'stochastic_run' in trajectory_df.columns:
                    trajectory_id_parts.append(trajectory_df['stochastic_run'].fillna('NA').astype(str))
                
                # Create composite trajectory ID
                if trajectory_id_parts:
                    trajectory_df['trajectory_id'] = trajectory_id_parts[0]
                    for part in trajectory_id_parts[1:]:
                        trajectory_df['trajectory_id'] = trajectory_df['trajectory_id'] + '_' + part
                    trajectory_id_col = 'trajectory_id'
                else:
                    print(f"❌ No valid trajectory ID columns found")
                    continue
                
                # Convert trajectories to DataFrame format by scenario (optimized)
                scenario_dfs = {}
                
                for scenario_id in trajectory_df['scenario_id'].unique():
                    scenario_data = trajectory_df[trajectory_df['scenario_id'] == scenario_id].copy()
                    
                    # Vectorized processing - much faster than looping
                    # Create week_enddate from origin_date + horizon
                    scenario_data['week_enddate'] = pd.to_datetime(scenario_data['origin_date']) + pd.to_timedelta(scenario_data['horizon'], unit='W')
                    
                    # Add sample identifier (use the trajectory ID as sample)
                    scenario_data['sample'] = scenario_data[trajectory_id_col].astype(str)
                    
                    # Rename location column to match expected format
                    scenario_data = scenario_data.rename(columns={'location': 'location_code'})
                    
                    # Select needed columns including trajectory ID components
                    columns_to_keep = ['week_enddate', 'location_code', 'sample', 'value']
                    
                    # Add trajectory ID components if they exist
                    if 'run_grouping' in scenario_data.columns:
                        columns_to_keep.append('run_grouping')
                    if 'output_type_id' in scenario_data.columns:
                        columns_to_keep.append('output_type_id')
                    if 'stochastic_run' in scenario_data.columns:
                        columns_to_keep.append('stochastic_run')
                    
                    scenario_df = scenario_data[columns_to_keep].copy()
                    
                    # Add season columns using season_setup if provided
                    if season_setup is not None:
                        from season_axis import add_season_columns
                        scenario_df = add_season_columns(scenario_df, season_setup, do_fluseason_year=False)
                    
                    scenario_dfs[scenario_id] = scenario_df
                    
                    n_trajectories = scenario_df['sample'].nunique()
                    n_timepoints = scenario_df.groupby('sample')['week_enddate'].nunique().mean()
                
                # Store scenario DataFrames
                if scenario_dfs:
                    key = f"round{round_num}_{team_model_name}"
                    trajectory_data[key] = scenario_dfs
                    
                    total_trajectories = sum(df['sample'].nunique() for df in scenario_dfs.values())
                    print(f"✅ {total_trajectories} trajectories across {len(scenario_dfs)} scenarios")
                else:
                    print(f"❌ No valid trajectories extracted")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed {len(trajectory_data)} model-round combinations:")
    for key in sorted(trajectory_data.keys()):
        scenario_dfs = trajectory_data[key]
        total_trajectories = sum(df['sample'].nunique() for df in scenario_dfs.values())
        
        # Get unique locations and dates from first scenario (should be same across scenarios)
        first_scenario_df = next(iter(scenario_dfs.values()))
        n_locations = first_scenario_df['location_code'].nunique()
        n_dates = first_scenario_df['week_enddate'].nunique()
        
        print(f"  {key}: {len(scenario_dfs)} scenarios, {total_trajectories} trajectories, {n_locations} locations, {n_dates} dates")
    
    return trajectory_data


def dataframe_to_xarray(
    df: pd.DataFrame,
    season_setup: SeasonAxis = None,
    xarray_name="data",
    xarrax_features="value",
    date_column="week_enddate",
    value_column="value",
    pad=True,
) -> xr.DataArray:
    """
    Convert a long form dataframe to an xarray. Dataframe must have columns:
    - location_code
    - value
    The dataset is a xarray object stored as netcdf on disk. 
    It has dimensions `(feature, date, place)` 
    where date and place are padded to have dimension 64.
    - dates are Saturdays
    - places are location from Flusight data locations
    - samples are integers
    """

    df_piv = df.reset_index(drop=False).pivot(columns="location_code", values=value_column, index=date_column)

    if not isinstance(xarrax_features, list):
        xarrax_features = [xarrax_features]

    if season_setup is None:
        print(" ⚠️ No season_setup provided, using all locations in the dataframe.")
        places = df_piv.columns.to_list()
    else:
        df_piv = df_piv[
            season_setup.locations_df["location_code"]
        ]  # make sure order is right w.r.t flusight_locations
        places = season_setup.locations_df["location_code"]
        df_piv = df_piv.sort_index(axis=1)

    df_xarr = xr.DataArray(
        np.array([df_piv.to_numpy()]),
        name=xarray_name,
        coords={
            "feature": xarrax_features,
            "date": list(df_piv.index),
            "place": places,
        },
        dims=["feature", "date", "place"],
    )
    if pad:
        df_xarr = df_xarr.pad(
            {
                "date": (0, 64 - len(df_xarr.date)),
                "place": (0, 64 - len(df_xarr.place)),
            },
            mode="constant",
            constant_values=0,
        )

    return df_xarr


def dataframe_to_arraylist(
    df: pd.DataFrame, season_setup: SeasonAxis = None, value_column="value",
) -> np.ndarray:

    samples = []

    df_piv = df.pivot(
        columns="location_code",
        values=value_column,
        index=["fluseason", "season_week"],
    )
    for season in df_piv.index.unique(level="fluseason"):
        array = df_piv.loc[season][
            season_setup.locations
        ].sort_index().to_numpy()  # make sure order is right w.r.t season_setup locations and the time is right
        # TODO: should give an error when dates are missing because it would be missaligned

        array[np.isnan(array)] = 0  # replace NaNs with 0

        samples.append(
            np.array([padto64x64(array)])
        )  # pad to 64x64 and add a dimension for channel

    return samples


def get_from_epidata(
    dataset,
    season_setup: SeasonAxis = None,
    locations="all",
    value_col=None,
    write=True,
    download=True,
    clean = True
):
    """ 
    Read a dataset from epidata. Each dataset is a dataframe with columns:
    - 'week_enddate' (datetime)  the date of the saturday at the end of the week
    - 'location_code' (str) location name in the format used by the flusight data
    - 'value' (float) the value of interest
    - 'fluseason' (int) the flu season (e.g. 2019)
    - 'fluseason_fraction' (float) the fraction of the flu season (e.g. 0.5 for the middle of the season)
    """

    if dataset == "flusurv" or dataset == "fluview":
        if download:
            import epiweeks
            # by location otherwise queries is too big
            df_list = []
            if locations == "all":
                locations = get_dataset_all_locations(dataset=dataset)

            for location in locations:
                if dataset == "flusurv":
                    res = Epidata.flusurv(
                        location, [Epidata.range(190001, 202451)]
                    )  # large range to get all data
                elif dataset == "fluview":
                    res = Epidata.fluview(
                        location, [Epidata.range(190001, 202451)]
                    )  # large range to get all data
                if res["result"] == 1:
                    flu_data_loc = pd.json_normalize(res["epidata"])
                    print(
                        f">> {location: <12} {res['result']}, {res['message']}, with {len(res['epidata']):4} data points from {flu_data_loc.epiweek.min()} to {flu_data_loc.epiweek.max()}"
                    )
                    df_list.append(flu_data_loc)
                else:
                    print(f"EE {location: <12} {res['result']}, {res['message']} !")

            df = pd.concat(df_list)
            df["week_enddate"] = (
                df["epiweek"]
                .astype(str)
                .apply(
                    lambda x: epiweeks.Week.fromstring(
                        week_string=x, system="cdc"
                    ).enddate()
                )
            )
        else:
            df = pd.read_csv(f"Flusight/flu-datasets/{dataset}.csv")
    elif dataset == "flusight2022":
        df = pd.read_csv(
            "Flusight/2022-2023/FluSight-forecast-hub-official/data-truth/truth-Incident Hospitalizations.csv",
            parse_dates=True,
            index_col="date",
        )
        df["week_enddate"] = df.index
    elif dataset == "flusight2023":
        df = pd.read_csv(
            "Flusight/2023-2024/FluSight-forecast-hub-official/target-data/target-hospital-admissions.csv",
            parse_dates=True,
            index_col="date",
        )
        df["week_enddate"] = df.index
    elif dataset == "flusight2024":
        df = pd.read_csv(
            "Flusight/2024-2025/FluSight-forecast-hub-official/target-data/target-hospital-admissions.csv",
            parse_dates=True,
            index_col="date",
        )
        df["week_enddate"] = df.index
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    df["week_enddate"] = pd.to_datetime(df["week_enddate"])

    if write:  # write before merge
        df.to_csv(f"Flusight/flu-datasets/{dataset}.csv", index=False)

    if season_setup is not None:
        # merge with locations, taking care of new york
        if dataset == "flusurv":
            df["location_tomerge"] = df["location"]
            df["location_tomerge"] = df["location_tomerge"].str.replace(
                "NY_albany", "NY"
            )
            df["location_tomerge"] = df["location_tomerge"].str.replace(
                "NY_rochester", "NY"
            )
            # sum the values for the different regions of NY, selecting only new york:
            df_ny = df[df["location_tomerge"] == "NY"]
            df_ny = df_ny.groupby(["week_enddate", "location_tomerge"]).sum(numeric_only=True).reset_index()
            df = df[df["location_tomerge"] != "NY"]
            df = pd.concat([df, df_ny])
            print(" >> summing NY_albany and NY_rochester into NY")
            right_on = "abbreviation"
        
        elif dataset == "fluview":
            df["location_tomerge"] = df["region"].str.upper()
            df["location_tomerge"] = df["location_tomerge"].str.replace(
                "jfk".upper(), "NY"
            )
            df["location_tomerge"] = df["location_tomerge"].str.replace(
                "ny_minus_jfk".upper(), "NY"
            )
            right_on = "abbreviation"
            


        elif "flusight" in dataset:
            print(
                "⚠️ ⚠️ ⚠️ If during season, make sure ./update_data.sh has been run"
            )
            df["location_tomerge"] = df["location"]
            df = df.drop(columns=["location_name"])
            right_on = "location_code"

        df = pd.merge(
            df,
            season_setup.locations_df[["location_code", "location_name", "abbreviation"]],
            left_on="location_tomerge",
            right_on=right_on,
            how="outer",

        )
        df.drop(columns=["location_tomerge"], inplace=True)

    if value_col is None:
        if "flusight" in dataset:
            value_col = "value"
        elif dataset == "flusurv":
            value_col = "rate_overall"
        elif dataset == "fluview":
            value_col = "ili"

    df["value"] = df[value_col]

    print(f"RAW Dataset {dataset} has {len(df)} data points, with {len(df['location_code'].unique())} locations,"
            f"and NA values: {df['value'].isna().sum()}, NA locations: {df['location_code'].isna().sum()}")
    # select only the columns we need
    if clean:
        # remove 
        df = clean_dataset(df, season_setup)

    return df


def clean_dataset(df, season_setup):
    df = df[["week_enddate", "location_code", "value"]]
    # remove locations that are not in the season_setup
    df = df[df["location_code"].isin(season_setup.locations)]
    # remove NaNs
    df = df.dropna(subset=["value"])
    print(f" >>> after clean: Dataset {len(df)} data points, with {len(df['location_code'].unique())} locations,"
            f"and NA values: {df['value'].isna().sum()}, NA locations: {df['location_code'].isna().sum()}")
    return df


def get_dataset_all_locations(dataset):
    if dataset == "flusurv":
        locations_fn = (
            "Flusight/flu-datasets/delphi-epidata/labels/flusurv_locations.txt"
        )
        locations = pd.read_csv(
            locations_fn, sep="\t", header=None, names=["location"]
        )["location"].to_list()
        return locations
    elif dataset == "fluview":
        import importlib

        fluview_locations_m = importlib.import_module(
            "Flusight.flu-datasets.delphi-epidata.src.acquisition.fluview.fluview_locations"
        )
        fll_dict = fluview_locations_m.cdc_to_delphi
        locations = []
        for region_type in fll_dict.keys():
            for region_name, flloc in fll_dict[region_type].items():
                locations.append(flloc)
        return locations
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented for getting all locations")
    
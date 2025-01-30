import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
from season_setup import SeasonSetup
import xarray as xr
import tqdm
import epiweeks


def add_season_columns(df, season_setup, do_fluseason_year=True):
    df = df.assign(fluseason_fraction=df["week_enddate"].apply(season_setup.get_fluseason_fraction))
    df = df.assign(season_week=df["week_enddate"].apply(season_setup.get_fluseason_week))
    df = df.assign(epiweek=df["week_enddate"].apply(lambda x: epiweeks.Week.fromdate(x).week))
    
    if do_fluseason_year:
        df = df.assign(fluseason=df["week_enddate"].apply(season_setup.get_fluseason_year))
        
    
    return df

def merge_datasets(dict_of_dfs: dict) -> pd.DataFrame:
    """
    Merges multiple epidemic datasets into a single dataframe, with support for dataset replication.
    
    Takes a dictionary of datasets where each dataset can be replicated multiple times. Each replication
    creates a new virtual "season" by offsetting the fluseason value. This is useful for augmenting
    small datasets or combining data from multiple sources.
    
    Args:
        dict_of_dfs (dict): Dictionary where:
            - keys are dataset names (str)
            - values are dictionaries containing:
                - 'df' (pd.DataFrame): DataFrame with required columns:
                    * season_week (int): Week number within the flu season
                    * location_code (str): Geographic location identifier
                    * value (float): Epidemic metric value
                    * fluseason (int): Season identifier
                    * week_enddate (datetime): End date of the week
                - 'multiplier' (int): Number of times to replicate this dataset
    
    Returns:
        pd.DataFrame: Combined dataset with additional columns:
            - dataset_name (str): Original dataset name with replica number
            - fluseason (int): Original season offset by 10000 for each replica
            
    Raises:
        ValueError: If any required columns are missing from input DataFrames
    
    Example:
        datasets = {
            'flu_2020': {
                'df': pd.DataFrame(...),
                'multiplier': 2
            },
            'flu_2021': {
                'df': pd.DataFrame(...),
                'multiplier': 1
            }
        }
        merged_df = merge_datasets(datasets)
    """
    combined_datasets = []
    for dataset_name, dataset_info in dict_of_dfs.items():
        df = dataset_info['df'].copy()
        multiplier = dataset_info['multiplier']
        
        # Validate required columns
        required_columns = ['season_week', 'location_code', 'value', 'fluseason', 'week_enddate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset '{dataset_name}': {missing_columns}")
            
        df = df[required_columns]
        
        # Create replicas with offset seasons
        for i in range(multiplier):
            df_copy = df.copy()
            df_copy.loc[:, "dataset_name"] = f"{dataset_name}_{i}"
            df_copy.loc[:, "fluseason"] = df_copy.loc[:, "fluseason"] + i * 1000 # Offset season by 1000, so they don't overlap
            combined_datasets.append(df_copy)

    return pd.concat(combined_datasets, ignore_index=True)

def pad_single_frame(frame: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Pads a location-specific epidemic time series to ensure complete weekly coverage.
        
    For a given location's time series, fills in missing weeks using the following rules:
    - Weeks before the first observation or after the last observation are filled with zeros
    - Weeks between existing observations are filled using the previous week's value
        
    Args:
        frame (pd.DataFrame): DataFrame containing epidemic data for a single location
        location (str): Location identifier for the current frame
    
    Returns:
        pd.DataFrame: Padded DataFrame with entries for all weeks 1-53
            
    Note:
        The function assumes weeks should range from 1 to 53. Missing weeks are identified
        and filled according to their position relative to existing data.
    """
    # Handle empty input
    if frame.empty:
        frame = pd.DataFrame({
            "season_week": [],
            "location_code": [],
            "value": []
        })
        
    # Get min/max weeks if data exists
    min_week = frame["season_week"].min() if not frame.empty else 53
    max_week = frame["season_week"].max() if not frame.empty else 0
        
    all_weeks = set(range(1, 54))
    missing_weeks = sorted(list(all_weeks - set(frame["season_week"])))
        
    # Fill in missing weeks
    for week in missing_weeks:
        # Determine fill value based on position
        if week <= min_week or week >= max_week:
            new_value = 0  # External gaps filled with zeros
        else:
            # Internal gaps filled with previous week's value
            previous_week = frame[frame["season_week"] == week-1]
            new_value = previous_week["value"].values[0] if not previous_week.empty else 0

        # Create new row for missing week
        new_frame = pd.DataFrame({
            "season_week": [week],
            "location_code": [location],
            "value": [new_value]
        })
        frame = pd.concat([frame, new_frame])

    return frame.sort_values("season_week").reset_index(drop=True)

def build_frames(dict_of_dfs: dict) -> list:
    """
    Constructs a list of complete epidemic frames from multiple datasets.
    
    Creates a comprehensive set of epidemic frames by:
    1. Merging input datasets with their specified multipliers
    2. Processing each season and location combination
    3. Handling missing data by borrowing from other seasons
    4. Ensuring complete weekly coverage through padding
    
    Args:
        dict_of_dfs (dict): Dictionary of datasets and their multipliers.
            Same format as required by merge_datasets()
    
    Returns:
        list: List of DataFrames, where each DataFrame contains:
            - Complete weekly data (weeks 1-53)
            - All specified locations
            - Values either from original data or appropriately filled
            
    Note:
        When a location-season combination has no data, the function borrows data
        from a random season for that location. This ensures all location-season
        combinations have complete coverage while maintaining realistic patterns.
    """
    # Merge all datasets according to their multipliers
    combined_df = merge_datasets(dict_of_dfs)
    
    # Get unique seasons and locations
    seasons = sorted(combined_df['fluseason'].unique())
    location_codes = combined_df.location_code.unique()
    
    all_frames = []
    for season in tqdm.tqdm(seasons, desc="Building frames", total=len(seasons)):
        season_df = combined_df[combined_df['fluseason'] == season]
        
        # Handle multiple datasets per location in this season
        n_repeat_location_in_season = season_df.groupby("location_code")["dataset_name"].nunique()
        
        for i in range(n_repeat_location_in_season.max()):
            new_frames = []
            
            for location in location_codes:
                # Get data for this location in this season
                frame = season_df[season_df["location_code"] == location].sort_values('season_week')
                
                # Handle missing location-season combinations
                if len(frame) == 0:
                    # Borrow data from another season for this location
                    alternative_frames = combined_df[combined_df['location_code'] == location]
                    alternative_season = np.random.choice(alternative_frames.fluseason.unique())
                    frame = combined_df[
                        (combined_df['fluseason'] == alternative_season) & 
                        (combined_df['location_code'] == location)
                    ]
                
                # Handle multiple datasets for this location
                n_datasets = frame["dataset_name"].nunique()
                if n_datasets > 1:
                    dataset_names = sorted(frame["dataset_name"].unique())
                    frame = frame[frame["dataset_name"] == dataset_names[i % n_datasets]]
                
                # Ensure complete weekly coverage
                frame = pad_single_frame(frame, location)
                new_frames.append(frame)
            
            all_frames.append(pd.concat(new_frames))
            
    return all_frames, combined_df

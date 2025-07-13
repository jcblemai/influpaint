"""
dataset_mixer.py - Epidemic Data Augmentation and Frame Construction

This module provides functionality for combining multiple epidemic surveillance datasets
into a unified training corpus for diffusion models. It implements a sophisticated 
data augmentation strategy that addresses common challenges in epidemic modeling:

- **Dataset Rebalancing**: Uses multipliers to weight high-quality data sources
- **Temporal Completeness**: Ensures all frames have complete weekly coverage (1-53)
- **Spatial Completeness**: Fills missing location-season combinations
- **Gap Handling**: Intelligent filling of missing weeks and locations

Key Components:
--------------
1. **Multiplier Calculation**: Compute dataset weights for target proportions
2. **Data Merging**: Replicate datasets according to multipliers
3. **Frame Construction**: Build complete epidemic frames for training
4. **Gap Filling**: Handle missing data with epidemiologically-aware strategies

Typical Usage:
--------------
# Step 1: Define target proportions
dict_of_dfs = {
    "fluview": {"df": fluview_df},
    "nc_payload": {"df": nc_df}, 
    "synthetic": {"df": model_df}
}

# Step 2: Calculate multipliers for desired composition
multipliers = calculate_multipliers(dict_of_dfs, total_samples=1000, 
                                  target_proportions={"fluview": 0.7})

# Step 3: Apply multipliers and build training frames
for name in dict_of_dfs:
    dict_of_dfs[name]["multiplier"] = multipliers[name]
    
final_frames, combined_df = build_frames(dict_of_dfs)

Output Format:
--------------
Each frame is a complete epidemic season with:
- All weeks (1-53) represented
- All locations covered  
- Consistent data structure for array conversion
- Ready for diffusion model training

This design enables robust training on heterogeneous surveillance data while
maintaining epidemiological realism and temporal structure.
"""

import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
from season_axis import SeasonAxis
import xarray as xr
import tqdm


def calculate_multipliers(dict_of_dfs: dict, total_samples: int, 
                         target_proportions: dict) -> dict:
    """
    Calculate dataset multipliers to achieve target proportions in final training corpus.
    
    Given multiple datasets and desired composition (e.g., "70% fluview data"), this function
    computes multipliers that will produce the target distribution when datasets are replicated.
    
    Args:
        dict_of_dfs (dict): Dictionary of datasets, each containing:
            - 'df' (pd.DataFrame): The actual dataset
            Expected columns: ['fluseason', 'location_code', 'season_week', 'value']
        total_samples (int): Desired total number of samples in final corpus
        target_proportions (dict): Desired proportions for specific datasets.
            - Keys: dataset names (must exist in dict_of_dfs)
            - Values: proportion (0.0-1.0), e.g., {"fluview": 0.7}
            - Unspecified datasets get equal share of remaining proportion
    
    Returns:
        dict: Multipliers for each dataset, keyed by dataset name
        
    Example:
        datasets = {
            "fluview": {"df": fluview_df},      # has 10 base samples
            "nc_data": {"df": nc_df},           # has 5 base samples
            "synthetic": {"df": synth_df}       # has 8 base samples
        }
        
        # Want 70% fluview, 1000 total samples
        multipliers = calculate_multipliers(
            datasets, 
            total_samples=1000,
            target_proportions={"fluview": 0.7}
        )
        # Result: {"fluview": 70, "nc_data": 30, "synthetic": 19}
        # This gives: 70*10=700 fluview + 30*5=150 nc + 19*8=152 synth ≈ 1000 total
        
    Notes:
        - Base sample count = number of unique (fluseason, location_code) combinations
        - Multipliers are rounded to integers
        - Final total may differ slightly from target due to rounding
        - At least multiplier=1 is guaranteed for all datasets
    """
    # Calculate base sample counts for each dataset
    base_counts = {}
    for name, dataset_info in dict_of_dfs.items():
        df = dataset_info['df']
        required_columns = ['fluseason', 'location_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Dataset '{name}' missing required columns: {missing_columns}")
        
        base_count = df.groupby(['fluseason', 'location_code']).ngroups
        base_counts[name] = base_count
    
    # Initialize multipliers - everyone gets at least 1
    multipliers = {name: 1 for name in dict_of_dfs.keys()}
    
    # Calculate target sample counts for specified datasets
    specified_names = set(target_proportions.keys())
    unspecified_names = set(dict_of_dfs.keys()) - specified_names
    
    # Validate that specified datasets exist
    missing_datasets = specified_names - set(dict_of_dfs.keys())
    if missing_datasets:
        raise ValueError(f"Target proportions specified for non-existent datasets: {missing_datasets}")
    
    # Validate that proportions sum to <= 1.0
    total_specified_proportion = sum(target_proportions.values())
    if total_specified_proportion > 1.0:
        raise ValueError(f"Target proportions sum to {total_specified_proportion:.3f} > 1.0")
    
    # Calculate multipliers for specified datasets
    for name, proportion in target_proportions.items():
        target_samples = int(total_samples * proportion)
        base_count = base_counts[name]
        multipliers[name] = max(1, round(target_samples / base_count))
    
    # Calculate remaining proportion for unspecified datasets
    remaining_proportion = 1.0 - total_specified_proportion
    if remaining_proportion > 0 and unspecified_names:
        remaining_samples = int(total_samples * remaining_proportion)
        samples_per_unspecified = remaining_samples / len(unspecified_names)
        
        for name in unspecified_names:
            base_count = base_counts[name]
            multipliers[name] = max(1, round(samples_per_unspecified / base_count))
    
    return multipliers


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
        required_columns = ['dataset', 'fluseason', 'location_code', 'sample', 'season_week', 'value', 'week_enddate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset '{dataset_name}': {missing_columns}")
            
        df = df[required_columns]
        
        # Create replicas with offset seasons
        for i in range(multiplier):
            df_copy = df.copy()
            df_copy.loc[:, "dataset"] = f"{dataset_name}_{i}"
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


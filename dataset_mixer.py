
import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
from season_setup import SeasonSetup
import xarray as xr
import tqdm

def merge_datasets(dict_of_dfs):
    """
    Merges the datasets in the dict_of_dfs dictionary into a single dataframe.
    The keys of the dictionary are the dataset names and the values are dictionaries
    with keys "df" and "multiplier". The "df" key contains the dataframe and the "multiplier"
    key contains the number of times the dataframe should be repeated in the final dataframe.
    """

    # Combine all input dataframes with their multipliers
    combined_datasets = []
    for dataset_name, dataset_info in dict_of_dfs.items():
        df = dataset_info['df'].copy()
        multiplier = dataset_info['multiplier']
        
        # Ensure required columns exist
        required_columns = ['season_week', 'location_code', 'value', 'fluseason', "week_enddate"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            
        df = df[required_columns]
        for i in range(multiplier):
            df_copy = df.copy()
            df_copy.loc[:,"dataset_name"] = dataset_name + "_" + str(i)
            df_copy.loc[:,"fluseason"] = df_copy.loc[:,"fluseason"] + i*10000
            combined_datasets.append(df_copy)

    combined_df = pd.concat(combined_datasets, ignore_index=True)

    return combined_df



def build_frames(dict_of_dfs):
    combined_df = merge_datasets(dict_of_dfs)
    # Get unique seasons and locations
    seasons = sorted(combined_df['fluseason'].unique())
    location_codes = combined_df.location_code.unique()
    all_frames = []
    for season in tqdm.tqdm(seasons, desc="Building frames", total=len(seasons)):
        
        season_df = combined_df[(combined_df['fluseason'] == season)]
        n_repeat_location_in_season = season_df.groupby("location_code")["dataset_name"].nunique()
        for i in range(n_repeat_location_in_season.max()):
            new_frames = []
            for location in location_codes:
                frame = season_df[season_df["location_code"] == location].sort_values('season_week')
                # this frame can either be empty, or contain one dataset, or severa dataset
                if len(frame) == 0:
                    # Find a random frame from another season for this location
                    alternative_frames = combined_df[
                        (combined_df['location_code'] == location)
                    ]
                    # sample a season in location_df.fluseason.unique()
                    alternative_season = np.random.choice(alternative_frames.fluseason.unique())
                    frame = combined_df[(combined_df['fluseason'] == alternative_season) & (combined_df['location_code'] == location)]
                
                # check if there are multiple datasets in the frame
                n_datasets = frame["dataset_name"].nunique()
                # if ther is several dataset, take the ith dataset in the list of sorted unique dataset names
                if n_datasets > 1:
                    dataset_names = sorted(frame["dataset_name"].unique())
                    frame = frame[frame["dataset_name"] == dataset_names[i%n_datasets]]
                
                #frame = frame.drop(["fluseason", "dataset_name"], axis=1)
                # we now have a frame with only one dataset, we enture it has 53 weeks
                frame = pad_single_frame(frame, location)
                if frame.shape[0] != 53:
                    print (f"Frame for location {location} in season {season} has {frame.shape[0]} weeks")
                    print(frame)
                new_frames.append(frame)
            new_frames = pd.concat(new_frames)#.pivot(index="location_code", columns="season_week", values="value")
            all_frames.append(new_frames)
    return all_frames


def pad_single_frame(frame, location):
    """
    Pad a single frame with missing weeks.
    """

    # we want frame.season_week to be from 1 to 53 so we need to fill the gaps
    # we fill external gap with 0 and internal gap with the previous value
    min_week = frame["season_week"].min()
    max_week = frame["season_week"].max()
    all_weeks = set(range(1, 54))
    missing_weeks = all_weeks - set(frame["season_week"])
    # sort the missing weeks
    missing_weeks = sorted(list(missing_weeks))
    if len(missing_weeks) > 0:
        pass
        #print(f"Missing weeks for location {location} in season {season}: {missing_weeks}")
    for week in missing_weeks:
        if week <= min_week or week >= max_week:
            new_value = 0
        else:
            previous_week = frame[frame["season_week"] == week-1]
            new_value = previous_week["value"].values[0]

        new_frame = pd.DataFrame({
                                "season_week":[week], 
                                "location_code":[location], 
                                "value":[new_value], 
                                #"dataset_name": [frame["dataset_name"].unique()[0]]
                                },)
        frame = pd.concat([frame, new_frame])
        

    return frame.reset_index(drop=True)
    



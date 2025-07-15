import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
from ..utils.season_axis import SeasonAxis
import xarray as xr

def padto64x64(x: np.ndarray) -> np.ndarray:
    return np.pad(
        x,
        ((0, 64 - x.shape[0]), (0, 64 - x.shape[1])),
        mode="constant",
        constant_values=0,
    )

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
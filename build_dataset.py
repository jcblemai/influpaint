import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata
from season_setup import SeasonSetup
import xarray as xr
import epiweeks

def padto64x64(x: np.ndarray) -> np.ndarray:
    return np.pad(
        x,
        ((0, 64 - x.shape[0]), (0, 64 - x.shape[1])),
        mode="constant",
        constant_values=0,
    )


def dataframe_to_xarray(
    df: pd.DataFrame,
    season_setup: SeasonSetup = None,
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
    df_piv = df.pivot(columns="location_code", values=value_column, index=date_column)

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
    df: pd.DataFrame, season_setup: SeasonSetup = None, value_column="value"
) -> np.ndarray:

    samples = []

    df_piv = df.pivot(
        columns="location_code",
        values=value_column,
        index=["fluseason", "fluseason_fraction"],
    )
    for season in df_piv.index.unique(level="fluseason"):
        array = df_piv.loc[season][
            season_setup.locations
        ].to_numpy()  # make sure order is right w.r.t flusight_locations
        array[np.isnan(array)] = 0  # replace NaNs with 0
        samples.append(
            np.array([padto64x64(array)])
        )  # pad to 64x64 and add a dimension for channel

    return samples


def get_from_epidata(
    dataset,
    season_setup: SeasonSetup = None,
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
    elif dataset == "flusight2022_23":
        df = pd.read_csv(
            "Flusight/2022-2023/FluSight-forecast-hub-official/data-truth/truth-Incident Hospitalizations.csv",
            parse_dates=True,
            index_col="date",
        )
        df["week_enddate"] = df.index
    elif dataset == "flusight2023_24":
        df = pd.read_csv(
            "Flusight/2023-2024/FluSight-forecast-hub-official/auxiliary-data/target-data-archive/target-hospital-admissions_2024-04-27.csv",
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

    # get the flu season year and it's fraction elapsed
    df["fluseason"] = df["week_enddate"].apply(season_setup.get_fluseason_year)
    df["fluseason_fraction"] = df["week_enddate"].apply(season_setup.get_fluseason_fraction)
    print(f"RAW Dataset {dataset} has {len(df)} data points, with {len(df['location_code'].unique())} locations,"
            f"and NA values: {df['value'].isna().sum()}, NA locations: {df['location_code'].isna().sum()}")
    # select only the columns we need
    if clean:
        # remove 
        df = clean_dataset(df, season_setup)
       

    

    return df


def clean_dataset(df, season_setup):
    df = df[["week_enddate", "location_code", "value", "fluseason", "fluseason_fraction"]]
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
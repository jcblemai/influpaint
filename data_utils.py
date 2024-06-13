import datetime
import pandas as pd
import numpy as np
from helpers.delphi_epidata import Epidata

# import epiweeks
import xarray as xr

# locations, in the right order
class FluSetup:
    """ 
    A FluSetup object contains locations and season information.

    Parameters:
    - locations (pd.DataFrame): A DataFrame containing location information.
    - fluseason_startdate (pd.Timestamp, optional): The start date of the flu season. Defaults to December 15, 2020.

    Attributes:
    - locations_df (pd.DataFrame): The DataFrame containing location information.
    - locations (list): A list of location codes.
    - fluseason_startdate (pd.Timestamp): The start date of the flu season.

    Methods:
    - get_location_name(location_code): Returns the location name for a given location code.
    - get_dates(length): Returns a date range for a given length.
    - from_flusight2022_23(csv_path, fluseason_startdate, remove_territories): Creates a FluSetup object from Flusight 2022-2023 data.
    - from_flusight2023_24(csv_path, fluseason_startdate, remove_territories): Creates a FluSetup object from Flusight 2023-2024 data.
    - get_fluseason_year(ts): Returns the flu season year for a given timestamp.
    - get_fluseason_fraction(ts): Returns the fraction of the flu season for a given timestamp.
    """

    def __init__(
        self, locations: pd.DataFrame, fluseason_startdate=pd.to_datetime("2020-12-15")
    ):
        self.locations_df = locations

        assert "location_code" in self.locations_df.columns
        # self.locations_df = self.locations_df.set_index("location_code", drop=False)
        if "location_name" not in self.locations_df.columns:
            self.locations_df["location_name"] = self.locations_df["location_code"]

        self.locations = self.locations_df["location_code"].to_list()

        print(f"Spatial Setup with {len(self.locations_df)} locations.")

        self.fluseason_startdate = fluseason_startdate

    def get_location_name(self, location_code):
        if pd.isna(location_code):
            return "NA"
        return self.locations_df[self.locations_df["location_code"] == location_code][
            "location_name"
        ].values[0]

    def get_dates(self, length=52):
        dr = pd.date_range(
            start=self.fluseason_startdate,
            end=self.fluseason_startdate + datetime.timedelta(days=7 * length),
            freq="W",
        )
        return dr

    @classmethod
    def from_flusight2022_23(
        cls,
        csv_path="Flusight/2022-2023/FluSight-forecast-hub-official/data-locations/locations.csv",
        fluseason_startdate=pd.to_datetime("2020-12-15"),
        remove_territories=False,
    ):
        flusight_locations = pd.read_csv(csv_path)
        flusight_locations["geoid"] = flusight_locations["location"] + "000"
        flusight_locations = flusight_locations.iloc[1:, :].reset_index(
            drop=True
        )  # skip first row, which is the US full
        flusight_locations["location_code"] = flusight_locations[
            "location"
        ]  # "location" collides with datasets column name
        flusight_locations.drop(columns=["location"], inplace=True)
        if remove_territories:
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "72"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "78"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "60"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "66"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "69"
            ]
        return cls(
            locations=flusight_locations, fluseason_startdate=fluseason_startdate
        )

    @classmethod
    def from_flusight2023_24(
        cls,
        csv_path="Flusight/2023-2024/FluSight-forecast-hub-official/auxiliary-data/locations.csv",
        fluseason_startdate=pd.to_datetime("2020-12-15"),
        remove_territories=False,
    ):
        flusight_locations = pd.read_csv(
            csv_path,
            converters={"location": lambda x: str(x).strip()},
            skipinitialspace=True,
        )
        flusight_locations["geoid"] = flusight_locations["location"] + "000"
        flusight_locations = flusight_locations.iloc[1:, :].reset_index(
            drop=True
        )  # skip first row, which is the US full
        flusight_locations["location_code"] = flusight_locations[
            "location"
        ]  # "location" collides with datasets column name
        flusight_locations.drop(columns=["location"], inplace=True)
        if remove_territories:
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "72"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "78"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "60"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "66"
            ]
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "69"
            ]
        return cls(
            locations=flusight_locations, fluseason_startdate=fluseason_startdate
        )

    def get_fluseason_year(self, ts):
        return get_season_year(ts, self.fluseason_startdate)

    def get_fluseason_fraction(self, ts):
        return get_season_fraction(ts, self.fluseason_startdate)


def get_season_year(ts, start_date):
    if ts.dayofyear >= start_date.dayofyear:
        return ts.year
    else:
        return ts.year - 1


def get_season_fraction(ts, start_date):
    if ts.dayofyear >= start_date.dayofyear:
        return (ts.dayofyear - start_date.dayofyear) / 365
    else:
        return ((ts.dayofyear + 365) - start_date.dayofyear) / 365


def padto64x64(x: np.ndarray) -> np.ndarray:
    return np.pad(
        x,
        ((0, 64 - x.shape[0]), (0, 64 - x.shape[1])),
        mode="constant",
        constant_values=0,
    )


def dataframe_to_xarray(
    df: pd.DataFrame,
    flusetup: FluSetup = None,
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

    if flusetup is None:
        print(" ⚠️ No FluSetup provided, using all locations in the dataframe.")
        places = df_piv.columns.to_list()
    else:
        df_piv = df_piv[
            flusetup.locations_df["location_code"]
        ]  # make sure order is right w.r.t flusight_locations
        places = flusetup.locations_df["location_code"]

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
    df: pd.DataFrame, flusetup: FluSetup = None, value_column="value"
) -> np.ndarray:

    samples = []

    df_piv = df.pivot(
        columns="location_code",
        values=value_column,
        index=["fluseason", "fluseason_fraction"],
    )
    for season in df_piv.index.unique(level="fluseason"):
        array = df_piv.loc[season][
            flusetup.locations
        ].to_numpy()  # make sure order is right w.r.t flusight_locations
        array[np.isnan(array)] = 0  # replace NaNs with 0
        samples.append(
            np.array([padto64x64(array)])
        )  # pad to 64x64 and add a dimension for channel

    return samples


def get_all_locations(dataset):
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
            "datasets.delphi-epidata.src.acquisition.fluview.fluview_locations"
        )
        fll_dict = fluview_locations_m.cdc_to_delphi
        locations = []
        for region_type in fll_dict.keys():
            for region_name, flloc in fll_dict[region_type].items():
                locations.append(flloc)
    return locations


def get_from_epidata(
    dataset,
    flusetup: FluSetup = None,
    locations="all",
    value_col=None,
    write=True,
    download=True,
):
    """ 
    Read a dataset from epidata. Each dataset is a dataframe with columns:
    - 'week_enddate' (datetime)
    - 'location' (str) original location name
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
                locations = get_all_locations(dataset=dataset)

            for location in locations:
                if dataset == "flusurv":
                    res = Epidata.flusurv(
                        location, [Epidata.range(190001, 202251)]
                    )  # large range to get all data
                elif dataset == "fluview":
                    res = Epidata.fluview(
                        location, [Epidata.range(190001, 202251)]
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

    if flusetup is not None:
        # merge with locations, taking care of new york
        if dataset == "flusurv":
            df["location_tomerge"] = df["location"]
            df["location_tomerge"] = df["location_tomerge"].str.replace(
                "NY_albany", "NY"
            )
            df["location_tomerge"] = df["location_tomerge"].str.replace(
                "NY_rochester", "NY"
            )
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
            right_on = "location_code"

        df = pd.merge(
            df,
            flusetup.locations_df,
            left_on="location_tomerge",
            right_on=right_on,
            how="left",
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
    df["fluseason"] = df["week_enddate"].apply(flusetup.get_fluseason_year)
    df["fluseason_fraction"] = df["week_enddate"].apply(flusetup.get_fluseason_fraction)

    return df

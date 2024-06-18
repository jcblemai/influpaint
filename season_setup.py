import datetime
import pandas as pd

# locations, in the right order
class SeasonSetup:
    """ 
    A SeasonSetup object contains locations and season information start date for a given season. It is useful 
    for creating datasets and models that are season-specific.

    Parameters:
    - locations (pd.DataFrame): A DataFrame containing location information.
    - fluseason_startdate (pd.Timestamp, optional): The start date of the flu season. Defaults to July 15, 2020.

    Attributes:
    - locations_df (pd.DataFrame): The DataFrame containing location information.
    - locations (list): A list of location codes.
    - fluseason_startdate (pd.Timestamp): The start date of the flu season.

    Methods:
    - get_location_name(location_code): Returns the location name for a given location code.
    - get_dates(length): Returns a date range for a given length.
    - from_flusight2022_23(location_filepath, fluseason_startdate, remove_territories): Creates a SeasonSetup object from Flusight 2022-2023 repository.
    - from_flusight2023_24(location_filepath, fluseason_startdate, remove_territories): Creates a SeasonSetup object from Flusight 2023-2024 repository.
    - get_fluseason_year(ts): Returns the flu season year for a given timestamp.
    - get_fluseason_fraction(ts): Returns the fraction of the flu season for a given timestamp.
    """

    def __init__(
        self, locations: pd.DataFrame, fluseason_startdate=pd.to_datetime("2020-07-15")
    ):
        self.locations_df = locations

        assert "location_code" in self.locations_df.columns
        # self.locations_df = self.locations_df.set_index("location_code", drop=False)
        if "location_name" not in self.locations_df.columns:
            self.locations_df["location_name"] = self.locations_df["location_code"]

        self.locations = self.locations_df["location_code"].to_list()

        

        self.fluseason_startdate = fluseason_startdate

        print(f"Spatial Setup with {len(self.locations_df)} locations, with a season start_date of {self.fluseason_startdate.strftime('%b %d')}")
    
    @classmethod
    def from_flusight(
        cls,
        location_filepath=None,
        fluseason_startdate=pd.to_datetime("2020-07-15"),
        remove_territories=False,
        remove_us=False,
    ):
        if location_filepath is None:
            #location_filepath = "Flusight/2022-2023/FluSight-forecast-hub-official/auxiliary-data/locations.csv"
            location_filepath = "Flusight/2023-2024/FluSight-forecast-hub-official/auxiliary-data/locations.csv"

        flusight_locations = pd.read_csv(
            location_filepath,
            converters={"location": lambda x: str(x).strip()},
            skipinitialspace=True,
        )
        flusight_locations["geoid"] = flusight_locations["location"] + "000"

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

        if remove_us:
            flusight_locations = flusight_locations[
                flusight_locations["location_code"] != "US"
            ]

        flusight_locations = flusight_locations[["abbreviation", "location_name", "population", "location_code", "geoid"]]
        return cls(
            locations=flusight_locations, fluseason_startdate=fluseason_startdate
        )

    def get_fluseason_year(self, ts):
        return get_season_year(ts, self.fluseason_startdate)

    def get_fluseason_fraction(self, ts):
        return get_season_fraction(ts, self.fluseason_startdate)
    
    def get_location_name(self, location_code):
        if pd.isna(location_code):
            return "NA"
        return self.locations_df[self.locations_df["location_code"] == location_code][
            "location_name"
        ].values[0]

    def get_dates(self, length=52, freq="W-SAT"):
        dr = pd.date_range(
            start=self.fluseason_startdate,
            end=self.fluseason_startdate + datetime.timedelta(days=7 * length),
            freq=freq,
        )
        return dr


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

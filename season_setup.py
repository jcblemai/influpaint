import datetime
import pandas as pd
import math
from typing import Union

def add_season_columns(df, season_setup, do_fluseason_year=True, do_epiweek=False):
    assert "week_enddate" in df.columns, "DataFrame must contain 'week_enddate' column"
    
    df = df.assign(fluseason_fraction=df["week_enddate"].apply(season_setup.get_fluseason_fraction))
    df = df.assign(season_week=df["week_enddate"].apply(season_setup.get_fluseason_week))
    
    if do_epiweek:
        import epiweeks
        df = df.assign(epiweek=df["week_enddate"].apply(lambda x: epiweeks.Week.fromdate(x).week))
    
    if do_fluseason_year:
        df = df.assign(fluseason=df["week_enddate"].apply(season_setup.get_fluseason_year))

    return df

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
    - from_flusight(location_filepath, fluseason_startdate, remove_territories): Creates a SeasonSetup object from Flusight 2022-2023 repository 
            (because 2022-2023 contains virgin islands, 2023-2024 does not)
    - get_fluseason_year(ts): Returns the flu season year for a given timestamp.
    - get_fluseason_fraction(ts): Returns the fraction of the flu season for a given timestamp.
    """

    def __init__(
        self, locations: pd.DataFrame, fluseason_startdate=pd.to_datetime("2020-08-01")
    ):
        self.update_locations(locations)
        
        self.fluseason_startdate = fluseason_startdate

        print(f"Spatial Setup with {len(self.locations_df)} locations, with a season start_date of {self.fluseason_startdate.strftime('%b %d')}")


    def update_locations(self, new_locations):
        self.locations_df = new_locations
        assert "location_code" in self.locations_df.columns

        if "location_name" not in self.locations_df.columns:
            self.locations_df["location_name"] = self.locations_df["location_code"]
        self.locations = self.locations_df["location_code"].to_list()
    
    @classmethod
    def from_flusight(
        cls,
        location_filepath=None,
        season_first_year=None,
        fluseason_startdate=None,
        remove_territories=False,
        remove_us=False,
    ):
        if location_filepath is None:
            if season_first_year == "2022":
                location_filepath = "Flusight/2022-2023/FluSight-forecast-hub-official/data-locations/locations.csv"
            elif season_first_year == "2023":
                location_filepath = "Flusight/2023-2024/FluSight-forecast-hub-official/auxiliary-data/locations.csv"
            elif season_first_year == "2024":
                location_filepath = "Flusight/2024-2025/FluSight-forecast-hub-official/auxiliary-data/locations.csv"
            elif season_first_year is None:
                print("No season nor file provided, loading 2022-2023 locations information")
                # 2022-2023 contains virgin islands, 2023-2024 does not. THourgh then population are not up to date.
                location_filepath = "Flusight/2022-2023/FluSight-forecast-hub-official/data-locations/locations.csv"
            else:
                raise ValueError(f"unreconized season {season_first_year}")
        
        if fluseason_startdate is None:
            fluseason_startdate = pd.to_datetime(f"2020-08-01")

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
        to_remove = []
        if remove_territories:
            to_remove += ["72", "78", "60", "66", "69"]
        if remove_us:
            to_remove += ["US"]

        flusight_locations = remove_locations(location_list=to_remove, locations_df=flusight_locations)

        flusight_locations = flusight_locations[["abbreviation", "location_name", "population", "location_code", "geoid"]]
        return cls(
            locations=flusight_locations, fluseason_startdate=fluseason_startdate
        )

    def get_fluseason_year(self, ts):
        return get_season_year(ts, self.fluseason_startdate)
    
    def get_fluseason_fraction(self, ts):
        return get_season_fraction(ts, self.fluseason_startdate)
    
    def get_fluseason_week(self, ts):
        return get_season_week(ts, start_month=self.fluseason_startdate.month, 
                                start_day=self.fluseason_startdate.day)


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
    
    def reorder_locations(self, ordered_list):
        self.locations_df = self.locations_df[self.locations_df["location_code"].isin(ordered_list)]
        self.locations = ordered_list
        

def remove_locations(location_list, locations_df):
    return locations_df[~locations_df["location_code"].isin(location_list)]

    
def get_season_year(ts, start_date):
    start_month= start_date.month
    start_day= start_date.day
    if isinstance(ts, datetime.datetime):
        ts = ts.date()

    if ts.month > start_month or (ts.month == start_month and ts.day >= start_day):
        return ts.year
    else:
        return ts.year - 1


def get_season_fraction(ts, start_date):
    if ts.dayofyear >= start_date.dayofyear:
        return (ts.dayofyear - start_date.dayofyear) / 365
    else:
        return ((ts.dayofyear + 365) - start_date.dayofyear) / 365
    

def get_season_week(ts: Union[str, datetime.date, datetime.datetime],
                    start_month: int = 8, 
                    start_day: int = 1) -> int:
    """
    Calculate the flu-season week number using fixed 7-day bins from season start.

    This function assigns a 1-based week number relative to a season defined
    by its start date (default: August 1). Each week corresponds to a
    contiguous 7-day period since the season start.
    All dates before the official start are clamped to Week 1.

    Parameters
    ----------
    ts : str or datetime.date or datetime.datetime
        The date to be converted. Strings must follow 'YYYY-MM-DD' format.
    start_month : int, optional
        Month that marks the beginning of the flu season, by default 8.
    start_day : int, optional
        Day of the start month that begins the flu season, by default 1.

    Returns
    -------
    int
        The week number within the flu season (1–53).

    Notes
    -----
    - A 365-day non-leap year yields 52 weeks plus 1 day (partial week).
    - A leap year yields a possible week 53 if the season crosses Feb 29.
    - Weeks are fixed 7-day bins. E.g., Week 1 = Aug 1–7, Week 2 = Aug 8–14, etc.

    Examples
    --------
    >>> get_season_week("2023-08-01")
    1
    >>> get_season_week(datetime.date(2023, 8, 10))
    2
    >>> get_season_week("2023-07-30")  # before season start
    1
    """

    if isinstance(ts, str):
        ts = datetime.datetime.strptime(ts, "%Y-%m-%d").date()
    elif isinstance(ts, datetime.datetime):
        ts = ts.date()

    year = ts.year
    start = datetime.date(year, start_month, start_day)
    if ts < start:
        start = datetime.date(year - 1, start_month, start_day)

    days_elapsed = (ts - start).days
    week = math.floor(days_elapsed / 7) + 1
    return max(1, week)



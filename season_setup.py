import datetime
import pandas as pd
import math
import epiweeks


def add_season_columns(df, season_setup, do_fluseason_year=True):
    df = df.assign(fluseason_fraction=df["week_enddate"].apply(season_setup.get_fluseason_fraction))
    df = df.assign(season_week=df["week_enddate"].apply(season_setup.get_fluseason_week))
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
    

def get_season_week(ts, start_month=8, start_day=1):
    """
    Calculate flu season week number using fixed 7-day bins from season start.
    
    This function provides consistent temporal alignment across flu seasons by using
    fixed calendar-based week boundaries. The same calendar date will always map to
    the same week number across different flu seasons, ensuring perfect alignment
    for seasonal comparisons and array-based modeling.
    
    Week boundaries:
    - Week 1: August 1-7
    - Week 2: August 8-14
    - ...
    - Week 52/53: Depends on season length (365 vs 366 days)
    
    Args:
        ts (date or datetime): Date to convert to season week
        start_month (int): Season start month (default: 8 for August)
        start_day (int): Season start day (default: 1)

    Returns:
        int: Season week number (1-based), typically 1-52 or 1-53
        
    Example:
        >>> get_season_week(datetime.date(2023, 8, 10))  # August 10th
        2
        >>> get_season_week(datetime.date(2024, 8, 10))  # August 10th next year
        2  # Same week number - perfect alignment!
    """

    # Convert to date if datetime is passed
    if isinstance(ts, datetime.datetime):
        ts = ts.date()

    if ts.month > start_month or (ts.month == start_month and ts.day >= start_day):
        season_start = datetime.date(ts.year, start_month, start_day)
    else:
        season_start = datetime.date(ts.year - 1, start_month, start_day)

    # Calculate days elapsed
    days_elapsed = (ts - season_start).days
    if days_elapsed > 365:
        print(f"Warning: days elapsed is {days_elapsed}, this should not happen")
        print(f"ts: {ts}, season_start: {season_start}")
        print(f"start_month: {start_month}, start_day: {start_day}")

    # Calculate week number (1-based, ensuring it never exceeds 53)
    return math.floor(days_elapsed / 7) + 1



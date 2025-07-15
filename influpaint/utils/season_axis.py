import datetime
import pandas as pd
import math
from typing import Union

# locations, in the right order
class SeasonAxis:
    """ 
    A SeasonAxis object manages locations and flu season temporal coordinates.
    
    This class serves two distinct purposes:
    
    1. **Abstract Temporal Coordinates (for arrays/training)**:
        - get_season_week(): Maps dates to season week numbers (1-53)
        - Provides consistent temporal alignment across flu seasons
        - Used for creating arrays, training models, seasonal overlays
    
    2. **Concrete Calendar Mapping (for forecasting)**:
        - get_week_dates(): Returns actual date ranges for specific weeks/years
        - week_to_saturday(): Maps season weeks to specific Saturday dates  
        - get_season_calendar(): Full calendar for a specific flu season year
        - Used for converting model predictions to real calendar dates

    Parameters:
    - locations (pd.DataFrame): DataFrame with location information
    - season_start_month (int, optional): Start month for flu seasons. Defaults to 8 (August).
    - season_start_day (int, optional): Start day for flu seasons. Defaults to 1.

    Attributes:
    - locations_df (pd.DataFrame): DataFrame containing location information
    - locations (list): List of location codes in order
    - season_start_month (int): Start month for flu seasons (1-12)
    - season_start_day (int): Start day for flu seasons (1-31)

    Key Methods:
    
    Abstract Temporal:
    - get_season_week(ts): Season week number (like epiweek but for flu seasons)
    - get_fluseason_year(ts): Flu season year for a given date
    - get_fluseason_fraction(ts): Fraction of flu season elapsed (0.0-1.0)
    
    Concrete Calendar:  
    - get_week_dates(year, week): Actual date range for week N in year Y
    - week_to_saturday(year, week): Saturday date for week N in year Y
    - get_season_calendar(year): Full calendar mapping for a specific season
    
    Utilities:
    - get_location_name(code): Location name for a given code
    - from_flusight(): Create from FluSight location data
    
    Example:
        # Abstract temporal coordinate (for arrays)
        week = season_setup.get_season_week("2023-11-25")  # Returns 17
        
        # Concrete calendar mapping (for forecasting)  
        start, end = season_setup.get_week_dates(2023, 17)  # Nov 19-25, 2023
        saturday = season_setup.week_to_saturday(2024, 17)  # Nov 23, 2024
    """

    def __init__(
        self, locations: pd.DataFrame, season_start_month: int = 8, season_start_day: int = 1
    ):
        self.update_locations(locations)
        
        self.season_start_month = season_start_month
        self.season_start_day = season_start_day

        print(f"Spatial Setup with {len(self.locations_df)} locations, with a season start_date of {pd.to_datetime(f'2020-{season_start_month:02d}-{season_start_day:02d}').strftime('%b %d')}")


    def update_locations(self, new_locations):
        self.locations_df = new_locations
        assert "location_code" in self.locations_df.columns

        if "location_name" not in self.locations_df.columns:
            self.locations_df["location_name"] = self.locations_df["location_code"]
        self.locations = self.locations_df["location_code"].to_list()
    
    @classmethod
    def for_flusight(
        cls,
        location_filepath=None,
        season_start_month=8,
        season_start_day=1,
        remove_territories=False,
        remove_us=False,
    ):
        location_filepath = "influpaint_locations.csv" if location_filepath is None else location_filepath
        
        influpaint_locations = pd.read_csv(
            location_filepath,
            converters={"location_code": lambda x: str(x).strip(), "geoid": lambda x: str(x).strip()},
            skipinitialspace=True,
        )
        
        to_remove = []
        if remove_territories:
            to_remove += ["72", "60", "66", "69"] # Virgin Island would be 78, but not in the file
        if remove_us:
            to_remove += ["US"]

        influpaint_locations = influpaint_locations[~influpaint_locations["location_code"].isin(to_remove)]

        return cls(
            locations=influpaint_locations, season_start_month=season_start_month, season_start_day=season_start_day
        )

    def get_fluseason_year(self, ts):
        return get_season_year(ts, self.season_start_month, self.season_start_day)
    
    def get_fluseason_fraction(self, ts):
        return get_season_fraction(ts, self.season_start_month, self.season_start_day)
    

    
    # === SEASON WEEK (abstract temporal coordinate like epiweek) ===
    def get_season_week(self, ts) -> int:
        """
        Get season week number (like epiweek but for flu seasons).
        
        Maps dates to week numbers using fixed 7-day bins from season start.
        This is the primary temporal coordinate for arrays and seasonal alignment.
        Can return 1-53 depending on calendar alignment.
        
        Args:
            ts: Date to convert
            
        Returns:
            int: Season week number (1-53)
        """
        return get_season_week(ts, start_month=self.season_start_month, 
                            start_day=self.season_start_day)
    
    # === CONCRETE CALENDAR MAPPING (for forecasting) ===
    def get_week_dates(self, season_year: int, week_number: int) -> tuple[datetime.date, datetime.date]:
        """
        Get actual start and end dates for a specific week in a specific season.
        
        Args:
            season_year: The year the flu season starts (e.g., 2023 for 2023-2024 season)
            week_number: Season week number (1-53)
            
        Returns:
            tuple: (start_date, end_date) for that week
        """
        season_start = datetime.date(season_year, self.season_start_month, self.season_start_day)
        week_start = season_start + datetime.timedelta(days=(week_number - 1) * 7)
        week_end = week_start + datetime.timedelta(days=6)
        return week_start, week_end
    
    def week_to_saturday(self, season_year: int, week_number: int) -> datetime.date:
        """
        Get the Saturday date for a specific season week in a specific year.
        
        Args:
            season_year: The year the flu season starts
            week_number: Season week number (1-53)
            
        Returns:
            datetime.date: The Saturday of that week
        """
        week_start, week_end = self.get_week_dates(season_year, week_number)
        
        # Find the Saturday in this week
        for day_offset in range(7):
            candidate = week_start + datetime.timedelta(days=day_offset)
            if candidate.weekday() == 5:  # Saturday is weekday 5
                return candidate
        
        # If no Saturday in the week range, return the last day
        return week_end
    
    def get_season_calendar(self, season_year: int) -> pd.DataFrame:
        """
        Get full calendar mapping for a specific flu season.
        
        Args:
            season_year: The year the flu season starts
            
        Returns:
            pd.DataFrame: Calendar with [season_week, start_date, end_date, saturday]
        """
        calendar_data = []
        
        # Generate up to 53 weeks
        for week_num in range(1, 54):
            start_date, end_date = self.get_week_dates(season_year, week_num)
            saturday = self.week_to_saturday(season_year, week_num)
            
            # Stop if we've gone past the end of the season
            next_season_start = datetime.date(season_year + 1, self.season_start_month, self.season_start_day)
            if start_date >= next_season_start:
                break
                
            calendar_data.append({
                'season_week': week_num,
                'start_date': start_date,
                'end_date': end_date,
                'saturday': saturday
            })
        
        return pd.DataFrame(calendar_data)

    def get_location_name(self, location_code):
        if pd.isna(location_code):
            return "NA"
        return self.locations_df[self.locations_df["location_code"] == location_code][
            "location_name"
        ].values[0]

    def get_dates(self, length=52, freq="W-SAT"):
        # Use a reference year for generating date ranges
        reference_start = datetime.date(2020, self.season_start_month, self.season_start_day)
        dr = pd.date_range(
            start=reference_start,
            end=reference_start + datetime.timedelta(days=7 * length),
            freq=freq,
        )
        return dr
    
    def reorder_locations(self, ordered_list):
        self.locations_df = self.locations_df[self.locations_df["location_code"].isin(ordered_list)]
        self.locations = ordered_list

    def add_season_columns(self, df,  do_fluseason_year=True, do_epiweek=False):
        assert "week_enddate" in df.columns, "DataFrame must contain 'week_enddate' column"
        
        df = df.assign(fluseason_fraction=df["week_enddate"].apply(self.get_fluseason_fraction))
        df = df.assign(season_week=df["week_enddate"].apply(self.get_season_week))
        
        if do_epiweek:
            import epiweeks
            df = df.assign(epiweek=df["week_enddate"].apply(lambda x: epiweeks.Week.fromdate(x).week))
        
        if do_fluseason_year:
            df = df.assign(fluseason=df["week_enddate"].apply(self.get_fluseason_year))

        return df

    
def get_season_year(ts, start_month: int, start_day: int):
    # Handle NaT/NaN values
    if pd.isna(ts):
        return float('nan')
        
    if isinstance(ts, datetime.datetime):
        ts = ts.date()

    if ts.month > start_month or (ts.month == start_month and ts.day >= start_day):
        return ts.year
    else:
        return ts.year - 1


def get_season_fraction(ts, start_month: int, start_day: int):
    # Handle NaT/NaN values
    if pd.isna(ts):
        return float('nan')
        
    if isinstance(ts, datetime.datetime):
        ts = ts.date()
    
    # Create a season start date for the current year
    season_start = datetime.date(ts.year, start_month, start_day)
    
    # If the date is before season start, use previous year's season start
    if ts < season_start:
        season_start = datetime.date(ts.year - 1, start_month, start_day)
    
    days_since_start = (ts - season_start).days
    return days_since_start / 365
    

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

    # Handle NaT/NaN values
    if pd.isna(ts):
        return float('nan')

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



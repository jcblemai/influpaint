"""
dataset_mixer.py - Epidemic Data Augmentation and Frame Construction

Combines multiple epidemic surveillance datasets into a unified training corpus 
for diffusion models. Addresses common challenges in epidemic modeling:

- **Dataset Rebalancing**: Uses multipliers to weight data sources
- **Temporal Completeness**: Ensures all frames have complete weekly coverage (1-53)
- **Spatial Completeness**: Fills missing location-season combinations
- **Gap Handling**: Fills missing weeks and locations

Key Components:
--------------
1. **Multiplier Calculation**: Compute dataset weights for target proportions
2. **Frame Construction**: Build complete epidemic frames for training
3. **Gap Filling**: Handle missing data

Typical Usage:
--------------
# Step 1: Combine datasets into hierarchical structure
all_datasets_df = pd.concat([fluview_df, nc_df, smh_traj_df])

# Step 2: Configure dataset inclusion and weighting
config = { 
    "fluview": {"proportion": 0.7, "total": 1000},    # 70% of final dataset
    "smh_traj": {"proportion": 0.3, "total": 1000}   # 30% of final dataset
}

# Step 3: Build complete frames with configurable location handling
frames = build_frames(all_datasets_df, config, season_axis, 
                     fill_missing_locations="zeros")

# Alternative: Use explicit multipliers instead of proportions
config = {
    "fluview": {"multiplier": 2},     # Include twice
    "smh_traj": {"multiplier": 1}     # Include once
}
frames = build_frames(all_datasets_df, config, season_axis)

Output Format:
--------------
Each frame is a complete epidemic season with:
- All weeks (1-53) represented
- All locations covered  
- Consistent data structure for array conversion

Enables training on heterogeneous surveillance data while maintaining 
epidemiological structure.
"""

import pandas as pd
import numpy as np
from season_axis import SeasonAxis
from tqdm import tqdm


def _validate_required_columns(df: pd.DataFrame, required_columns: list, 
                              context: str) -> None:
    """Validate that DataFrame contains all required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{context} missing required columns: {missing_columns}")


def _validate_config_consistency(config: dict) -> tuple[bool, bool]:
    """Validate config consistency and return flags for approach types."""
    has_proportions = any("proportion" in cfg for cfg in config.values())
    has_multipliers = any("multiplier" in cfg for cfg in config.values())
    
    if has_proportions and has_multipliers:
        raise ValueError(
            "Cannot mix 'proportion' and 'multiplier' configs. Choose one approach.\n\n"
            "Option A - All proportions for exact composition control:\n"
            "config = {\n"
            '    "fluview": {"proportion": 0.7, "total": 1000},\n'
            '    "nc_payload": {"proportion": 0.3, "total": 1000}\n'
            "}\n\n"
            "Option B - All multipliers for explicit replication control:\n"
            "multipliers = calculate_multipliers(data, total=1000, target_proportions={'fluview': 0.7})\n"
            "config = {name: {'multiplier': mult} for name, mult in multipliers.items()}"
        )
    
    return has_proportions, has_multipliers

def build_frames(all_datasets_df: pd.DataFrame, config: dict, season_axis: SeasonAxis, 
                 fill_missing_locations: str = "error") -> list:
    """
    Build complete epidemic frames from hierarchical dataset structure.
    
    Handles 4-level hierarchy: H1 → H2 → Season → Sample and creates complete 
    frames while preserving dataset origins.
    
    Args:
        all_datasets_df (pd.DataFrame): Combined dataset with required columns:
            - datasetH1: Top-level dataset category (e.g., 'fluview', 'smh_traj')  
            - datasetH2: Sub-dataset within H1 (e.g., 'round4_CADPH-FluCAT_A-2024-08-01')
            - fluseason: Flu season year
            - sample: Sample identifier within each H2/season combination
            - location_code, season_week, value, week_enddate: Epidemic data
            
        config (dict): Configuration for dataset inclusion and weighting:
            - Keys: H1 dataset names (must exist in datasetH1 column)
            - Values: Either {"multiplier": int} or {"proportion": float, "total": int}
            
        season_axis (SeasonAxis): Season axis object providing location definitions
        
        fill_missing_locations (str): Strategy for handling missing locations:
            - "error": Fail if any expected location is missing (default)
            - "zeros": Fill missing locations with zeros
            - "random": Fill missing locations with random other season data
            - "skip": Skip frames with missing locations
            
    Returns:
        list: Complete epidemic frames, where each frame contains:
            - All weeks (1-53) for all expected locations (based on season_axis)
            - Origin column tracking source: "H1/H2/season/sample"
            - Replicated datasets as specified by config
            
    Example:
        config = {
            "fluview": {"multiplier": 1},
            "smh_traj": {"proportion": 0.7, "total": 1000}
        }
        frames = build_frames(all_datasets_df, config, season_axis, 
                             fill_missing_locations="zeros")
        
    Notes:
        - If H1 dataset is included, ALL H2s and seasons within it are included
        - Minimum frames = sum(n_H2 * n_seasons) for each included H1
        - Samples are replicated, not created (e.g., sample_1_copy1, sample_1_copy2)
        - Location completeness is enforced based on season_axis.locations
    """
    # Validate input dataframe
    required_columns = ['datasetH1', 'datasetH2', 'fluseason', 'sample', 
                       'location_code', 'season_week', 'value', 'week_enddate']
    _validate_required_columns(all_datasets_df, required_columns, "Input dataframe")
    
    # Validate config references existing H1 datasets
    available_h1 = set(all_datasets_df['datasetH1'].unique())
    config_h1 = set(config.keys())
    missing_h1 = config_h1 - available_h1
    if missing_h1:
        raise ValueError(f"Config references non-existent H1 datasets: {missing_h1}")
    
    # Validate fill_missing_locations parameter
    valid_strategies = {"error", "zeros", "random", "skip"}
    if fill_missing_locations not in valid_strategies:
        raise ValueError(f"fill_missing_locations must be one of: {valid_strategies}")
    
    # Calculate multipliers for each H1 dataset
    h1_multipliers = _calculate_h1_multipliers(all_datasets_df, config)
    
    # Pre-compute global lookup table for intelligent filling (do this once)
    global_lookup = None
    if fill_missing_locations == "random":
        print("Pre-computing intelligent fill lookup table...")
        global_lookup = _build_global_lookup_table(all_datasets_df)
    
    # Build frames for each included H1 dataset  
    all_frames = []
    frame_summary = {}
    
    print("Building frames...")
    for h1_name, multiplier in h1_multipliers.items():
        print(f"Processing {h1_name} (multiplier={multiplier})...")
        h1_data = all_datasets_df[all_datasets_df['datasetH1'] == h1_name].copy()
        h1_frames = _build_h1_frames(h1_data, h1_name, multiplier, season_axis, fill_missing_locations, all_datasets_df, global_lookup)
        all_frames.extend(h1_frames)
        
        # Build summary for this H1 dataset
        h2_counts = {}
        for frame in h1_frames:
            if 'datasetH2' in frame.columns:
                h2 = frame['datasetH2'].iloc[0]
                # Handle NaN values properly
                if pd.isna(h2):
                    h2 = "<missing_datasetH2>"
                h2_counts[h2] = h2_counts.get(h2, 0) + 1
        
        frame_summary[h1_name] = {
            'total_frames': len(h1_frames),
            'multiplier': multiplier,
            'h2_breakdown': h2_counts
        }
    
    # Print summary
    print(f"Created {len(all_frames)} total frames:")
    for h1_name, summary in frame_summary.items():
        print(f"  {h1_name}: {summary['total_frames']} frames (multiplier={summary['multiplier']})")
        for h2, count in summary['h2_breakdown'].items():
            print(f"    {h2}: {count} frames")
    
    # Clean up frames by removing unnecessary metadata columns
    print("Cleaning up frame columns...")
    cleaned_frames = []
    essential_columns = ['location_code', 'season_week', 'value', 'week_enddate', 'origin']
    
    for frame in all_frames:
        # Keep only essential columns that have actual data
        available_essential = [col for col in essential_columns if col in frame.columns]
        cleaned_frame = frame[available_essential].copy()
        cleaned_frames.append(cleaned_frame)
    
    return cleaned_frames


def _calculate_h1_multipliers(all_datasets_df: pd.DataFrame, config: dict) -> dict:
    """Calculate multipliers for each H1 dataset based on config."""
    # Validate config consistency - no mixing approaches
    has_proportions, has_multipliers = _validate_config_consistency(config)
    
    # Process configs based on consistent approach
    if has_proportions:
        return _calculate_proportional_multipliers(all_datasets_df, config)
    else:
        return _calculate_explicit_multipliers(config)


def _calculate_proportional_multipliers(all_datasets_df: pd.DataFrame, config: dict) -> dict:
    """Calculate multipliers from proportion-based config."""
    h1_multipliers = {}
    total_target = None
    
    # Validate all configs have proportion + total
    for h1_name, h1_config in config.items():
        if "proportion" not in h1_config or "total" not in h1_config:
            raise ValueError(f"Config for '{h1_name}' must have both 'proportion' and 'total'")
        
        if total_target is None:
            total_target = h1_config["total"]
        elif total_target != h1_config["total"]:
            raise ValueError("All proportion configs must have same 'total' value")
    
    # Calculate base sample counts
    base_counts = {}
    for h1_name in config:
        h1_data = all_datasets_df[all_datasets_df['datasetH1'] == h1_name]
        base_count = len(h1_data.groupby(['datasetH2', 'fluseason', 'sample']))
        base_counts[h1_name] = base_count
    
    # Calculate multipliers to achieve target proportions
    for h1_name, h1_config in config.items():
        target_samples = int(total_target * h1_config["proportion"])
        base_count = base_counts[h1_name]
        h1_multipliers[h1_name] = max(1, round(target_samples / base_count))
    
    return h1_multipliers


def _calculate_explicit_multipliers(config: dict) -> dict:
    """Extract explicit multipliers from config."""
    h1_multipliers = {}
    
    for h1_name, h1_config in config.items():
        if "multiplier" not in h1_config:
            raise ValueError(f"Config for '{h1_name}' must have 'multiplier'")
        h1_multipliers[h1_name] = h1_config["multiplier"]
    
    return h1_multipliers


def _build_h1_frames(h1_data: pd.DataFrame, h1_name: str, multiplier: int, 
                    season_axis: SeasonAxis, fill_missing_locations: str, 
                    all_datasets_df: pd.DataFrame = None, global_lookup: dict = None) -> list:
    """Build frames for a single H1 dataset with replication."""
    frames = []
    
    # Calculate total work for progress bar
    total_work = 0
    for copy_num in range(multiplier):
        for (h2, season), group in h1_data.groupby(['datasetH2', 'fluseason']):
            total_work += len(group['sample'].unique())
    
    # Create replicated copies with progress bar
    pbar = tqdm(total=total_work, desc=f"  {h1_name} frames", leave=False)
    
    for copy_num in range(multiplier):
        copy_suffix = f"_copy{copy_num}" if copy_num > 0 else ""
        
        # Process each H2/season combination
        for (h2, season), group in h1_data.groupby(['datasetH2', 'fluseason']):
            # Each sample in this H2/season becomes a separate frame
            for sample_id, sample_data in group.groupby('sample'):
                # Create origin identifier
                origin = f"{h1_name}/{h2}/{season}/{sample_id}{copy_suffix}"
                
                # Build complete frame for this sample
                frame = sample_data.copy()
                frame['origin'] = origin
                
                # Ensure complete weekly and location coverage
                frame = _pad_frame_complete(frame, season_axis, fill_missing_locations, all_datasets_df, global_lookup)
                
                # Skip frames with missing locations if strategy is "skip"
                if frame is None:
                    pbar.update(1)
                    continue
                    
                frames.append(frame)
                pbar.update(1)
    
    pbar.close()
    return frames


def _pad_frame_complete(frame: pd.DataFrame, season_axis: SeasonAxis, 
                       fill_missing_locations: str, all_datasets_df: pd.DataFrame = None, 
                       global_lookup: dict = None) -> pd.DataFrame:
    """Ensure frame has complete weekly and location coverage."""
    expected_locations = set(season_axis.locations)
    actual_locations = set(frame['location_code'].unique())
    missing_locations = expected_locations - actual_locations
    
    # Handle missing locations
    if missing_locations:
        frame = _handle_missing_locations(frame, missing_locations, actual_locations, 
                                        fill_missing_locations, all_datasets_df, global_lookup)
        if frame is None:  # Skip strategy returned None
            return None
    
    # Ensure complete weekly coverage for all locations
    complete_frames = []
    for location in expected_locations:
        location_data = frame[frame['location_code'] == location].copy()
        if location_data.empty:
            continue
            
        # Use existing padding logic for weekly completeness
        location_complete = _pad_single_location(location_data, location)
        
        # Preserve other columns from original frame
        _preserve_columns(location_complete, frame, location_data)
        complete_frames.append(location_complete)
    
    if not complete_frames:
        return None
        
    final_frame = pd.concat(complete_frames, ignore_index=True)
    return final_frame.sort_values(['location_code', 'season_week']).reset_index(drop=True)


def _handle_missing_locations(frame: pd.DataFrame, missing_locations: set, 
                            actual_locations: set, strategy: str, 
                            all_datasets_df: pd.DataFrame = None, global_lookup: dict = None) -> pd.DataFrame:
    """Handle missing locations according to specified strategy."""
    if strategy == "error":
        # Get frame context for better error message
        origin = frame['origin'].iloc[0] if 'origin' in frame.columns and not frame.empty else "unknown"
        h1 = frame['datasetH1'].iloc[0] if 'datasetH1' in frame.columns and not frame.empty else "unknown"
        h2 = frame['datasetH2'].iloc[0] if 'datasetH2' in frame.columns and not frame.empty else "unknown"
        season = frame['fluseason'].iloc[0] if 'fluseason' in frame.columns and not frame.empty else "unknown"
        sample = frame['sample'].iloc[0] if 'sample' in frame.columns and not frame.empty else "unknown"
        
        raise ValueError(f"Missing required locations: {missing_locations}\n"
                        f"Frame: {origin} (H1={h1}, H2={h2}, season={season}, sample={sample})\n"
                        f"Available locations: {actual_locations}")
    elif strategy == "skip":
        return None
    elif strategy == "zeros":
        return _fill_missing_with_zeros(frame, missing_locations)
    elif strategy == "random":
        return _fill_missing_with_random(frame, missing_locations, actual_locations, all_datasets_df, global_lookup)


def _fill_missing_with_zeros(frame: pd.DataFrame, missing_locations: set) -> pd.DataFrame:
    """Fill missing locations with zero values for all weeks."""
    if not missing_locations:
        return frame
        
    # Batch create all missing rows at once
    missing_rows = []
    
    # Get metadata once and modify origin to indicate synthetic data
    origin = frame['origin'].iloc[0] if 'origin' in frame.columns and not frame.empty else None
    if origin:
        origin = f"{origin}[filled_zeros]"
    
    metadata = {}
    if not frame.empty:
        for col in ['datasetH1', 'datasetH2', 'fluseason', 'sample']:
            if col in frame.columns:
                metadata[col] = frame[col].iloc[0]
    
    # Create all missing rows in batch
    # Get a reference week_enddate from existing data if available
    ref_enddate = None
    if not frame.empty and 'week_enddate' in frame.columns:
        ref_enddate = frame['week_enddate'].iloc[0]
    
    for location in missing_locations:
        for week in range(1, 54):
            missing_row = {
                'location_code': location,
                'season_week': week,
                'value': 0.0,
                'origin': origin,
                **metadata
            }
            
            # Add week_enddate if it exists in original frame
            if ref_enddate is not None:
                missing_row['week_enddate'] = ref_enddate
                
            missing_rows.append(missing_row)
    
    # Single concat
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        frame = pd.concat([frame, missing_df], ignore_index=True)
    
    return frame


def _fill_missing_with_random(frame: pd.DataFrame, missing_locations: set, 
                            actual_locations: set, all_datasets_df: pd.DataFrame = None, 
                            global_lookup: dict = None) -> pd.DataFrame:
    """Fill missing locations with intelligent data from the full dataset."""
    if not actual_locations:
        # Fallback to zeros if no existing data
        return _fill_missing_with_zeros(frame, missing_locations)
    
    # Use pre-computed global lookup for super fast access
    for location in missing_locations:
        fill_data, fill_source = _get_fill_data_from_global_lookup(frame, location, global_lookup, actual_locations)
        
        if fill_data is not None:
            # Use the fill data
            fill_data['location_code'] = location
            # Always set the origin for filled data
            original_origin = frame['origin'].iloc[0]
            fill_data['origin'] = f"{original_origin}[filled_{fill_source}]"
            frame = pd.concat([frame, fill_data], ignore_index=True)
        else:
            # Fallback to zeros if no intelligent fill is possible
            frame = _fill_missing_with_zeros(frame, {location})
    
    return frame


def _build_global_lookup_table(all_datasets_df: pd.DataFrame) -> dict:
    """Pre-compute a global lookup table for all locations/years/samples - do this once."""
    print("  Building location data lookup...")
    
    # Create a hierarchical lookup: location -> h1 -> priority -> data
    lookup = {}
    
    # Group all data by location for fast access
    location_groups = all_datasets_df.groupby('location_code')
    
    for location, location_data in location_groups:
        lookup[location] = {}
        
        # Group by H1 dataset
        h1_groups = location_data.groupby('datasetH1')
        
        for h1_name, h1_data in h1_groups:
            lookup[location][h1_name] = {
                'by_year': {},
                'by_sample': {}
            }
            
            # Group by year for quick year-based lookups
            year_groups = h1_data.groupby('fluseason')
            for year, year_data in year_groups:
                # Store first sample for each year (could be optimized further)
                sample_data = year_data.groupby('sample').first().iloc[0]
                lookup[location][h1_name]['by_year'][year] = {
                    'data': year_data[year_data['sample'] == sample_data.name][['season_week', 'value', 'week_enddate']].copy(),
                    'h2': sample_data['datasetH2'],
                    'sample': sample_data.name
                }
            
            # Also store by sample for sample-based lookups
            sample_groups = h1_data.groupby(['fluseason', 'sample'])
            for (year, sample), sample_data in sample_groups:
                key = f"{year}_{sample}"
                lookup[location][h1_name]['by_sample'][key] = {
                    'data': sample_data[['season_week', 'value', 'week_enddate']].copy(),
                    'h2': sample_data['datasetH2'].iloc[0],
                    'year': year,
                    'sample': sample
                }
    
    print(f"  Lookup table built for {len(lookup)} locations")
    return lookup


def _get_fill_data_from_global_lookup(frame: pd.DataFrame, location: str, 
                                     global_lookup: dict, actual_locations: set) -> tuple:
    """Get fill data using pre-computed global lookup - super fast."""
    if global_lookup is None or location not in global_lookup:
        # Fallback to random
        random_location = np.random.choice(list(actual_locations))
        random_data = frame[frame['location_code'] == random_location].copy()
        return random_data, f"random_from_{random_location}"
    
    # Get frame metadata
    current_h1 = frame['datasetH1'].iloc[0] if 'datasetH1' in frame.columns else None
    current_season = frame['fluseason'].iloc[0] if 'fluseason' in frame.columns else None
    current_sample = frame['sample'].iloc[0] if 'sample' in frame.columns else None
    
    location_lookup = global_lookup[location]
    
    # Strategy 1: Same H1, different year
    if current_h1 in location_lookup:
        h1_lookup = location_lookup[current_h1]
        
        # Find different years
        for year, year_info in h1_lookup['by_year'].items():
            if year != current_season:
                data = year_info['data'].copy()
                source = f"same_location_year_{year}_sample_{year_info['sample']}"
                return data, source
        
        # Strategy 2: Same H1, different sample 
        current_key = f"{current_season}_{current_sample}"
        for sample_key, sample_info in h1_lookup['by_sample'].items():
            if sample_key != current_key and sample_info['year'] == current_season:
                data = sample_info['data'].copy()
                source = f"same_location_sample_{sample_info['sample']}"
                return data, source
    
    # Strategy 3: Different H1
    for h1_name, h1_lookup in location_lookup.items():
        if h1_name != current_h1 and h1_lookup['by_year']:
            # Pick first available year
            year = next(iter(h1_lookup['by_year']))
            year_info = h1_lookup['by_year'][year]
            data = year_info['data'].copy()
            source = f"same_location_{h1_name}_year_{year}"
            return data, source
    
    # Fallback to random
    random_location = np.random.choice(list(actual_locations))
    random_data = frame[frame['location_code'] == random_location].copy()
    return random_data, f"random_from_{random_location}"


def _build_fill_cache(frame: pd.DataFrame, missing_locations: set, all_datasets_df: pd.DataFrame) -> dict:
    """Build a cache of fill data for all missing locations at once."""
    if all_datasets_df is None:
        return {}
    
    # Get frame metadata once
    current_h1 = frame['datasetH1'].iloc[0] if 'datasetH1' in frame.columns else None
    current_season = frame['fluseason'].iloc[0] if 'fluseason' in frame.columns else None
    current_sample = frame['sample'].iloc[0] if 'sample' in frame.columns else None
    
    cache = {}
    
    # Pre-filter the dataset for better performance
    same_h1_data = all_datasets_df[all_datasets_df['datasetH1'] == current_h1] if current_h1 else pd.DataFrame()
    other_h1_data = all_datasets_df[all_datasets_df['datasetH1'] != current_h1] if current_h1 else pd.DataFrame()
    
    for location in missing_locations:
        fill_data = None
        fill_source = None
        
        # Strategy 1: Same location, different year in same H1
        if not same_h1_data.empty:
            location_same_h1 = same_h1_data[same_h1_data['location_code'] == location]
            if not location_same_h1.empty:
                # Try different year first
                different_year = location_same_h1[location_same_h1['fluseason'] != current_season]
                if not different_year.empty:
                    # Pick first available year/sample combination
                    sample_row = different_year.iloc[0]
                    fill_data = _extract_specific_data(all_datasets_df, location, sample_row)
                    fill_source = f"same_location_year_{sample_row['fluseason']}_sample_{sample_row['sample']}"
                else:
                    # Try different sample same year
                    different_sample = location_same_h1[
                        (location_same_h1['fluseason'] == current_season) & 
                        (location_same_h1['sample'] != current_sample)
                    ]
                    if not different_sample.empty:
                        sample_row = different_sample.iloc[0]
                        fill_data = _extract_specific_data(all_datasets_df, location, sample_row)
                        fill_source = f"same_location_sample_{sample_row['sample']}"
        
        # Strategy 2: Same location, different H1 dataset
        if fill_data is None and not other_h1_data.empty:
            location_other_h1 = other_h1_data[other_h1_data['location_code'] == location]
            if not location_other_h1.empty:
                sample_row = location_other_h1.iloc[0]
                fill_data = _extract_specific_data(all_datasets_df, location, sample_row)
                fill_source = f"same_location_{sample_row['datasetH1']}_year_{sample_row['fluseason']}"
        
        # Cache the result (None if no intelligent fill found)
        if fill_data is not None:
            cache[location] = (fill_data, fill_source)
    
    return cache


def _extract_specific_data(all_datasets_df: pd.DataFrame, location: str, sample_row: pd.Series) -> pd.DataFrame:
    """Extract specific location data based on sample row metadata."""
    extracted = all_datasets_df[
        (all_datasets_df['datasetH1'] == sample_row['datasetH1']) &
        (all_datasets_df['datasetH2'] == sample_row['datasetH2']) &
        (all_datasets_df['fluseason'] == sample_row['fluseason']) &
        (all_datasets_df['sample'] == sample_row['sample']) &
        (all_datasets_df['location_code'] == location)
    ][['season_week', 'value', 'week_enddate']].copy()
    
    return extracted if not extracted.empty else None


def _find_intelligent_fill_data(frame: pd.DataFrame, missing_location: str, 
                               actual_locations: set, all_datasets_df: pd.DataFrame = None) -> tuple:
    """
    Find intelligent fill data for a missing location.
    
    Priority order:
    1. Same location from different year/sample in same H1 dataset
    2. Same location from different H1 dataset  
    3. Random location from current frame (fallback)
    
    Returns:
        tuple: (fill_data: pd.DataFrame, source_description: str) or (None, None)
    """
    if all_datasets_df is None:
        # Fallback to random location from current frame
        random_location = np.random.choice(list(actual_locations))
        random_data = frame[frame['location_code'] == random_location].copy()
        return random_data, f"random_from_{random_location}"
    
    # Get current frame metadata
    current_h1 = frame['datasetH1'].iloc[0] if 'datasetH1' in frame.columns else None
    current_h2 = frame['datasetH2'].iloc[0] if 'datasetH2' in frame.columns else None
    current_season = frame['fluseason'].iloc[0] if 'fluseason' in frame.columns else None
    current_sample = frame['sample'].iloc[0] if 'sample' in frame.columns else None
    
    # Strategy 1: Same location, different year/sample in same H1 dataset
    if current_h1:
        same_h1_data = all_datasets_df[
            (all_datasets_df['datasetH1'] == current_h1) & 
            (all_datasets_df['location_code'] == missing_location)
        ]
        
        # Prefer different year first, then different sample
        if not same_h1_data.empty:
            # Try different year first
            different_year = same_h1_data[same_h1_data['fluseason'] != current_season]
            if not different_year.empty:
                sample_data = different_year.groupby(['datasetH2', 'fluseason', 'sample']).first().reset_index()
                if not sample_data.empty:
                    chosen = sample_data.iloc[0]
                    fill_data = all_datasets_df[
                        (all_datasets_df['datasetH1'] == chosen['datasetH1']) &
                        (all_datasets_df['datasetH2'] == chosen['datasetH2']) &
                        (all_datasets_df['fluseason'] == chosen['fluseason']) &
                        (all_datasets_df['sample'] == chosen['sample']) &
                        (all_datasets_df['location_code'] == missing_location)
                    ][['season_week', 'value', 'week_enddate']].copy()
                    
                    source = f"same_location_year_{chosen['fluseason']}_sample_{chosen['sample']}"
                    return fill_data, source
            
            # Try different sample in same year
            different_sample = same_h1_data[
                (same_h1_data['fluseason'] == current_season) & 
                (same_h1_data['sample'] != current_sample)
            ]
            if not different_sample.empty:
                sample_data = different_sample.groupby(['datasetH2', 'sample']).first().reset_index()
                if not sample_data.empty:
                    chosen = sample_data.iloc[0]
                    fill_data = all_datasets_df[
                        (all_datasets_df['datasetH1'] == chosen['datasetH1']) &
                        (all_datasets_df['datasetH2'] == chosen['datasetH2']) &
                        (all_datasets_df['fluseason'] == current_season) &
                        (all_datasets_df['sample'] == chosen['sample']) &
                        (all_datasets_df['location_code'] == missing_location)
                    ][['season_week', 'value', 'week_enddate']].copy()
                    
                    source = f"same_location_sample_{chosen['sample']}"
                    return fill_data, source
    
    # Strategy 2: Same location from different H1 dataset
    different_h1_data = all_datasets_df[
        (all_datasets_df['datasetH1'] != current_h1) & 
        (all_datasets_df['location_code'] == missing_location)
    ]
    
    if not different_h1_data.empty:
        # Group by H1/H2/season/sample and pick first available
        sample_data = different_h1_data.groupby(['datasetH1', 'datasetH2', 'fluseason', 'sample']).first().reset_index()
        if not sample_data.empty:
            chosen = sample_data.iloc[0]
            fill_data = all_datasets_df[
                (all_datasets_df['datasetH1'] == chosen['datasetH1']) &
                (all_datasets_df['datasetH2'] == chosen['datasetH2']) &
                (all_datasets_df['fluseason'] == chosen['fluseason']) &
                (all_datasets_df['sample'] == chosen['sample']) &
                (all_datasets_df['location_code'] == missing_location)
            ][['season_week', 'value', 'week_enddate']].copy()
            
            source = f"same_location_{chosen['datasetH1']}_year_{chosen['fluseason']}"
            return fill_data, source
    
    # Strategy 3: Fallback to random location from current frame
    random_location = np.random.choice(list(actual_locations))
    random_data = frame[frame['location_code'] == random_location].copy()
    return random_data, f"random_from_{random_location}"


def _preserve_columns(target_df: pd.DataFrame, source_df: pd.DataFrame, 
                     location_data: pd.DataFrame) -> None:
    """Preserve columns from source dataframe in target dataframe."""
    for col in source_df.columns:
        if col not in target_df.columns and not location_data.empty:
            target_df[col] = location_data[col].iloc[0]
        elif col == 'origin' and col in target_df.columns and not location_data.empty:
            # Special handling for origin: only overwrite if target has NaN values
            # This preserves intelligent filling origins while filling missing ones
            target_has_nan = target_df[col].isna().any()
            if target_has_nan:
                # Only fill NaN values, preserve existing filled origins
                mask = target_df[col].isna()
                target_df.loc[mask, col] = location_data[col].iloc[0]


def _pad_single_location(frame: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Pads a location-specific epidemic time series to ensure complete weekly coverage.
    
    Fills in missing weeks:
    - Weeks before first/after last observation: filled with zeros
    - Weeks between observations: filled using previous week's value
    
    Args:
        frame (pd.DataFrame): DataFrame containing epidemic data for a single location
        location (str): Location identifier for the current frame
    
    Returns:
        pd.DataFrame: Padded DataFrame with entries for all weeks 1-53
    """
    # Handle empty input
    if frame.empty:
        # Create empty frame for all weeks - metadata will be preserved by _preserve_columns later
        missing_data = [{
            "season_week": week,
            "location_code": location,
            "value": 0
        } for week in range(1, 54)]
        return pd.DataFrame(missing_data)
        
    # Get min/max weeks if data exists
    min_week = frame["season_week"].min()
    max_week = frame["season_week"].max()
        
    all_weeks = set(range(1, 54))
    missing_weeks = sorted(list(all_weeks - set(frame["season_week"])))
    
    # Early return if no missing weeks
    if not missing_weeks:
        return frame.sort_values("season_week").reset_index(drop=True)
    
    # Extract metadata from first row to preserve in missing rows
    metadata = {}
    if not frame.empty:
        first_row = frame.iloc[0]
        for col in frame.columns:
            if col not in ["season_week", "location_code", "value"]:
                metadata[col] = first_row[col]
        
    # Batch create missing rows for better performance
    missing_rows = []
    for week in missing_weeks:
        # Determine fill value based on position
        if week < min_week or week > max_week:
            new_value = 0  # External gaps filled with zeros
        else:
            # Internal gaps filled with previous week's value
            previous_week = frame[frame["season_week"] == week-1]
            new_value = previous_week["value"].values[0] if not previous_week.empty else 0

        missing_row = {
            "season_week": week,
            "location_code": location,
            "value": new_value,
            **metadata  # Include all metadata from original frame
        }
        missing_rows.append(missing_row)
    
    # Single concat instead of multiple
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        frame = pd.concat([frame, missing_df], ignore_index=True)

    return frame.sort_values("season_week").reset_index(drop=True)


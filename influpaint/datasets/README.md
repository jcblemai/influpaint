# InfluPaint Datasets Module

The datasets module provides tools for epidemic data processing, augmentation, and frame construction for training diffusion models on epidemiological surveillance data.

## Overview

This module addresses common challenges in epidemic modeling:
- **Dataset Rebalancing**: Weight multiple data sources for target proportions
- **Temporal Completeness**: Ensure all frames have complete weekly coverage (1-53)
- **Spatial Completeness**: Fill missing location-season combinations
- **Intelligent Gap Filling**: Handle missing data with different fill strategies

## Core Components

### 1. Dataset Mixer (`mixer.py`)

The main component for combining multiple surveillance datasets into unified training frames.

**Key Features:**
- **Hierarchical Data Structure**: Handles 4-level hierarchy (H1 → H2 → Season → Sample)
- **Multiple Fill Strategies**: Error, zeros, random intelligent filling, or skip
- **Origin Tracking**: Maintains provenance of all data points
- **Randomized Augmentation**: Each multiplied frame gets different random fills

**Example Usage:**
```python
from influpaint.datasets import mixer as dataset_mixer
from influpaint.utils import SeasonAxis

# Configure dataset mixing
config = {
    "fluview": {"multiplier": 2},        # Include twice
    "SMH_R4-R5": {"multiplier": 1}       # Include once
}

season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)

# Build complete frames with intelligent filling
frames = dataset_mixer.build_frames(
    all_datasets_df, 
    config, 
    season_setup, 
    fill_missing_locations="random"
)
```

**Configuration Options:**

*Multiplier-based (explicit replication):*
```python
config = {
    "fluview": {"multiplier": 2},
    "smh_traj": {"multiplier": 1}
}
```

*Proportion-based (target composition):*
```python
config = {
    "fluview": {"proportion": 0.7, "total": 1000},    # 70% of final dataset
    "smh_traj": {"proportion": 0.3, "total": 1000}   # 30% of final dataset
}
```

**Fill Strategies:**
- `"error"`: Fail if any expected location is missing
- `"zeros"`: Fill missing locations with zeros
- `"random"`: Intelligent hierarchical filling with randomization
- `"skip"`: Skip frames with missing locations

### 2. Intelligent Location Filling

The `random` fill strategy uses a sophisticated hierarchy:

1. **Priority 1**: Same H1 dataset, different year → random year + random sample
2. **Priority 2**: Same H1 dataset, different sample → random sample from same year  
3. **Priority 3**: Different H1 dataset → random H1 + random year

**Randomization Behavior:**
Each multiplied frame is processed independently. When location filling is needed, `np.random.choice()` ensures different random choices across multiplied frames, providing excellent data augmentation for training.

Example with 10 copies of the same frame missing location '11':
- Copy 1: filled with `round4_USC-SIkJalpha_B-2023-08-14/sample_15`
- Copy 2: filled with `round5_NotreDame-FRED_D-2023-08-14/sample_8`
- Copy 3: filled with `round4_MOBS_NEU-GLEAM_FLU_C-2023-08-14/sample_22`
- etc. (each gets different random fill data)

### 3. Data Loaders (`loaders.py`)

Load and transform epidemic data for training.

### 4. Source Readers (`read_datasources.py`)

Read epidemic data from various surveillance sources.

## Data Format

**Input DataFrame Requirements:**
- `datasetH1`: Top-level dataset category (e.g., 'fluview', 'smh_traj')
- `datasetH2`: Sub-dataset within H1 (e.g., 'round4_CADPH-FluCAT_A-2024-08-01')
- `fluseason`: Flu season year
- `sample`: Sample identifier within each H2/season combination
- `location_code`: Geographic location identifier
- `season_week`: Week number (1-53)
- `value`: Epidemic measurement value
- `week_enddate`: Week ending date

**Output Frame Structure:**
Each frame contains:
- All weeks (1-53) for all expected locations
- Origin column tracking source: `"H1/H2/season/sample"` or `"H1/H2/season/sample[filled_source]"`
- Consistent data structure ready for array conversion

## Testing

The module includes tests:

### Fast Frame 880 Regression Test
Specifically targets the duplicate bug discovered in frame 880:
```bash
python influpaint/datasets/test/test_frame_880_duplicate_bug.py
```

Tests:
- ✅ No duplicate weeks for filled locations (was 131 rows, now 53)
- ✅ Proper intelligent location filling
- ✅ Randomization across multiplied frames

### Full Test Suite
```bash
pytest influpaint/datasets/test/test_dataset_mixer_pytest.py
```

Tests all functionality:
- Basic frame building
- All fill strategies (error, zeros, random, skip)
- Multiplier and proportion configurations
- Performance benchmarks
- Data integrity and origin tracking

## Bug Fixes

### Frame 880 Duplicate Bug (Fixed)
**Issue**: Global lookup table was returning multiple H2/sample combinations instead of selecting one unique combination, causing duplicate rows.

**Fix**: Modified lookup table building to explicitly pick first unique H2/sample combination:
```python
unique_combos = year_data.groupby(['datasetH2', 'sample']).first().reset_index()
```

**Result**: 
- Before: Frame 880 had 131+ duplicate rows for location '11'
- After: Frame 880 has exactly 53 rows (one per week)

## Performance

- **Global Lookup Table**: Pre-computed once for fast intelligent filling
- **Batch Operations**: Vectorized operations for missing data creation
- **Progress Tracking**: Real-time progress bars for large datasets
- **Memory Efficient**: Processes frames incrementally

## Integration

This module integrates with:
- **Training Pipeline**: Outputs frames ready for diffusion model training
- **Season Axis**: Uses `influpaint.utils.SeasonAxis` for location definitions
- **Data Converters**: Works with `influpaint.utils.converters` for array conversion
- **Plotting**: Compatible with `influpaint.utils.plotting` for visualization

## Example Workflow

```python
# 1. Load surveillance datasets
all_datasets_df = pd.read_parquet("Flusight/flu-datasets/all_datasets.parquet")

# 2. Configure mixing strategy
config = {
    "fluview": {"proportion": 0.7, "total": 3000},
    "SMH_R4-R5": {"proportion": 0.3, "total": 3000}
}

# 3. Build frames with intelligent filling
frames = dataset_mixer.build_frames(
    all_datasets_df, config, season_setup, 
    fill_missing_locations="random"
)

# 4. Convert to training arrays
all_frames_df = pd.concat(frames).reset_index(drop=True)
array_list = converters.dataframe_to_arraylist(
    df=all_frames_df, season_setup=season_setup
)

# 5. Save as training dataset
array = np.array(array_list)
flu_payload_array = xr.DataArray(array, dims=["sample", "feature", "season_week", "place"])
flu_payload_array.to_netcdf("training_datasets/TS_dataset_2024-07-15.nc")
```

This provides a complete pipeline from raw surveillance data to training-ready epidemic frames.
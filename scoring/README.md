# Scoring Package - Forecast Evaluation System

This package provides tools for evaluating forecast performance with standardized scoring metrics and visualizations.

## Quick Start

### 1. Add a Forecast Record

```python
import scoring.evaluation as eval_mod
import pandas as pd

# Your forecast data as a DataFrame (see format details below)
forecast_df = pd.read_csv("your_forecast.csv")  # Hubverse format

# Create a forecast record
record = eval_mod.ForecastRecord(
    model="i806::celebahq",              # Model identifier
    group="influpaint",                  # Group for visualization (e.g., "influpaint", "flusight") 
    display_name="i806\n::celebahq",     # How to display in plots
    forecast_date=pd.Timestamp("2023-10-14"),
    df=forecast_df                       # Your forecast DataFrame
)
```

### 2. Score Multiple Models

```python
# Collect multiple forecast records
records = [record1, record2, record3, ...]

# Create dataset and compute scores
dataset = eval_mod.ForecastDataset(records)
scores, missing_counts = eval_mod.score_dataset(dataset, ground_truth_df, expected_dates)

# Optional: Add relative scores (same format, just normalized by baseline)
relative_scores = eval_mod.compute_relative_scores(scores, "FluSight-baseline")
```

### 3. Visualize Results

```python
import scoring.plotting as plot_mod

# Define colors for model groups
colors = {"influpaint": "green", "flusight": "blue"}

# Create performance heatmap
plot_mod.forecast_scores_heatmap(
    scores_df=scores,
    dataset=dataset,
    group_colors=colors,
    title="Forecast Performance",
    filename="heatmap.png",
    save_dir="results/"
)
```

## Complete Example

Here's a full workflow showing how to evaluate multiple models:

```python
import scoring.evaluation as eval_mod
import scoring.plotting as plot_mod
import pandas as pd

# 1. Load your forecast data (must be in Hubverse format - see below)
influpaint_df = pd.read_csv("influpaint_forecast.csv")
flusight_df = pd.read_csv("flusight_forecast.csv") 
ground_truth_df = pd.read_csv("ground_truth.csv")

# 2. Create forecast records
records = [
    eval_mod.ForecastRecord(
        model="i806::celebahq",
        group="influpaint",
        display_name="i806\n::celebahq", 
        forecast_date=pd.Timestamp("2023-10-14"),
        df=influpaint_df
    ),
    eval_mod.ForecastRecord(
        model="FluSight-baseline",
        group="flusight",
        display_name="FluSight-baseline",
        forecast_date=pd.Timestamp("2023-10-14"), 
        df=flusight_df
    )
]

# 3. Score all forecasts
dataset = eval_mod.ForecastDataset(records)
scores, missing_counts = eval_mod.score_dataset(dataset, ground_truth_df, expected_dates)

# 4. Create visualizations
colors = {"influpaint": "green", "flusight": "blue"}

# Absolute performance heatmap
plot_mod.forecast_scores_heatmap(
    scores_df=scores,
    dataset=dataset,
    group_colors=colors,
    title="Absolute WIS Scores",
    filename="heatmap_absolute.png", 
    save_dir="results/",
    missing_counts=missing_counts,  # Shows missing forecasts
    scoring_metric="wis_total",
    top_n=5  # Top 5 models per group
)

# Relative performance (optional - same interface, different metric)
relative_scores = eval_mod.compute_relative_scores(scores, "FluSight-baseline")
plot_mod.forecast_scores_heatmap(
    scores_df=relative_scores,  # Different data, same function
    dataset=dataset,
    group_colors=colors,
    title="Relative WIS (vs Baseline)",
    filename="heatmap_relative.png",
    save_dir="results/",
    scoring_metric="wis_total",  # Same metric name
    top_n=5  # Same top_n behavior
)
```

## Data Formats

### Input: Hubverse CSV Format

Forecast data must follow the [Hubverse format](https://hubdocs.readthedocs.io/en/latest/quickstart-hub-admin/data-formats.html):

```csv
reference_date,target,horizon,target_end_date,location,output_type,output_type_id,value
2023-10-14,wk inc flu hosp,0,2023-10-14,01,quantile,0.01,5.2
2023-10-14,wk inc flu hosp,0,2023-10-14,01,quantile,0.025,6.1
2023-10-14,wk inc flu hosp,0,2023-10-14,01,quantile,0.05,7.3
...
2023-10-14,wk inc flu hosp,1,2023-10-21,01,quantile,0.975,8.9
2023-10-14,wk inc flu hosp,1,2023-10-21,01,quantile,0.99,9.4
```

**Required Columns:**
- `reference_date`: Date when forecast was made
- `target`: Target variable (e.g., "wk inc flu hosp")  
- `horizon`: Forecast horizon in weeks (0, 1, 2, 3, ...)
- `target_end_date`: Date the forecast targets (epidemiological week ending)
- `location`: FIPS codes as strings ("01", "02", ..., "US")
- `output_type`: Must be `"quantile"` (point forecasts ignored)
- `output_type_id`: Quantile levels (0.01, 0.025, 0.05, ..., 0.95, 0.975, 0.99)
- `value`: Predicted value at that quantile

### Ground Truth Format

Ground truth observations:
```csv
date,location,value
2023-10-14,01,125.4
2023-10-14,02,89.2
2023-10-14,US,2341.7
```

### Output: Scoring Dataframe

Both `score_dataset()` and `compute_relative_scores()` return the same tidy dataframe format:

```python
# Columns: model, forecast_date, target, target_end_date, scoring_metric, location, value

# Absolute scores
model              forecast_date  target       target_end_date  scoring_metric      location  value
i806::celebahq     2023-10-14     0 wk ahead   2023-10-14       wis_total          01        12.5
i806::celebahq     2023-10-14     0 wk ahead   2023-10-14       wis_sharpness      01        8.2
FluSight-baseline  2023-10-14     0 wk ahead   2023-10-14       wis_total          01        15.8

# Relative scores (same format, normalized by baseline)
i806::celebahq     2023-10-14     0 wk ahead   2023-10-14       wis_total          01        0.79  # 12.5/15.8
Another-model      2023-10-14     0 wk ahead   2023-10-14       wis_total          01        1.12  # Worse than baseline
```

**Key Point**: Relative scores are just absolute scores divided by the baseline model's score. Same interface, same plotting functions!

## Scoring Metrics

### Weighted Interval Score (WIS)

WIS follows the [standard definition](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618):

```python
# 1. Extract quantile forecasts at standard levels
quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

# 2. Form symmetric prediction intervals
# E.g., 90% interval uses quantiles 0.05 and 0.95
alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 3. Compute interval score for each alpha level
for alpha in alphas:
    lower = quantile[alpha/2]      # e.g., 0.05 for 90% interval  
    upper = quantile[1-alpha/2]    # e.g., 0.95 for 90% interval
    width = upper - lower          # Sharpness component
    
    if observation < lower:
        penalty = (2/alpha) * (lower - observation)  # Overprediction
    elif observation > upper:
        penalty = (2/alpha) * (observation - upper)   # Underprediction  
    else:
        penalty = 0
    
    interval_score = width + penalty

# 4. Weight by alpha/2 and average
weights = [alpha/2 for alpha in alphas]
WIS = sum(weight * score for weight, score in zip(weights, interval_scores)) / sum(weights)
```

**WIS Components:**
- `wis_total`: Overall Weighted Interval Score (lower = better)
- `wis_sharpness`: Average interval width (narrower intervals = better)
- `wis_overprediction`: Penalty when observation < lower bounds
- `wis_underprediction`: Penalty when observation > upper bounds

## Key Implementation Details

### Top-N Model Selection

All plotting functions use the same `top_n` logic - **top N per group**, not globally:

```python
# Example: top_n=5 
# Result: Top 5 InfluPaint + Top 5 FluSight = 10 models total

# For absolute scores: best = lowest values  
# For relative scores: best = closest to 1.0 (baseline performance)
```

This ensures fair representation of both groups in all visualizations.

### Expected Dates Parameter

The `expected_dates` parameter controls which dates are considered when counting missing forecasts:

```python
# Season-specific evaluation: only count missing within season dates
season_dates = [date(2023, 10, 14), date(2023, 10, 21), date(2023, 10, 28)]
scores, missing_counts = eval_mod.score_dataset(dataset, ground_truth_df, season_dates)

# Combined seasons: use all dates from all seasons
all_dates = season1_dates + season2_dates + season3_dates
scores, missing_counts = eval_mod.score_dataset(dataset, ground_truth_df, all_dates)

# Default behavior: use all ground truth dates (may overcount missing forecasts)
scores, missing_counts = eval_mod.score_dataset(dataset, ground_truth_df)
```


### Location Filtering

Two aggregation modes supported:

```python
location_filter="US"    # Single location (e.g., US national)
location_filter="ALL"   # Sum across all available locations
```

### Model Display Names

InfluPaint models get line breaks for readability:

```python
# Full name: "i806::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_No::celebahq"
# Display:   "i806::m_U500cRx124::ds_30S70M\ntr_Sqrt::ri_No::celebahq"
#           (line break every 3 components)

# FluSight models kept as-is: "CEPH-Rtrend_fluH", "FluSight-baseline"
```

### Time Series Visualization

Time series plots use multiple visual differentiators for models within the same group:

```python
# Each model gets:
# 1. Different marker shapes: 'o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '8'
# 2. Color tone variations: darker/lighter versions of group base color
# 3. White marker edges for better visibility
# 4. Dashed lines for models with missing forecasts

plot_mod.forecast_performance_timeseries(
    scores_df=scores,
    dataset=dataset,
    group_colors=colors,
    missing_counts=missing_counts,  # Include missing data patterns
    title="Performance Over Time",
    filename="timeseries.png",
    save_dir="results/"
)
```

**Visual Differentiation Strategy:**
- **Group 1 (e.g., InfluPaint)**: Green base color → green circle, dark green square, light green triangle, etc.
- **Group 2 (e.g., FluSight)**: Blue base color → blue circle, dark blue square, light blue triangle, etc.
- **Missing data**: Dashed lines with transparency, labeled "(missing:N)" in legend

## Validation and Error Handling

The scoring package includes comprehensive validation to catch data format issues early:

### Automatic Validation

When you call `score_dataset()`, the following validations are performed automatically:

```python
# This will raise ValidationError if any issues are detected
scores = eval_mod.score_dataset(dataset, ground_truth_df)
```

**Validation Checks:**
- ✅ **Hubverse format compliance**: Required columns, quantile output_type
- ✅ **Quantile completeness**: All 23 standard quantile levels present
- ✅ **Date alignment**: target_end_date values match ground truth dates  
- ✅ **Location alignment**: Forecast locations exist in ground truth
- ✅ **Group consistency**: Same models not assigned to different groups
- ✅ **Display names**: Non-empty display_name fields in all records

### Manual Validation

You can also run individual validation functions:

```python
import scoring.validation as validation

try:
    validation.validate_hubverse_format(forecast_df, "MyModel")
    validation.validate_quantile_completeness(forecast_df, "MyModel") 
    validation.validate_date_alignment(forecast_df, ground_truth_df, "MyModel")
    validation.validate_location_alignment(forecast_df, ground_truth_df, "MyModel")
    print("✓ All validations passed")
except validation.ValidationError as e:
    print(f"❌ Validation failed: {e}")
```

### Common Validation Errors

**Missing Quantile Levels:**
```
ValidationError: MyModel missing required quantile levels: [0.01, 0.025, 0.05]
```

**Date Misalignment:**  
```
ValidationError: MyModel target_end_date values do not align with ground truth dates.
Forecast dates: [2023-10-14, 2023-10-21]..., Ground truth dates: [2023-10-15, 2023-10-22]...
```

**Location Misalignment:**
```
ValidationError: MyModel locations do not align with ground truth.
Forecast locations: ['01', '02', '03']..., Ground truth locations: ['1', '2', '3']...
```

## Module Reference

### scoring.evaluation

**Data Structures:**
- `ForecastRecord`: Single model forecast for one date
- `ForecastDataset`: Collection of forecast records
- `ModelGroupConfig`: Configuration for model grouping (optional)

**Functions:**
- `score_dataset(dataset, ground_truth, expected_dates=None)`: Compute WIS scores with validation - Returns (scores, missing_counts)
- `compute_relative_scores(scores, baseline_model)`: Relative performance vs baseline

### scoring.plotting  

**Functions:**
- `forecast_scores_heatmap()`: Heatmap of scores across models and dates
- `forecast_components_breakdown()`: Component breakdown (sharpness, over/under prediction)
- `forecast_performance_timeseries()`: Performance over time with 2x2 horizon subplots

### scoring.validation

**Exception:**
- `ValidationError`: Custom exception for validation failures

**Functions:**
- `validate_hubverse_format(df, context)`: Check required columns and format
- `validate_quantile_completeness(df, context)`: Verify all quantile levels present
- `validate_ground_truth_format(df)`: Check ground truth format
- `validate_date_alignment(forecasts_df, gt_df, context)`: Check date alignment
- `validate_location_alignment(forecasts_df, gt_df, context)`: Check location alignment
- `validate_forecast_record(record, context)`: Validate single ForecastRecord
- `validate_forecast_dataset(dataset, gt_df)`: Comprehensive dataset validation
- `validate_group_consistency(dataset)`: Check model group assignments

### scoring.weighted_interval_score

**Functions:**
- `weighted_interval_score_fast()`: Core WIS computation with components
- `score_Nwk_forecasts_hub()`: Hub-format WIS scoring wrapper

## Extending with New Metrics

To add new scoring metrics (e.g., RMSE, MAE):

```python
# 1. Implement scoring function that returns same tidy format
def compute_rmse_scores(forecasts_df, ground_truth_df):
    # ... implementation
    return pd.DataFrame({
        'model': [...],
        'forecast_date': [...], 
        'target': [...],
        'target_end_date': [...],
        'scoring_metric': 'rmse',  # New metric name
        'location': [...],
        'value': [...]  # RMSE values
    })

# 2. Integrate into score_dataset() 
# 3. Use in any plotting function
plot_mod.forecast_scores_heatmap(..., scoring_metric="rmse")
```

The tidy dataframe format makes the system extensible to any scoring metric that can be computed per model/date/location.
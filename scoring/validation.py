"""
Validation Module - Data format and consistency checks.

This module provides validation functions to catch common data format issues
and configuration errors before they cause silent failures in scoring.
"""

from typing import Set, List, Dict, Optional
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_hubverse_format(df: pd.DataFrame, context: str = "forecast") -> None:
    """
    Validate that DataFrame follows Hubverse format requirements.
    
    Args:
        df: DataFrame to validate
        context: Description for error messages (e.g., "forecast", "model X")
    
    Raises:
        ValidationError: If format requirements are not met
    """
    required_columns = [
        'target_end_date', 'location', 'output_type', 'output_type_id', 
        'value', 'target', 'horizon'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"{context} missing required columns: {missing_cols}"
        )
    
    # Check output_type is quantile only
    if 'output_type' in df.columns:
        non_quantile = df[df['output_type'] != 'quantile']
        if not non_quantile.empty:
            unique_types = df['output_type'].unique()
            raise ValidationError(
                f"{context} contains non-quantile output_type values: {unique_types}. "
                "Only 'quantile' forecasts are supported."
            )


def validate_quantile_completeness(df: pd.DataFrame, context: str = "forecast") -> None:
    """
    Validate that all required quantile levels are present.
    
    Args:
        df: DataFrame with quantile forecasts
        context: Description for error messages
        
    Raises:
        ValidationError: If quantile levels are incomplete
    """
    expected_quantiles = {
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
    }
    
    if 'output_type_id' not in df.columns:
        raise ValidationError(f"{context} missing 'output_type_id' column")
    
    # Convert to numeric and get unique quantile levels
    df_clean = df.copy()
    df_clean['output_type_id'] = pd.to_numeric(df_clean['output_type_id'], errors='coerce')
    
    # Check for NaN values from conversion
    nan_quantiles = df_clean['output_type_id'].isna()
    if nan_quantiles.any():
        raise ValidationError(
            f"{context} contains non-numeric quantile levels in 'output_type_id'"
        )
    
    actual_quantiles = set(df_clean['output_type_id'].unique())
    missing_quantiles = expected_quantiles - actual_quantiles
    
    if missing_quantiles:
        raise ValidationError(
            f"{context} missing required quantile levels: {sorted(missing_quantiles)}"
        )


def validate_ground_truth_format(df: pd.DataFrame) -> None:
    """
    Validate ground truth DataFrame format.
    
    Args:
        df: Ground truth DataFrame
        
    Raises:
        ValidationError: If format requirements are not met
    """
    required_columns = ['date', 'location', 'value']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValidationError(
            f"Ground truth missing required columns: {missing_cols}"
        )


def validate_date_alignment(forecasts_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                          context: str = "forecast") -> None:
    """
    Validate that forecast target_end_date values align with ground truth dates.
    
    Args:
        forecasts_df: Forecast DataFrame
        ground_truth_df: Ground truth DataFrame
        context: Description for error messages
        
    Raises:
        ValidationError: If no dates align or format issues detected
    """
    if 'target_end_date' not in forecasts_df.columns:
        raise ValidationError(f"{context} missing 'target_end_date' column")
    
    if 'date' not in ground_truth_df.columns:
        raise ValidationError("Ground truth missing 'date' column")
    
    # Normalize date formats
    try:
        forecast_dates = pd.to_datetime(forecasts_df['target_end_date']).dt.date
        gt_dates = pd.to_datetime(ground_truth_df['date']).dt.date
    except Exception as e:
        raise ValidationError(
            f"Date parsing failed - ensure dates are valid format: {e}"
        )
    
    # Check for alignment
    forecast_dates_set = set(forecast_dates.unique())
    gt_dates_set = set(gt_dates.unique())
    aligned_dates = forecast_dates_set & gt_dates_set
    
    if not aligned_dates:
        raise ValidationError(
            f"{context} target_end_date values do not align with ground truth dates. "
            f"Forecast dates: {sorted(list(forecast_dates_set)[:5])}..., "
            f"Ground truth dates: {sorted(list(gt_dates_set)[:5])}..."
        )


def validate_location_alignment(forecasts_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                              context: str = "forecast") -> None:
    """
    Validate that forecast locations align with ground truth locations.
    
    Args:
        forecasts_df: Forecast DataFrame
        ground_truth_df: Ground truth DataFrame
        context: Description for error messages
        
    Raises:
        ValidationError: If no locations align
    """
    if 'location' not in forecasts_df.columns:
        raise ValidationError(f"{context} missing 'location' column")
    
    if 'location' not in ground_truth_df.columns:
        raise ValidationError("Ground truth missing 'location' column")
    
    # Normalize location formats (convert to string, strip whitespace)
    forecast_locations = set(forecasts_df['location'].astype(str).str.strip().unique())
    gt_locations = set(ground_truth_df['location'].astype(str).str.strip().unique())
    
    aligned_locations = forecast_locations & gt_locations
    
    if not aligned_locations:
        raise ValidationError(
            f"{context} locations do not align with ground truth. "
            f"Forecast locations: {sorted(list(forecast_locations)[:5])}..., "
            f"Ground truth locations: {sorted(list(gt_locations)[:5])}..."
        )


def validate_forecast_record(record, context: Optional[str] = None) -> None:
    """
    Validate a single ForecastRecord for completeness and consistency.
    
    Args:
        record: ForecastRecord instance
        context: Optional context for error messages
        
    Raises:
        ValidationError: If record is invalid
    """
    if context is None:
        context = f"ForecastRecord({record.model})"
    
    # Check required fields are not empty
    if not record.model or not record.model.strip():
        raise ValidationError(f"{context} has empty model name")
    
    if not record.group or not record.group.strip():
        raise ValidationError(f"{context} has empty group name")
    
    if not record.display_name or not record.display_name.strip():
        raise ValidationError(f"{context} has empty display_name")
    
    if record.df is None or record.df.empty:
        raise ValidationError(f"{context} has empty forecast DataFrame")
    
    # Validate the forecast DataFrame format
    validate_hubverse_format(record.df, context)


def validate_forecast_dataset(dataset, ground_truth_df: pd.DataFrame, 
                             expected_dates: Optional[List] = None) -> Dict[str, int]:
    """
    Validate ForecastDataset for consistency across records and count missing data.
    
    Args:
        dataset: ForecastDataset instance
        ground_truth_df: Ground truth DataFrame for alignment checks
        expected_dates: Optional list of expected forecast dates. If None, uses all ground truth dates.
        
    Returns:
        Dict[str, int]: Dictionary mapping model names to missing forecast counts
        
    Raises:
        ValidationError: If dataset is invalid or inconsistent
    """
    if not dataset.records:
        raise ValidationError("ForecastDataset contains no records")
    
    # Validate each record individually
    for i, record in enumerate(dataset.records):
        validate_forecast_record(record, f"Record {i} ({record.model})")
    
    # Check for duplicate model-date combinations
    model_date_pairs = [(r.model, r.forecast_date) for r in dataset.records]
    unique_pairs = set(model_date_pairs)
    
    if len(model_date_pairs) != len(unique_pairs):
        raise ValidationError("ForecastDataset contains duplicate model-date combinations")
    
    # Validate ground truth format
    validate_ground_truth_format(ground_truth_df)
    
    # Check date and location alignment for all records and count missing data
    missing_counts = {}
    
    # Get expected forecast dates
    if expected_dates is not None and len(expected_dates) > 0:
        # Use provided expected dates
        if hasattr(expected_dates[0], 'date'):
            # Convert datetime objects to date objects
            gt_dates = set(d.date() if hasattr(d, 'date') else d for d in expected_dates)
        else:
            # Convert string dates to date objects
            gt_dates = set(pd.to_datetime(expected_dates).date)
    else:
        # Fall back to all ground truth dates
        gt_dates = set(pd.to_datetime(ground_truth_df['date']).dt.date)
    
    # Group records by model to count missing dates
    model_records = {}
    for record in dataset.records:
        if record.model not in model_records:
            model_records[record.model] = []
        model_records[record.model].append(record)
    
    for model, records in model_records.items():
        # Validate each record
        for record in records:
            validate_date_alignment(record.df, ground_truth_df, f"Model {record.model}")
            validate_location_alignment(record.df, ground_truth_df, f"Model {record.model}")
            validate_quantile_completeness(record.df, f"Model {record.model}")
        
        # Count missing forecast dates
        model_dates = set(record.forecast_date.date() for record in records)
        missing_dates = gt_dates - model_dates
        missing_counts[model] = len(missing_dates)
    
    return missing_counts


def validate_group_consistency(dataset) -> None:
    """
    Validate that model group assignments are consistent.
    
    Args:
        dataset: ForecastDataset instance
        
    Raises:
        ValidationError: If same model appears in different groups
    """
    model_groups = {}
    
    for record in dataset.records:
        if record.model in model_groups:
            if model_groups[record.model] != record.group:
                raise ValidationError(
                    f"Model '{record.model}' assigned to multiple groups: "
                    f"'{model_groups[record.model]}' and '{record.group}'"
                )
        else:
            model_groups[record.model] = record.group
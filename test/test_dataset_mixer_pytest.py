#!/usr/bin/env python3
"""
Pytest-compatible tests for dataset_mixer.py

Tests the intelligent location filling, performance, and data integrity.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import pytest

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import dataset_mixer
from season_axis import SeasonAxis


@pytest.fixture
def test_df():
    """Load test dataset."""
    test_path = current_dir / 'test_dataset.parquet'
    if not test_path.exists():
        pytest.skip(f"Test dataset not found at {test_path}. Run create_test_dataset.py first.")
    return pd.read_parquet(test_path)


@pytest.fixture
def season_setup():
    """Create season setup."""
    return SeasonAxis.for_flusight(remove_us=True, remove_territories=True)


def test_basic_functionality(test_df, season_setup):
    """Test basic frame building functionality."""
    config = {"fluview": {"multiplier": 1}}
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="zeros"
    )
    
    assert frames, "No frames returned"
    assert len(frames) == 2, f"Expected 2 frames, got {len(frames)}"  # 2010 and 2021
    
    first_frame = frames[0]
    expected_cols = ['location_code', 'season_week', 'value', 'week_enddate', 'origin']
    assert list(first_frame.columns) == expected_cols, f"Wrong columns. Expected {expected_cols}, got {list(first_frame.columns)}"


def test_intelligent_filling(test_df, season_setup):
    """Test that intelligent filling works correctly."""
    config = {"fluview": {"multiplier": 1}}
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="random"
    )
    
    # Find the 2010 frame (should have missing locations 12, 22)
    frame_2010 = None
    for frame in frames:
        if '2010' in frame['origin'].iloc[0]:
            frame_2010 = frame
            break
    
    assert frame_2010 is not None, "Could not find 2010 frame"
    
    # Check that locations 12 and 22 are present (filled)
    frame_locations = set(frame_2010['location_code'].unique())
    assert '12' in frame_locations, f"Location 12 not filled. Frame has: {frame_locations}"
    assert '22' in frame_locations, f"Location 22 not filled. Frame has: {frame_locations}"
    
    # Check that filled locations have intelligent origins
    loc_12_data = frame_2010[frame_2010['location_code'] == '12']
    loc_22_data = frame_2010[frame_2010['location_code'] == '22']
    
    origin_12 = loc_12_data['origin'].iloc[0]
    origin_22 = loc_22_data['origin'].iloc[0]
    
    # Should have intelligent filling origins from 2021
    assert 'same_location_year_2021' in origin_12, f"Location 12 not filled intelligently. Origin: {origin_12}"
    assert 'same_location_year_2021' in origin_22, f"Location 22 not filled intelligently. Origin: {origin_22}"


def test_error_strategy(test_df, season_setup):
    """Test error strategy for missing locations."""
    config = {"fluview": {"multiplier": 1}}
    
    # Error strategy should fail with missing locations
    with pytest.raises(ValueError, match="Missing required locations"):
        dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="error"
        )


def test_zeros_strategy(test_df, season_setup):
    """Test zeros strategy for missing locations."""
    config = {"fluview": {"multiplier": 1}}
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="zeros"
    )
    
    # Find 2010 frame and check zero-filled locations
    frame_2010 = next(f for f in frames if '2010' in f['origin'].iloc[0])
    loc_12_zeros = frame_2010[frame_2010['location_code'] == '12']
    
    assert (loc_12_zeros['value'] == 0.0).all(), "Zeros strategy didn't fill with zeros"
    assert 'filled_zeros' in loc_12_zeros['origin'].iloc[0], "Origin should indicate zero filling"


def test_skip_strategy(test_df, season_setup):
    """Test skip strategy for missing locations."""
    config = {"fluview": {"multiplier": 1}}
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="skip"
    )
    
    # Should have fewer frames (2010 frame should be skipped)
    assert len(frames) == 1, f"Skip strategy should return 1 frame, got {len(frames)}"
    
    # Remaining frame should be 2021
    remaining_frame = frames[0]
    assert '2021' in remaining_frame['origin'].iloc[0], "Remaining frame should be from 2021"


def test_multiplier_configs(test_df, season_setup):
    """Test multiplier configurations."""
    config = {
        "fluview": {"multiplier": 2},
        "flusurv": {"multiplier": 1}
    }
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="zeros"
    )
    
    # Should have: 2 fluview frames (2010, 2021) * 2 + 1 flusurv frame = 5 frames
    assert len(frames) == 5, f"Expected 5 frames with multipliers, got {len(frames)}"
    
    # Count origins
    fluview_count = sum(1 for f in frames if 'fluview' in f['origin'].iloc[0])
    flusurv_count = sum(1 for f in frames if 'flusurv' in f['origin'].iloc[0])
    
    assert fluview_count == 4, f"Expected 4 fluview frames, got {fluview_count}"  # 2 original * 2 multiplier
    assert flusurv_count == 1, f"Expected 1 flusurv frame, got {flusurv_count}"


def test_proportion_configs(test_df, season_setup):
    """Test proportion-based configurations."""
    config = {
        "fluview": {"proportion": 0.8, "total": 10},
        "flusurv": {"proportion": 0.2, "total": 10}
    }
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="zeros"
    )
    
    assert len(frames) > 0, "No frames returned for proportion config"
    
    # Check that we have frames from both datasets
    has_fluview = any('fluview' in f['origin'].iloc[0] for f in frames)
    has_flusurv = any('flusurv' in f['origin'].iloc[0] for f in frames)
    
    assert has_fluview, "No fluview frames found"
    assert has_flusurv, "No flusurv frames found"


def test_performance(test_df, season_setup):
    """Test that performance is acceptable."""
    config = {
        "fluview": {"multiplier": 2},
        "flusurv": {"multiplier": 2}
    }
    
    start_time = time.time()
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="random"
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    assert duration < 5.0, f"Performance too slow: {duration:.2f} seconds"
    assert len(frames) > 0, "No frames returned"


def test_column_cleanup(test_df, season_setup):
    """Test that unnecessary columns are removed."""
    config = {"fluview": {"multiplier": 1}}
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="zeros"
    )
    
    expected_cols = ['location_code', 'season_week', 'value', 'week_enddate', 'origin']
    
    for i, frame in enumerate(frames):
        assert list(frame.columns) == expected_cols, f"Frame {i} has wrong columns: {list(frame.columns)}"
        
        # Check for NaN values
        for col in frame.columns:
            nan_count = frame[col].isna().sum()
            assert nan_count == 0, f"Frame {i} has {nan_count} NaN values in column {col}"


def test_origin_tracking(test_df, season_setup):
    """Test that origin tracking is accurate."""
    config = {"fluview": {"multiplier": 1}}
    
    frames = dataset_mixer.build_frames(
        test_df, config, season_setup, fill_missing_locations="random"
    )
    
    frame_2010 = next(f for f in frames if '2010' in f['origin'].iloc[0])
    
    # Check original data origins
    original_data = frame_2010[~frame_2010['origin'].str.contains('filled', na=False)]
    for origin in original_data['origin'].unique():
        assert origin == 'fluview/fluview/2010/1', f"Wrong original origin: {origin}"
    
    # Check filled data origins
    filled_data = frame_2010[frame_2010['origin'].str.contains('filled', na=False)]
    if not filled_data.empty:
        for origin in filled_data['origin'].unique():
            assert origin.startswith('fluview/fluview/2010/1[filled_'), f"Wrong filled origin format: {origin}"


def test_config_validation(test_df, season_setup):
    """Test that invalid configs are rejected."""
    # Mixed proportion and multiplier config should fail
    mixed_config = {
        "fluview": {"proportion": 0.7, "total": 100},
        "flusurv": {"multiplier": 2}
    }
    
    with pytest.raises(ValueError, match="Cannot mix 'proportion' and 'multiplier' configs"):
        dataset_mixer.build_frames(
            test_df, mixed_config, season_setup, fill_missing_locations="zeros"
        )


def test_invalid_h1_dataset(test_df, season_setup):
    """Test that non-existent H1 datasets are rejected."""
    config = {"nonexistent_dataset": {"multiplier": 1}}
    
    with pytest.raises(ValueError, match="Config references non-existent H1 datasets"):
        dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="zeros"
        )


def test_invalid_fill_strategy(test_df, season_setup):
    """Test that invalid fill strategies are rejected."""
    config = {"fluview": {"multiplier": 1}}
    
    with pytest.raises(ValueError, match="fill_missing_locations must be one of"):
        dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="invalid_strategy"
        )
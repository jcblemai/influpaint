#!/usr/bin/env python3
"""
Comprehensive tests for dataset_mixer.py

Tests the intelligent location filling, performance, and data integrity.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import os

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import dataset_mixer
from season_axis import SeasonAxis

def run_test_suite():
    """Run comprehensive test suite for dataset mixer."""
    print("=" * 60)
    print("DATASET MIXER TEST SUITE")
    print("=" * 60)
    
    # Load test dataset
    test_path = current_dir / 'test_dataset.parquet'
    if not test_path.exists():
        print(f"❌ Test dataset not found at {test_path}. Run create_test_dataset.py first.")
        return False
    
    test_df = pd.read_parquet(test_path)
    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    
    print(f"✅ Loaded test dataset: {test_df.shape}")
    print(f"   H1 datasets: {list(test_df['datasetH1'].unique())}")
    print(f"   Years: {sorted(test_df['fluseason'].unique())}")
    
    # Run all tests
    all_passed = True
    tests = [
        test_basic_functionality,
        test_intelligent_filling,
        test_missing_location_strategies,
        test_multiplier_configs,
        test_proportion_configs,
        test_performance,
        test_column_cleanup,
        test_origin_tracking
    ]
    
    for test_func in tests:
        print(f"\n{'-' * 40}")
        try:
            passed = test_func(test_df, season_setup)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED with exception: {e}")
            all_passed = False
    
    print(f"\n{'=' * 60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'=' * 60}")
    
    return all_passed


def test_basic_functionality(test_df, season_setup):
    """Test basic frame building functionality."""
    print("TEST: Basic functionality")
    
    config = {"fluview": {"multiplier": 1}}
    
    try:
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="zeros"
        )
        
        if not frames:
            print("❌ No frames returned")
            return False
            
        if len(frames) != 2:  # 2010 and 2021
            print(f"❌ Expected 2 frames, got {len(frames)}")
            return False
            
        first_frame = frames[0]
        expected_cols = ['location_code', 'season_week', 'value', 'week_enddate', 'origin']
        
        if list(first_frame.columns) != expected_cols:
            print(f"❌ Wrong columns. Expected {expected_cols}, got {list(first_frame.columns)}")
            return False
            
        print("✅ Basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False


def test_intelligent_filling(test_df, season_setup):
    """Test that intelligent filling works correctly."""
    print("TEST: Intelligent filling")
    
    config = {"fluview": {"multiplier": 1}}
    
    try:
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="random"
        )
        
        # Find the 2010 frame (should have missing locations 12, 22)
        frame_2010 = None
        for frame in frames:
            # Check if this is a 2010 frame by looking for characteristic data
            if '2010' in frame['origin'].iloc[0]:
                frame_2010 = frame
                break
        
        if frame_2010 is None:
            print("❌ Could not find 2010 frame")
            return False
        
        # Check that locations 12 and 22 are present (filled)
        frame_locations = set(frame_2010['location_code'].unique())
        if '12' not in frame_locations or '22' not in frame_locations:
            print(f"❌ Missing locations not filled. Frame has: {frame_locations}")
            return False
        
        # Check that filled locations have intelligent origins
        loc_12_data = frame_2010[frame_2010['location_code'] == '12']
        loc_22_data = frame_2010[frame_2010['location_code'] == '22']
        
        origin_12 = loc_12_data['origin'].iloc[0]
        origin_22 = loc_22_data['origin'].iloc[0]
        
        # Should have intelligent filling origins, not random
        if 'same_location_year_2021' not in origin_12:
            print(f"❌ Location 12 not filled intelligently. Origin: {origin_12}")
            return False
            
        if 'same_location_year_2021' not in origin_22:
            print(f"❌ Location 22 not filled intelligently. Origin: {origin_22}")
            return False
        
        print(f"✅ Intelligent filling works: loc 12 from {origin_12[:50]}...")
        print(f"✅ Intelligent filling works: loc 22 from {origin_22[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent filling failed: {e}")
        return False


def test_missing_location_strategies(test_df, season_setup):
    """Test different missing location strategies."""
    print("TEST: Missing location strategies")
    
    config = {"fluview": {"multiplier": 1}}
    
    strategies = ["error", "zeros", "random", "skip"]
    
    try:
        # Test error strategy - should fail
        try:
            frames = dataset_mixer.build_frames(
                test_df, config, season_setup, fill_missing_locations="error"
            )
            print("❌ Error strategy should have failed but didn't")
            return False
        except ValueError as e:
            if "Missing required locations" in str(e):
                print("✅ Error strategy correctly fails with missing locations")
            else:
                print(f"❌ Error strategy failed with wrong error: {e}")
                return False
        
        # Test zeros strategy
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="zeros"
        )
        
        frame_2010 = next(f for f in frames if '2010' in f['origin'].iloc[0])
        loc_12_zeros = frame_2010[frame_2010['location_code'] == '12']
        
        if not (loc_12_zeros['value'] == 0.0).all():
            print("❌ Zeros strategy didn't fill with zeros")
            return False
        
        print("✅ Zeros strategy works")
        
        # Test skip strategy  
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="skip"
        )
        
        # Should have fewer frames (2010 frame should be skipped)
        if len(frames) != 1:  # Only 2021 frame should remain
            print(f"❌ Skip strategy should return 1 frame, got {len(frames)}")
            return False
            
        print("✅ Skip strategy works")
        return True
        
    except Exception as e:
        print(f"❌ Missing location strategies test failed: {e}")
        return False


def test_multiplier_configs(test_df, season_setup):
    """Test multiplier configurations."""
    print("TEST: Multiplier configurations")
    
    try:
        config = {
            "fluview": {"multiplier": 2},
            "flusurv": {"multiplier": 1}
        }
        
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="zeros"
        )
        
        # Should have: 2 fluview frames (2010, 2021) * 2 + 1 flusurv frame = 5 frames
        if len(frames) != 5:
            print(f"❌ Expected 5 frames with multipliers, got {len(frames)}")
            return False
        
        # Count origins
        fluview_count = sum(1 for f in frames if 'fluview' in f['origin'].iloc[0])
        flusurv_count = sum(1 for f in frames if 'flusurv' in f['origin'].iloc[0])
        
        if fluview_count != 4:  # 2 original * 2 multiplier
            print(f"❌ Expected 4 fluview frames, got {fluview_count}")
            return False
            
        if flusurv_count != 1:
            print(f"❌ Expected 1 flusurv frame, got {flusurv_count}")
            return False
        
        print("✅ Multiplier configurations work")
        return True
        
    except Exception as e:
        print(f"❌ Multiplier configurations test failed: {e}")
        return False


def test_proportion_configs(test_df, season_setup):
    """Test proportion-based configurations."""
    print("TEST: Proportion configurations")
    
    try:
        config = {
            "fluview": {"proportion": 0.8, "total": 10},
            "flusurv": {"proportion": 0.2, "total": 10}
        }
        
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="zeros"
        )
        
        if len(frames) == 0:
            print("❌ No frames returned for proportion config")
            return False
        
        print(f"✅ Proportion configurations work - got {len(frames)} frames")
        return True
        
    except Exception as e:
        print(f"❌ Proportion configurations test failed: {e}")
        return False


def test_performance(test_df, season_setup):
    """Test that performance is acceptable."""
    print("TEST: Performance")
    
    config = {
        "fluview": {"multiplier": 2},
        "flusurv": {"multiplier": 2}
    }
    
    try:
        start_time = time.time()
        
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="random"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if duration > 5.0:  # Should complete in under 5 seconds
            print(f"❌ Performance too slow: {duration:.2f} seconds")
            return False
        
        print(f"✅ Performance good: {duration:.2f} seconds for {len(frames)} frames")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_column_cleanup(test_df, season_setup):
    """Test that unnecessary columns are removed."""
    print("TEST: Column cleanup")
    
    config = {"fluview": {"multiplier": 1}}
    
    try:
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="zeros"
        )
        
        expected_cols = ['location_code', 'season_week', 'value', 'week_enddate', 'origin']
        
        for i, frame in enumerate(frames):
            if list(frame.columns) != expected_cols:
                print(f"❌ Frame {i} has wrong columns: {list(frame.columns)}")
                return False
            
            # Check for NaN values
            for col in frame.columns:
                if frame[col].isna().any():
                    print(f"❌ Frame {i} has NaN values in column {col}")
                    return False
        
        print("✅ Column cleanup works - only essential columns, no NaN values")
        return True
        
    except Exception as e:
        print(f"❌ Column cleanup test failed: {e}")
        return False


def test_origin_tracking(test_df, season_setup):
    """Test that origin tracking is accurate."""
    print("TEST: Origin tracking")
    
    config = {"fluview": {"multiplier": 1}}
    
    try:
        frames = dataset_mixer.build_frames(
            test_df, config, season_setup, fill_missing_locations="random"
        )
        
        frame_2010 = next(f for f in frames if '2010' in f['origin'].iloc[0])
        
        # Check original data origins
        original_data = frame_2010[~frame_2010['origin'].str.contains('filled', na=False)]
        if not all('fluview/fluview/2010/1' == origin for origin in original_data['origin'].unique()):
            print(f"❌ Wrong original origins: {original_data['origin'].unique()}")
            return False
        
        # Check filled data origins
        filled_data = frame_2010[frame_2010['origin'].str.contains('filled', na=False)]
        if not filled_data.empty:
            for origin in filled_data['origin'].unique():
                if not origin.startswith('fluview/fluview/2010/1[filled_'):
                    print(f"❌ Wrong filled origin format: {origin}")
                    return False
        
        print("✅ Origin tracking is accurate")
        return True
        
    except Exception as e:
        print(f"❌ Origin tracking test failed: {e}")
        return False


if __name__ == "__main__":
    run_test_suite()
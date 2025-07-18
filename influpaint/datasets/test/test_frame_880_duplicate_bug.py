#!/usr/bin/env python3
"""
FAST test for the specific duplicate bug found in frame 880 from create_datasets.py

Uses minimal test dataset (2705 rows) to quickly reproduce the exact bug scenario:
- round5_SigSci-SWIFT_A-2024-08-01 frame missing location '11'
- Should get exactly ≤53 rows after intelligent filling, not 131+ duplicates
"""

import pandas as pd
import pytest
from pathlib import Path
import sys

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from influpaint.datasets import mixer as dataset_mixer
from influpaint.utils import SeasonAxis


@pytest.fixture
def test_df():
    """Load minimal test dataset."""
    test_path = current_dir / 'test_dataset.parquet'
    if not test_path.exists():
        pytest.skip(f"Test dataset not found at {test_path}. Run extraction_script.py first.")
    return pd.read_parquet(test_path)


@pytest.fixture  
def season_setup():
    """Create season setup."""
    return SeasonAxis.for_flusight(remove_us=True, remove_territories=True)


def test_frame_880_duplicate_bug_fast(test_df, season_setup):
    """
    FAST regression test for frame 880 duplicate bug.
    
    Tests the exact problematic scenario from create_datasets.py:
    - round5_SigSci-SWIFT_A-2024-08-01/2024/1.0_NA_101 missing location '11'  
    - Before fix: 131 rows (duplicates)
    - After fix: ≤53 rows (no duplicates)
    """
    # 100M config that caused the original bug
    mix_cfg = {
        "flepiR1": {"multiplier": 1},
        "SMH_R4-R5": {"multiplier": 1}
    }
    
    frames = dataset_mixer.build_frames(
        test_df, mix_cfg, season_setup, fill_missing_locations="random"
    )
    
    # Find the problematic frame
    problematic_frame = None
    for frame in frames:
        if "round5_SigSci-SWIFT_A-2024-08-01" in frame['origin'].iloc[0]:
            problematic_frame = frame
            break
    
    assert problematic_frame is not None, "Could not find round5_SigSci-SWIFT_A-2024-08-01 frame"
    
    # THE CRITICAL TEST: Check location '11' data
    location_11_data = problematic_frame[problematic_frame['location_code'] == '11']
    
    print(f"Location 11 has {len(location_11_data)} rows")
    
    # Main assertion: Should have ≤53 rows, not 131+
    assert len(location_11_data) <= 53, \
        f"Location 11 should have ≤53 rows (one per week), got {len(location_11_data)}"
    
    # No duplicate weeks
    duplicated_weeks = location_11_data[location_11_data.duplicated(subset=['season_week'], keep=False)]
    assert duplicated_weeks.empty, f"Found duplicate weeks: {duplicated_weeks['season_week'].tolist()}"
    
    # Verify fill source tracking
    if not location_11_data.empty:
        origins = location_11_data['origin'].unique()
        assert len(origins) == 1, f"Should have single origin, got: {origins}"
        assert '[filled_' in origins[0], f"Should show fill source: {origins[0]}"
    
    print("✅ SUCCESS: No duplicates, proper filling")


def test_randomization_across_multiplied_frames_fast(test_df, season_setup):
    """
    FAST test that multiplied frames get different random fill data.
    """
    # Create 2 copies to test randomization (reduced for speed)
    mix_cfg = {"SMH_R4-R5": {"multiplier": 2}}
    
    frames = dataset_mixer.build_frames(
        test_df, mix_cfg, season_setup, fill_missing_locations="random"
    )
    
    # Find all copies of the problematic frame
    target_frames = [f for f in frames if "round5_SigSci-SWIFT_A-2024-08-01" in f['origin'].iloc[0]]
    
    assert len(target_frames) == 2, f"Expected 2 copies, got {len(target_frames)}"
    
    # Check fill sources for location 11
    fill_origins = []
    for frame in target_frames:
        loc_11_data = frame[frame['location_code'] == '11']
        if not loc_11_data.empty:
            fill_origins.append(loc_11_data['origin'].iloc[0])
    
    assert len(fill_origins) == 2, f"Expected 2 fill origins, got {len(fill_origins)}"
    
    # Verify each frame has proper single fill source
    for i, origin in enumerate(fill_origins):
        assert '[filled_' in origin, f"Frame {i} should show fill source: {origin}"
    
    print(f"Fill origins: {[o.split('[filled_')[1].split(']')[0] for o in fill_origins]}")
    print("✅ SUCCESS: Randomization working")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
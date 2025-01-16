import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from dataset_mixer import merge_datasets, pad_single_frame, build_frames

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing"""
    base_data = {
        'season_week': [1, 2, 3],
        'location_code': ['loc1', 'loc1', 'loc1'],
        'value': [10.0, 20.0, 30.0],
        'fluseason': [2020, 2020, 2020],
        'week_enddate': [
            datetime(2020, 1, 4),
            datetime(2020, 1, 11),
            datetime(2020, 1, 18)
        ]
    }
    return pd.DataFrame(base_data)

def test_merge_datasets_basic(sample_data):
    """Test basic dataset merging functionality"""
    datasets = {
        'test1': {'df': sample_data, 'multiplier': 2}
    }
    
    result = merge_datasets(datasets)
    
    assert len(result) == 6  # 3 rows * 2 multipliers
    assert set(result['dataset_name']) == {'test1_0', 'test1_1'}
    assert set(result['fluseason']) == {2020, 10200}

def test_merge_datasets_multiple(sample_data):
    """Test merging multiple datasets"""
    datasets = {
        'test1': {'df': sample_data, 'multiplier': 1},
        'test2': {'df': sample_data, 'multiplier': 2}
    }
    
    result = merge_datasets(datasets)
    
    assert len(result) == 9  # (3 * 1) + (3 * 2)
    assert set(result['dataset_name']) == {'test1_0', 'test2_0', 'test2_1'}

def test_pad_single_frame():
    """Test padding of a single location's data"""
    test_data = pd.DataFrame({
        'season_week': [1, 3],
        'location_code': ['loc1', 'loc1'],
        'value': [10.0, 30.0]
    })
    
    result = pad_single_frame(test_data, 'loc1')
    
    assert len(result) == 53  # Should have all weeks
    assert result[result['season_week'] == 2]['value'].values[0] == 10.0  # Filled with previous
    assert result[result['season_week'] == 4]['value'].values[0] == 0.0  # Filled with zero

def test_build_frames_basic(sample_data):
    """Test basic frame building functionality"""
    datasets = {
        'test1': {'df': sample_data, 'multiplier': 1}
    }
    
    frames = build_frames(datasets)
    
    assert len(frames) == 1  # One season, one multiplier
    frame = frames[0]
    assert len(frame) == 53  # Should have all weeks for one location
    assert set(frame['location_code']) == {'loc1'}

def test_build_frames_missing_data(sample_data):
    """Test handling of missing location-season combinations"""
    # Create data with missing location
    sample_data2 = sample_data.copy()
    sample_data2['location_code'] = 'loc2'
    
    datasets = {
        'test1': {'df': sample_data, 'multiplier': 1},
        'test2': {'df': sample_data2, 'multiplier': 1}
    }
    
    frames = build_frames(datasets)
    
    # Should have frames for both locations
    assert len(frames) == 2
    assert set([f['location_code'].iloc[0] for f in frames]) == {'loc1', 'loc2'}

def test_merge_datasets_validation(sample_data):
    """Test validation of required columns"""
    invalid_data = sample_data.drop(columns=['season_week'])
    
    with pytest.raises(ValueError) as excinfo:
        merge_datasets({'test1': {'df': invalid_data, 'multiplier': 1}})
    
    assert "Missing required columns" in str(excinfo.value)

def test_pad_single_frame_edge_cases():
    """Test edge cases in padding"""
    # Test empty frame
    empty_frame = pd.DataFrame(columns=['season_week', 'location_code', 'value'])
    result = pad_single_frame(empty_frame, 'loc1')
    assert len(result) == 53
    assert (result['value'] == 0).all()

    # Test single week
    single_week = pd.DataFrame({
        'season_week': [1],
        'location_code': ['loc1'],
        'value': [10.0]
    })
    result = pad_single_frame(single_week, 'loc1')
    assert len(result) == 53
    assert (result[result['season_week'] > 1]['value'] == 0).all()

def test_build_frames_multiple_multipliers(sample_data):
    """Test handling of multiple multipliers"""
    datasets = {
        'test1': {'df': sample_data, 'multiplier': 3}
    }
    
    frames = build_frames(datasets)
    
    assert len(frames) == 3  # Should have 3 frames (one per multiplier)
    assert all(len(f) == 53 for f in frames)  # Each should be complete

"""
Basic tests for scoring module functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scoring module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import scoring.evaluation as scoring_eval
from scoring.evaluation import MetricRegistry


class TestScoringBasics:
    """Test basic scoring functionality."""
    
    def test_data_loads(self, forecast_dataset, ground_truth_df):
        """Test that test data loads correctly."""
        assert isinstance(forecast_dataset, scoring_eval.ForecastDataset)
        assert len(forecast_dataset.records) > 0
        assert isinstance(ground_truth_df, pd.DataFrame)
        assert not ground_truth_df.empty
    
    def test_scoring_works(self, scoring_results):
        """Test that scoring produces results."""
        assert hasattr(scoring_results, 'forecast_metrics')
        assert hasattr(scoring_results, 'model_metrics')
        assert not scoring_results.forecast_metrics.empty
        
        # Check we have expected metrics
        metrics = set(scoring_results.forecast_metrics['scoring_metric'].unique())
        expected = {'wis_total', 'wis_sharpness', 'wis_overprediction', 'wis_underprediction'}
        assert expected.issubset(metrics)
    
    def test_wis_values_reasonable(self, scoring_results):
        """Test WIS values are reasonable."""
        fm = scoring_results.forecast_metrics
        wis_total = fm[fm['scoring_metric'] == 'wis_total']['value']
        
        assert len(wis_total) > 0, "Should have WIS values"
        # Exclude NaN values when checking non-negativity (NaN >= 0 is False)
        non_nan_wis = wis_total.dropna()
        assert len(non_nan_wis) > 0, "Should have non-NaN WIS values"
        assert all(non_nan_wis >= 0), "WIS should be non-negative"
    
    def test_model_metrics_computed(self, scoring_results):
        """Test per-model metrics are computed."""
        mm = scoring_results.model_metrics
        
        if not mm.empty:
            assert 'coverage_95' in mm['scoring_metric'].values
            coverage = mm[mm['scoring_metric'] == 'coverage_95']['value']
            if len(coverage) > 0:
                assert all(coverage >= 0) and all(coverage <= 1), "Coverage should be 0-1"
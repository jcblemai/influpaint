"""
Test comparison with R scoringutils package results.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestRComparison:
    """Test comparison with pre-computed R results."""
    
    @pytest.fixture(scope="class")
    def r_results_file(self):
        """Path to pre-computed R results."""
        return Path(__file__).parent.parent / "scoring/tests/small_flusight/r_scoringutils_results.csv"
    
    def test_export_python_results(self, scoring_results, results_dir):
        """Export Python results for comparison."""
        # Export forecast metrics
        python_file = results_dir / "python_wis_results.csv"
        wis_results = scoring_results.forecast_metrics[
            scoring_results.forecast_metrics['scoring_metric'] == 'wis_total'
        ].copy()
        wis_results.to_csv(python_file, index=False)
        
        assert python_file.exists()
        assert wis_results.shape[0] > 0
    
    def test_compare_with_r_results(self, scoring_results, r_results_file):
        """Compare Python WIS with pre-computed R results per forecast with exact alignment."""
        if not r_results_file.exists():
            pytest.skip("R results file not found - run R script first")
        
        # Load R results
        r_results = pd.read_csv(r_results_file)
        r_wis_data = r_results.copy()
        
        # Get Python WIS
        py_results = scoring_results.forecast_metrics
        py_wis_data = py_results[py_results['scoring_metric'] == 'wis_total'].copy()
        
        # Normalize data types and formats for exact alignment
        align_cols = ['model', 'target_end_date', 'location', 'horizon']
        
        # Prepare R data - ensure exact format matching, handle NaN values
        r_wis_data['target_end_date'] = pd.to_datetime(r_wis_data['target_end_date']).dt.date
        r_wis_data['location'] = r_wis_data['location'].astype(str).str.strip()
        r_wis_data['model'] = r_wis_data['model'].astype(str).str.strip()
        # Handle NaN in horizon - drop rows with missing horizon
        r_wis_data = r_wis_data.dropna(subset=['horizon'])
        r_wis_data['horizon'] = r_wis_data['horizon'].astype(int)
        r_align = r_wis_data[align_cols + ['wis']].copy()
        r_align.columns = align_cols + ['r_wis']
        
        # Prepare Python data - ensure exact format matching  
        py_wis_data['target_end_date'] = pd.to_datetime(py_wis_data['target_end_date']).dt.date
        py_wis_data['location'] = py_wis_data['location'].astype(str).str.strip()
        py_wis_data['model'] = py_wis_data['model'].astype(str).str.strip()
        # Handle NaN in horizon - drop rows with missing horizon
        py_wis_data = py_wis_data.dropna(subset=['horizon'])
        py_wis_data['horizon'] = py_wis_data['horizon'].astype(int)
        py_align = py_wis_data[align_cols + ['value']].copy()
        py_align.columns = align_cols + ['py_wis']
        
        # Debug: Print unique values in each dimension before merging
        print(f"\\nR data dimensions:")
        print(f"  Models: {sorted(r_align['model'].unique())}")
        print(f"  Dates: {sorted(r_align['target_end_date'].unique())}")
        print(f"  Locations: {sorted(r_align['location'].unique())[:10]}...")
        print(f"  Horizons: {sorted(r_align['horizon'].unique())}")
        print(f"  Total R records: {len(r_align)}")
        
        print(f"\\nPython data dimensions:")
        print(f"  Models: {sorted(py_align['model'].unique())}")
        print(f"  Dates: {sorted(py_align['target_end_date'].unique())}")
        print(f"  Locations: {sorted(py_align['location'].unique())[:10]}...")
        print(f"  Horizons: {sorted(py_align['horizon'].unique())}")
        print(f"  Total Python records: {len(py_align)}")
        
        # Inner join to find exactly matching forecasts
        merged = pd.merge(py_align, r_align, on=align_cols, how='inner', suffixes=('_py', '_r'))
        
        print(f"\\nMatching forecasts: {len(merged)}")
        
        if len(merged) == 0:
            # Show detailed alignment failures
            print("\\nAlignment analysis:")
            for col in align_cols:
                py_vals = set(py_align[col].unique())
                r_vals = set(r_align[col].unique()) 
                common = py_vals & r_vals
                py_only = py_vals - r_vals
                r_only = r_vals - py_vals
                print(f"  {col}: {len(common)} common, {len(py_only)} Python-only, {len(r_only)} R-only")
                if len(py_only) > 0:
                    print(f"    Python-only {col}: {sorted(list(py_only))[:5]}...")
                if len(r_only) > 0:
                    print(f"    R-only {col}: {sorted(list(r_only))[:5]}...")
            
            pytest.fail("No matching forecasts found between Python and R results")
        
        # Sort for reproducible testing
        merged = merged.sort_values(align_cols).reset_index(drop=True)
        
        # Compare each forecast individually with strict tolerance
        mismatches = []
        for idx, row in merged.iterrows():
            py_val = row['py_wis']
            r_val = row['r_wis']
            
            if pd.notna(py_val) and pd.notna(r_val) and py_val > 0:
                relative_diff = abs(py_val - r_val) / py_val
                if relative_diff >= 0.01:  # 1% tolerance
                    mismatch = {
                        'model': row['model'],
                        'target_end_date': row['target_end_date'],
                        'location': row['location'], 
                        'horizon': row['horizon'],
                        'py_wis': py_val,
                        'r_wis': r_val,
                        'diff_pct': relative_diff * 100
                    }
                    mismatches.append(mismatch)
        
        # Report results
        py_mean = merged['py_wis'].mean()
        r_mean = merged['r_wis'].mean()
        overall_diff = abs(py_mean - r_mean) / py_mean if py_mean > 0 else float('inf')
        
        print(f"\\nComparison summary:")
        print(f"  Forecasts compared: {len(merged)}")
        print(f"  Mismatches (>1%): {len(mismatches)}")
        print(f"  Python mean WIS: {py_mean:.6f}")
        print(f"  R mean WIS: {r_mean:.6f}")
        print(f"  Overall difference: {overall_diff:.1%}")
        
        if mismatches:
            print(f"\\nFirst few mismatches:")
            for i, mm in enumerate(mismatches[:3]):
                print(f"  {mm['model']} | {mm['target_end_date']} | {mm['location']} | h={mm['horizon']}: "
                      f"Py={mm['py_wis']:.3f}, R={mm['r_wis']:.3f}, diff={mm['diff_pct']:.1f}%")
            
            # Fail on first mismatch with detailed info
            first_mm = mismatches[0]
            pytest.fail(f"WIS differs for {first_mm['model']} on {first_mm['target_end_date']} "
                       f"loc={first_mm['location']} h={first_mm['horizon']}: "
                       f"Python={first_mm['py_wis']:.6f}, R={first_mm['r_wis']:.6f}, "
                       f"diff={first_mm['diff_pct']:.1f}%")
        
        print("\\n✅ All matching forecasts within 1% tolerance")
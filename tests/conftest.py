"""
Pytest fixtures for InfluPaint scoring tests.
"""

import pytest
import pandas as pd
from pathlib import Path
from datetime import date
import sys

# Add scoring module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import scoring.evaluation as scoring_eval
from scoring.evaluation import MetricRegistry


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "scoring/tests/small_flusight"


@pytest.fixture(scope="session") 
def results_dir():
    """Directory for test results."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture(scope="session")
def ground_truth_df(test_data_dir):
    """Load ground truth data."""
    gt_file = test_data_dir / "target-data/target-hospital-admissions.csv"
    gt_df = pd.read_csv(gt_file)
    gt_df['date'] = pd.to_datetime(gt_df['date']).dt.date
    # Normalize location to consistent format (remove quotes, pad single digits)
    gt_df['location'] = gt_df['location'].astype(str).str.strip()
    gt_df['location'] = gt_df['location'].str.replace(r'^"([^"]*)"$', r'\1', regex=True)  # Remove quotes
    gt_df['location'] = gt_df['location'].apply(lambda x: f'0{x}' if len(x) == 1 and x.isdigit() else x)  # Pad single digits
    return gt_df


@pytest.fixture(scope="session")
def forecast_dates():
    """Expected forecast dates for test data."""
    return [date(2024, 12, 28), date(2025, 3, 22)]


@pytest.fixture(scope="session")
def forecast_dataset(test_data_dir, forecast_dates):
    """Load test forecast dataset."""
    model_output_dir = test_data_dir / "model_output"
    records = []
    
    for model_dir in model_output_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        for forecast_date in forecast_dates:
            forecast_file = model_dir / f"{forecast_date}-{model_name}.csv"
            
            if forecast_file.exists():
                df = pd.read_csv(forecast_file)
                
                # Prepare data
                df['target_end_date'] = pd.to_datetime(df['target_end_date']).dt.date
                df['output_type_id'] = pd.to_numeric(df['output_type_id'], errors='coerce')
                df['horizon'] = pd.to_numeric(df['horizon'], errors='coerce')
                # Normalize location format consistently
                df['location'] = df['location'].astype(str).str.strip()
                df['location'] = df['location'].str.replace(r'^"([^"]*)"$', r'\1', regex=True)  # Remove quotes  
                df['location'] = df['location'].apply(lambda x: f'0{x}' if len(x) == 1 and x.isdigit() else x)  # Pad single digits
                
                # Filter to quantile forecasts
                df = df[
                    (df['output_type'] == 'quantile') & 
                    (df['target'] == 'wk inc flu hosp')
                ].copy()
                
                if len(df) > 0:
                    group = 'influpaint' if model_name.startswith('UNC_IDD') else 'flusight'
                    
                    record = scoring_eval.ForecastRecord(
                        model=model_name,
                        group=group,
                        display_name=model_name,
                        forecast_date=pd.Timestamp(forecast_date),
                        df=df
                    )
                    records.append(record)
    
    return scoring_eval.ForecastDataset(records)


@pytest.fixture(scope="session")
def scoring_results(forecast_dataset, ground_truth_df, forecast_dates):
    """Compute scoring results for test data."""
    per_model_metrics = [
        MetricRegistry.COVERAGE_95,
        MetricRegistry.COVERAGE_95_GAP,
        MetricRegistry.COMPLETION_RATE
    ]
    
    # For testing, we'll filter to overlapping locations only
    # Get forecast locations from first model
    first_record = forecast_dataset.records[0] if forecast_dataset.records else None
    if first_record is None:
        # Return empty results
        from scoring.evaluation import ScoringResults
        return ScoringResults(
            forecast_metrics=pd.DataFrame(),
            model_metrics=pd.DataFrame(),
            meta={}
        )
    
    forecast_locations = set(first_record.df['location'].astype(str).unique())
    gt_locations = set(ground_truth_df['location'].astype(str).unique())
    common_locations = forecast_locations & gt_locations
    
    if not common_locations:
        # No overlapping locations, create minimal test data
        from scoring.evaluation import ScoringResults
        return ScoringResults(
            forecast_metrics=pd.DataFrame(columns=['model', 'forecast_date', 'target', 'target_end_date', 'scoring_metric', 'location', 'value']),
            model_metrics=pd.DataFrame(columns=['model', 'horizon', 'scoring_metric', 'value']),
            meta={'note': 'test data has no overlapping locations'}
        )
    
    # Filter datasets to common locations
    filtered_records = []
    for record in forecast_dataset.records:
        filtered_df = record.df[record.df['location'].isin(common_locations)].copy()
        if not filtered_df.empty:
            new_record = scoring_eval.ForecastRecord(
                model=record.model,
                group=record.group,
                display_name=record.display_name,
                forecast_date=record.forecast_date,
                df=filtered_df
            )
            filtered_records.append(new_record)
    
    filtered_dataset = scoring_eval.ForecastDataset(filtered_records)
    filtered_gt = ground_truth_df[ground_truth_df['location'].isin(common_locations)].copy()
    
    return scoring_eval.score_dataset(
        dataset=filtered_dataset,
        ground_truth=filtered_gt,
        expected_dates=forecast_dates,
        metrics=per_model_metrics
    )
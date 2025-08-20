# %% [markdown]
# InfluPaint vs FluSight Evaluation Orchestrator
# This notebook-style script orchestrates the evaluation process by:
# - Loading CSV forecast data for InfluPaint and FluSight models
# - Creating ForecastDataset objects with proper grouping and display names
# - Using evaluation_module for scoring and plotting
# - Supporting flexible model grouping (influpaint vs flusight, scenarios, etc.)

# %%
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Union

# Import our evaluation module
from evaluation_module import (
    ForecastRecord, 
    ForecastDataset, 
    score_dataset, 
    compute_relative_scores,
    create_heatmap_plot,
    create_component_plot,
    create_time_series_plot
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sns.set_theme()

# %%
# Configuration Constants
class Config:
    """Configuration settings for evaluation."""
    
    # File paths
    JOBS_FILE = "inpaint_jobs_paper-2025-07-22.txt"
    INPAINT_RES_BASE = "from_longleaf/influpaint_res"
    
    FLUSIGHT_BASES = {
        "2023-2024": "Flusight/2023-2024/FluSight-forecast-hub-official",
        "2024-2025": "Flusight/2024-2025/FluSight-forecast-hub-official",
    }
    
    # Model filtering
    IGNORED_FLUSIGHT_MODELS = {
        "LosAlamos_NAU-CModel_Flu",
        "SigSci-CREG", 
        "SigSci-TSENS",
        "LUcompUncertLab-experthuman",
        "VTSanghani-ExogModel",
        "CADPH-FluCAT_Ensemble",
    }
    
    # Evaluation parameters
    TARGET_NAME = "wk inc flu hosp"
    HORIZONS = [0, 1, 2, 3]  # horizons to evaluate
    ALLOW_MISSING_DATES_PER_MODEL = 5
    
    # Quantile requirements for forecasts
    REQUIRED_QUANTILES = [0.01, 0.025] + list(np.arange(0.05, 0.95 + 0.05, 0.05)) + [0.975, 0.99]
    
    # Model grouping and colors
    GROUP_COLORS = {
        'influpaint': 'green',
        'flusight': 'blue'
    }
    
    # WIS components for stacking (excluding calibration)
    STACK_COMPONENTS = ["wis_sharpness", "wis_overprediction", "wis_underprediction"]

# %%
# WIS scoring handled in evaluate_deprecated.py


# %%
# Helpers
def read_jobs(jobs_file: Optional[str] = None) -> pd.DataFrame:
    """
    Read job dates from inpaint jobs file.
    
    Args:
        jobs_file: Path to jobs file. If None, uses Config.JOBS_FILE.
        
    Returns:
        DataFrame with columns: job_id, scenario_id, run_id, season, date, config
    """
    if jobs_file is None:
        jobs_file = Config.JOBS_FILE
    df = pd.read_csv(jobs_file)
    # Expect columns: job_id, scenario_id, run_id, season, date, config
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def season_dates_from_jobs(df: pd.DataFrame) -> Dict[str, List[dt.date]]:
    """
    Extract unique dates per season from jobs dataframe.
    
    Args:
        df: Jobs dataframe containing season and date columns
        
    Returns:
        Dictionary mapping season names to lists of dates
    """
    out = {}
    for season, g in df.groupby("season"):
        out[str(season)] = sorted(g["date"].unique().tolist())
    return out


def load_ground_truth(season: str) -> pd.DataFrame:
    """
    Load ground truth hospital admissions data for a season.
    
    Args:
        season: Season identifier (e.g., "2023-2024")
        
    Returns:
        DataFrame with columns: date, location, value
        
    Raises:
        KeyError: If season not found in configuration
        FileNotFoundError: If ground truth file doesn't exist
    """
    if season not in Config.FLUSIGHT_BASES:
        raise KeyError(f"Season '{season}' not found in configuration. Available: {list(Config.FLUSIGHT_BASES.keys())}")
    
    base = Config.FLUSIGHT_BASES[season]
    p = os.path.join(base, "target-data", "target-hospital-admissions.csv")
    
    if not os.path.exists(p):
        raise FileNotFoundError(f"Ground truth file not found: {p}")
    
    try:
        gt = pd.read_csv(p)
        gt["date"] = pd.to_datetime(gt["date"]).dt.date
        # standardize columns used later
        required_cols = ["date", "location", "value"]
        missing_cols = [col for col in required_cols if col not in gt.columns]
        if missing_cols:
            raise ValueError(f"Ground truth file missing required columns: {missing_cols}")
        return gt[required_cols]
    except Exception as e:
        logging.error(f"Error loading ground truth for {season}: {str(e)}")
        raise


def list_flusight_models(season: str) -> List[str]:
    """
    List available FluSight models for a season, excluding ignored models.
    
    Args:
        season: Season identifier (e.g., "2023-2024")
        
    Returns:
        Sorted list of model names
    """
    base = Config.FLUSIGHT_BASES[season]
    model_output = os.path.join(base, "model-output")
    if not os.path.isdir(model_output):
        return []
    models = [d for d in os.listdir(model_output) if os.path.isdir(os.path.join(model_output, d))]
    models = [m for m in models if m not in Config.IGNORED_FLUSIGHT_MODELS]
    return sorted(models)


def load_flusight_forecast(season: str, model: str, ref_date: dt.date) -> pd.DataFrame:
    base = Config.FLUSIGHT_BASES[season]
    p = os.path.join(base, "model-output", model, f"{str(ref_date)}-{model}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # Normalize column names to match hub format we use
    # Some repos use different column orders; enforce expected columns
    expected_cols = {"reference_date", "horizon", "target", "target_end_date", "location", "output_type", "output_type_id", "value"}
    #if not expected_cols.issubset(set(df.columns)):
    #    # Try legacy names from older code
    #    rename_map = {"type": "output_type", "quantile": "output_type_id"}
    #    df = df.rename(columns=rename_map)
    df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
    df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    # Read locations as strings without padding (like in influpaint)
    df["location"] = df["location"].astype(str).str.strip()
    return df


def find_influpaint_csvs(base_dir: str, config: str, ref_date: dt.date) -> List[Tuple[str, str]]:
    """
    Search for all InfluPaint CSVs matching a given config and reference date under base_dir.
    Returns list of tuples (model_name, csv_path). model_name encodes the subfolder (iXXX...) and config.
    """
    res = []
    date_str = str(ref_date)
    # Only search inside batches that match this scenario
    batch_prefix = "07b44fa_paper-2025-07-22_inpainting_"
    for batch in os.listdir(base_dir):
        if not batch.startswith(batch_prefix):
            continue
        batch_dir = os.path.join(base_dir, batch)
        if not os.path.isdir(batch_dir):
            continue
        for root, _, files in os.walk(batch_dir):
            if f"::conf_{config}::{date_str}" in root:
                for f in files:
                    # Look for files with the full model name format from our resave script
                    if f.endswith(".csv") and f.startswith(date_str) and not f.endswith("-copaint.csv"):
                        path = os.path.join(root, f)
                        parts = root.split("::conf_")
                        left = parts[0].rstrip(":")
                        model_id = os.path.basename(left)
                        
                        # Use full model specification for complete identification
                        # Include full model_id with all components plus config
                        scenario_id = model_id.split("::")[0]  # e.g., i806
                        full_name = f"{model_id}::{config}"  # e.g., i806::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_No::celebahq_noTTJ5
                        
                        res.append((full_name, path))
    return res


def filter_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter forecast dataframe to quantile forecasts for target of interest.
    
    Args:
        df: Raw forecast dataframe
        
    Returns:
        Filtered dataframe with specified horizons and target
    """
    keep = (df["output_type"] == "quantile") & (df["target"] == Config.TARGET_NAME)
    df = df.loc[keep].copy()
    # Keep only horizons in HORIZONS list
    df = df[df["horizon"].isin(Config.HORIZONS)]
    # Ensure types
    df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
    df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
    # Read locations as strings without padding (like in influpaint)
    df["location"] = df["location"].astype(str).str.strip()
    return df


# Local score_Nwk_forecasts_hub removed; use evaluate_deprecated.score_Nwk_forecasts_hub


# %%
#! Plotting helpers moved to evaluate_plot.ModelEvaluator


#! Removed legacy evaluate_flusight_models in favor of collect_flusight_records


def create_display_name(model_name: str) -> str:
    """Create a readable display name for plots."""
    if model_name.startswith('i') and '::' in model_name:
        # InfluPaint model: keep full name but add line breaks every 3 components
        parts = model_name.split("::")
        if len(parts) >= 3:
            # Group parts into chunks of 3, separated by newlines
            chunks = []
            for i in range(0, len(parts), 3):
                chunk = "::".join(parts[i:i+3])
                chunks.append(chunk)
            return "\n".join(chunks)
        else:
            return model_name
    else:
        # FluSight model: keep as is
        return model_name


def collect_flusight_records(season: str, dates: List[dt.date]) -> Tuple[List[ForecastRecord], Dict[str, int]]:
    """Collect FluSight forecasts into ForecastRecord list and missing counts."""
    flusight_models = list_flusight_models(season)
    records: List[ForecastRecord] = []
    missing_counts: Dict[str, int] = {}
    for model in flusight_models:
        present = 0
        for d in dates:
            try:
                df = load_flusight_forecast(season, model, d)
                df = filter_forecast_df(df)
                have = sorted(df["output_type_id"].unique().tolist())
                missing_q = sorted(set(np.round(Config.REQUIRED_QUANTILES, 6)) - set(np.round(have, 6)))
                if missing_q:
                    continue
                records.append(
                    ForecastRecord(
                        model=model,
                        group="flusight", 
                        display_name=create_display_name(model),
                        forecast_date=pd.to_datetime(d),
                        df=df,
                    )
                )
                present += 1
            except Exception:
                continue
        missing_counts[model] = len(dates) - present
    keep = {m for m, miss in missing_counts.items() if miss <= Config.ALLOW_MISSING_DATES_PER_MODEL}
    records = [r for r in records if r.model in keep]
    missing_counts = {m: c for m, c in missing_counts.items() if m in keep}
    return records, missing_counts



def collect_influpaint_records(season: str, dates: List[dt.date], season_jobs: pd.DataFrame) -> Tuple[List[ForecastRecord], Dict[str, int], Dict[str, Dict]]:
    """Collect InfluPaint forecasts into ForecastRecord list, missing counts, and failures."""
    configs = sorted(season_jobs["config"].unique().tolist())
    model_paths: Dict[str, Dict[dt.date, str]] = {}
    for config in configs:
        for d in dates:
            matches = find_influpaint_csvs(Config.INPAINT_RES_BASE, config, d)
            for run_name, path in matches:
                model_paths.setdefault(run_name, {})[d] = path

    records: List[ForecastRecord] = []
    missing_counts: Dict[str, int] = {}
    failures: Dict[str, Dict] = {}

    for run_name, by_date in model_paths.items():
        present = 0
        failure_details = {}
        for d in dates:
            if d not in by_date:
                failure_details[d] = "No CSV found"
                continue
            path = by_date[d]
            try:
                df = pd.read_csv(path)
                if "output_type" not in df.columns and "type" in df.columns:
                    df = df.rename(columns={"type": "output_type"})
                if "output_type_id" not in df.columns and "quantile" in df.columns:
                    df = df.rename(columns={"quantile": "output_type_id"})
                df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
                df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
                df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
                df = filter_forecast_df(df)
                have = sorted(df["output_type_id"].unique().tolist())
                missing_q = sorted(set(np.round(Config.REQUIRED_QUANTILES, 6)) - set(np.round(have, 6)))
                if missing_q:
                    failure_details[d] = f"Missing quantiles {missing_q[:3]}{'...' if len(missing_q) > 3 else ''}"
                    continue
                records.append(
                    ForecastRecord(
                        model=run_name,
                        group="influpaint",
                        display_name=create_display_name(run_name),
                        forecast_date=pd.to_datetime(d),
                        df=df,
                    )
                )
                present += 1
            except Exception as e:
                failure_details[d] = f"Error: {str(e)[:80]}"

        missing = len(dates) - present
        missing_counts[run_name] = missing
        failures[run_name] = {
            "present_count": present,
            "missing_count": missing,
            "failure_details": failure_details,
            "kept": (missing <= Config.ALLOW_MISSING_DATES_PER_MODEL) and (present > 0),
        }

    # Models expected but not discovered at all
    expected = season_jobs[["scenario_id", "config"]].drop_duplicates()
    found = set()
    for name in model_paths:
        try:
            scenario_id = int(name.split("::")[0][1:])
            config_name = name.split("::")[-1]
            found.add((scenario_id, config_name))
        except Exception:
            pass
    for _, row in expected.iterrows():
        key = (row["scenario_id"], row["config"]) 
        if key not in found:
            name = f"i{row['scenario_id']}::{row['config']}"
            missing_counts[name] = len(dates)
            failures[name] = {
                "present_count": 0,
                "missing_count": len(dates),
                "failure_details": {d: "No CSV found" for d in dates},
                "kept": False,
            }

    keep = {m for m, miss in missing_counts.items() if miss <= Config.ALLOW_MISSING_DATES_PER_MODEL}
    records = [r for r in records if r.model in keep]
    missing_counts = {m: c for m, c in missing_counts.items() if m in keep}
    failures = {m: v for m, v in failures.items() if m in keep or v.get("present_count", 0) == 0}
    return records, missing_counts, failures


def build_dataset_for_season(season: str, dates: List[dt.date]) -> Tuple[ForecastDataset, Dict[str, int], Dict[str, Dict]]:
    """Load forecasts for both groups and return a ForecastDataset plus missing counts and failures."""
    jobs = read_jobs()
    season_jobs = jobs[jobs["season"] == season]
    flusight_recs, flusight_missing = collect_flusight_records(season, dates)
    influpaint_recs, influpaint_missing, influpaint_failures = collect_influpaint_records(season, dates, season_jobs)
    dataset = ForecastDataset(records=flusight_recs + influpaint_recs)
    missing_counts = {**flusight_missing, **influpaint_missing}
    return dataset, missing_counts, influpaint_failures


#! Legacy evaluate_models removed; dataset-based loader is used instead


def create_failed_jobs_file(season: str, failures: Dict, save_dir: str):
    """Create failed jobs files from failure information."""
    if not failures:
        return
    
    # Read original jobs file to get run_id mapping
    jobs_df = read_jobs()
    season_jobs = jobs_df[jobs_df["season"] == season]
    
    failed_jobs = []
    
    for run_name, details in failures.items():
        print(f"  {run_name}: {details['present_count']}/{details['present_count'] + details['missing_count']} dates successful")
        
        # Extract scenario_id and config from run_name (e.g., "i806::celebahq" -> 806, "celebahq")
        try:
            scenario_id = int(run_name.split("::")[0][1:])  # Remove 'i' prefix
            config_name = run_name.split("::")[-1]
        except (ValueError, IndexError):
            print(f"    Warning: Could not parse scenario_id from {run_name}")
            continue
        
        # Find matching run_id from original jobs
        matching_jobs = season_jobs[
            (season_jobs["scenario_id"] == scenario_id) & 
            (season_jobs["config"] == config_name)
        ]
        
        if matching_jobs.empty:
            print(f"    Warning: No matching job found for scenario {scenario_id}, config {config_name}")
            continue
            
        run_id = matching_jobs.iloc[0]["run_id"]
        
        # Add failed dates to the failed jobs list
        for failed_date, reason in details['failure_details'].items():
            # Find the exact job_id from original jobs file for this scenario/date/config combination
            matching_job = season_jobs[
                (season_jobs["scenario_id"] == scenario_id) & 
                (season_jobs["config"] == config_name) & 
                (season_jobs["date"] == failed_date)
            ]
            
            if matching_job.empty:
                print(f"    Warning: No matching job_id found for scenario {scenario_id}, config {config_name}, date {failed_date}")
                continue
                
            original_job_id = matching_job.iloc[0]["job_id"]
            
            failed_jobs.append({
                'job_id': original_job_id,
                'scenario_id': scenario_id,
                'run_id': run_id,
                'season': season,
                'date': failed_date,
                'config': config_name,
                'failure_reason': reason
            })
        
        # Show failure summary and complete list of failed dates
        if details['failure_details']:
            failure_summary = {}
            for reason in details['failure_details'].values():
                failure_type = reason.split(':')[0] if ':' in reason else reason
                failure_summary[failure_type] = failure_summary.get(failure_type, 0) + 1
            print(f"    Failure reasons: {dict(failure_summary)}")
            print(f"    Status: {'INCLUDED in analysis' if details['kept'] else 'EXCLUDED from analysis'}")
            # Show all failed dates formatted nicely
            failed_dates = sorted(details['failure_details'].keys())
            formatted_dates = [d.strftime('%Y-%m-%d') for d in failed_dates]
            print(f"    Failed dates: {formatted_dates}")
        print()
    
    # Write failed jobs files
    print(f"Total failed jobs collected: {len(failed_jobs)}")
    if failed_jobs:
        failed_jobs_df = pd.DataFrame(failed_jobs)
        
        # Standard format (for rerunning jobs)
        failed_jobs_file = os.path.join(save_dir, f"failed_inpaint_jobs_{season}.txt")
        failed_jobs_df[['job_id', 'scenario_id', 'run_id', 'season', 'date', 'config']].to_csv(
            failed_jobs_file, index=False
        )
        print(f"Written {len(failed_jobs)} failed jobs to: {failed_jobs_file}")
        
        # Detailed format (with failure reasons)
        detailed_file = os.path.join(save_dir, f"failed_inpaint_jobs_{season}_detailed.txt")
        failed_jobs_df.to_csv(detailed_file, index=False)
        print(f"Written detailed failure report to: {detailed_file}")
        
        # Create additional file for completely failed models (zero successful dates)
        completely_failed_jobs = []
        completely_failed_models = []
        
        for run_name, details in failures.items():
            if details['present_count'] == 0:  # All dates failed
                completely_failed_models.append(run_name)
                
                # Extract scenario_id and config from run_name
                try:
                    scenario_id = int(run_name.split("::")[0][1:])  # Remove 'i' prefix
                    config_name = run_name.split("::")[-1]
                except (ValueError, IndexError):
                    continue
                
                # Find matching run_id from original jobs
                matching_jobs = season_jobs[
                    (season_jobs["scenario_id"] == scenario_id) & 
                    (season_jobs["config"] == config_name)
                ]
                
                if matching_jobs.empty:
                    continue
                    
                run_id = matching_jobs.iloc[0]["run_id"]
                
                # Add all failed dates for this completely failed model
                for failed_date, reason in details['failure_details'].items():
                    # Find the exact job_id from original jobs file
                    matching_job = season_jobs[
                        (season_jobs["scenario_id"] == scenario_id) & 
                        (season_jobs["config"] == config_name) & 
                        (season_jobs["date"] == failed_date)
                    ]
                    
                    if matching_job.empty:
                        continue
                        
                    original_job_id = matching_job.iloc[0]["job_id"]
                    
                    completely_failed_jobs.append({
                        'job_id': original_job_id,
                        'scenario_id': scenario_id,
                        'run_id': run_id,
                        'season': season,
                        'date': failed_date,
                        'config': config_name,
                        'failure_reason': reason
                    })
        
        # Write completely failed jobs file if any exist
        if completely_failed_jobs:
            completely_failed_df = pd.DataFrame(completely_failed_jobs)
            completely_failed_file = os.path.join(save_dir, f"completely_failed_inpaint_jobs_{season}.txt")
            completely_failed_df[['job_id', 'scenario_id', 'run_id', 'season', 'date', 'config']].to_csv(
                completely_failed_file, index=False
            )
            print(f"Written {len(completely_failed_jobs)} completely failed jobs (zero successful dates) to: {completely_failed_file}")
            print(f"Completely failed models ({len(completely_failed_models)}): {sorted(completely_failed_models)}")


def create_plots(season: str, all_scores_t: pd.DataFrame, all_scores_rel: pd.DataFrame, 
                dataset: ForecastDataset, save_dir: str, model_missing_counts: Dict[str, int]):
    """
    Create visualization plots for the season results using evaluation_module.
    
    Args:
        season: Season identifier
        all_scores_t: Absolute scores dataframe
        all_scores_rel: Relative scores dataframe
        dataset: ForecastDataset with model info
        save_dir: Directory to save plots
        model_missing_counts: Dictionary of missing counts per model
    """
    # 1a) Heatmap of absolute WIS - US only
    create_heatmap_plot(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: Absolute WIS (US National only)", 
        f"{season}_absolute_wis_heatmap_US.png", 
        save_dir, model_missing_counts,
        location_filter="US", wis_type="wis_total"
    )

    # 1b) Heatmap of relative WIS - US only  
    create_heatmap_plot(
        all_scores_rel, dataset, Config.GROUP_COLORS,
        f"{season}: Relative WIS vs Baseline (US National only)",
        f"{season}_relative_wis_heatmap_US.png", 
        save_dir, model_missing_counts,
        center=1, vmin=0, vmax=2, location_filter="US", wis_type="wis_total"
    )

    # 1c) Heatmap of absolute WIS - All locations summed
    create_heatmap_plot(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: Absolute WIS (All 51 locations summed)",
        f"{season}_absolute_wis_heatmap_AllSum.png", 
        save_dir, model_missing_counts,
        location_filter="ALL", wis_type="wis_total"
    )

    # 1d) Heatmap of relative WIS - All locations summed
    create_heatmap_plot(
        all_scores_rel, dataset, Config.GROUP_COLORS,
        f"{season}: Relative WIS vs Baseline (All 51 locations summed)",
        f"{season}_relative_wis_heatmap_AllSum.png", 
        save_dir, model_missing_counts,
        center=1, vmin=0, vmax=2, location_filter="ALL", wis_type="wis_total"
    )

    # 2a) WIS components - US only
    create_component_plot(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: WIS Components (US National only)",
        f"{season}_wis_components_US.png",
        save_dir, model_missing_counts, location_filter="US"
    )

    # 2b) WIS components - All locations summed  
    create_component_plot(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: WIS Components (All 51 locations summed)",
        f"{season}_wis_components_AllSum.png",
        save_dir, model_missing_counts, location_filter="ALL"
    )
    
    # 3a) Absolute time series - US only
    create_time_series_plot(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: Absolute WIS Over Time (US National)",
        f"{season}_absolute_timeseries_US.png",
        save_dir, model_missing_counts,
        location_filter="US", wis_type="wis_total", is_relative=False
    )
    
    # 3b) Relative time series - US only
    create_time_series_plot(
        all_scores_rel, dataset, Config.GROUP_COLORS,
        f"{season}: Relative WIS Over Time (US National)",
        f"{season}_relative_timeseries_US.png", 
        save_dir, model_missing_counts,
        location_filter="US", wis_type="wis_total", is_relative=True
    )


def evaluate_season(season: str, dates: List[dt.date], save_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Main evaluation function for a single season.
    
    Args:
        season: Season identifier  
        dates: List of forecast dates to evaluate
        save_dir: Directory to save results and plots
        
    Returns:
        Dictionary containing evaluation results
        
    Raises:
        RuntimeError: If no models could be scored
    """
    logging.info(f"Starting evaluation for season {season} with {len(dates)} dates")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        dataset, model_missing_counts, failures = build_dataset_for_season(season, dates)
        if not dataset.records:
            raise RuntimeError(f"No forecasts found for season {season}")
        gt = load_ground_truth(season)
        all_scores_t = score_dataset(dataset, gt)
        if all_scores_t.empty:
            raise RuntimeError(f"Scoring produced no results for season {season}")
        logging.info(f"Successfully scored {len(all_scores_t['model'].unique())} models for season {season}")
    except Exception as e:
        logging.error(f"Error evaluating models for season {season}: {str(e)}")
        raise

    kept_models = sorted(all_scores_t["model"].unique())
    print(f"Kept models ({season}): {len(kept_models)}")
    
    # Report InfluPaint failures and create failed jobs file
    influpaint_failures = {k: v for k, v in failures.items() if k.startswith('i')}
    print(f"\nInfluPaint failures found: {len(influpaint_failures)} models")
    if influpaint_failures:
        print(f"\nDetailed InfluPaint failure report for {season}:")
        create_failed_jobs_file(season, influpaint_failures, save_dir)

    # Calculate relative scores vs baseline (use FluSight-baseline specifically)
    baseline_candidates = [m for m in kept_models if m == "FluSight-baseline" or m.startswith("FluSight-baseline")]
    baseline_model = baseline_candidates[0] if baseline_candidates else "FluSight-baseline"
    try:
        all_scores_rel = compute_relative_scores(all_scores_t, baseline_model)
    except Exception:
        # Fallback: no baseline present, just copy wis_total
        all_scores_rel = all_scores_t[all_scores_t["wis_type"] == "wis_total"].copy()

    # Create plots using evaluation_module
    create_plots(season, all_scores_t, all_scores_rel, dataset, save_dir, model_missing_counts)

    return {"all_scores": all_scores_t, "all_scores_rel": all_scores_rel, "dataset": dataset, "model_missing_counts": model_missing_counts}


# %%
# NOTEBOOK-STYLE FUNCTIONS FOR DIFFERENT EVALUATION APPROACHES


def evaluate_influpaint_vs_flusight():
    """
    Default evaluation: InfluPaint (green) vs FluSight (blue) by season.
    This is the main notebook cell for standard evaluation.
    """
    jobs = read_jobs()
    by_season = season_dates_from_jobs(jobs)
    results = {}
    
    for season, dates in by_season.items():
        if season not in Config.FLUSIGHT_BASES:
            print(f"Skipping {season}: no local FluSight data")
            continue
        
        save_dir = os.path.join("results", f"evaluate_{season}")
        print(f"Evaluating {season}: {len(dates)} dates → {save_dir}")
        results[season] = evaluate_season(season, dates, save_dir)
    
    return results


# %%
def evaluate_combined_seasons(seasons: List[str] = None, save_dir: str = "results/combined_seasons"):
    """
    Evaluate multiple seasons together using the same evaluation functions.
    
    Args:
        seasons: List of season names to combine (default: all available seasons)
        save_dir: Directory to save combined results
    """
    if seasons is None:
        seasons = list(Config.FLUSIGHT_BASES.keys())
    
    print(f"Evaluating combined seasons: {seasons}")
    
    # Collect data from all seasons
    all_records = []
    combined_missing_counts = {}
    all_failures = {}
    
    for season in seasons:
        if season not in Config.FLUSIGHT_BASES:
            print(f"Skipping {season}: no local FluSight data")
            continue
            
        jobs = read_jobs()
        season_jobs = jobs[jobs["season"] == season]
        dates = sorted(season_jobs["date"].unique().tolist())
        
        print(f"Loading {season}: {len(dates)} dates")
        
        # Collect records for this season
        flusight_recs, flusight_missing = collect_flusight_records(season, dates)
        influpaint_recs, influpaint_missing, influpaint_failures = collect_influpaint_records(season, dates, season_jobs)
        
        # Add to combined collections
        all_records.extend(flusight_recs + influpaint_recs)
        
        # Combine missing counts (prefix with season to avoid conflicts)
        for model, count in flusight_missing.items():
            combined_missing_counts[f"{season}_{model}"] = count
        for model, count in influpaint_missing.items():
            combined_missing_counts[f"{season}_{model}"] = count
            
        all_failures.update(influpaint_failures)
    
    if not all_records:
        raise RuntimeError("No forecast records found across all seasons")
    
    # Create combined dataset
    combined_dataset = ForecastDataset(records=all_records)
    
    # Use ground truth from 2024-2025 season only
    combined_gt_df = load_ground_truth("2024-2025")
    
    # Score the combined dataset
    print("Computing WIS scores...")
    all_scores_t = score_dataset(combined_dataset, combined_gt_df)
    
    if all_scores_t.empty:
        raise RuntimeError("Scoring produced no results")
    
    kept_models = sorted(all_scores_t["model"].unique())
    print(f"Kept models (combined): {len(kept_models)}")
    
    # Calculate relative scores vs baseline
    baseline_candidates = [m for m in kept_models if m == "FluSight-baseline" or m.startswith("FluSight-baseline")]
    baseline_model = baseline_candidates[0] if baseline_candidates else "FluSight-baseline"
    
    try:
        all_scores_rel = compute_relative_scores(all_scores_t, baseline_model)
    except Exception:
        # Fallback: no baseline present, just copy wis_total
        all_scores_rel = all_scores_t[all_scores_t["wis_type"] == "wis_total"].copy()
    
    # Create plots for combined seasons
    season_label = "_".join(seasons)
    print(f"Creating plots → {save_dir}")
    create_plots(f"Combined ({season_label})", all_scores_t, all_scores_rel, 
                combined_dataset, save_dir, combined_missing_counts)
    
    return {
        "all_scores": all_scores_t, 
        "all_scores_rel": all_scores_rel, 
        "dataset": combined_dataset, 
        "model_missing_counts": combined_missing_counts,
        "seasons": seasons
    }


if __name__ == "__main__":
    # Default evaluation when run as script
    evaluate_influpaint_vs_flusight()
    
    # Also run combined season evaluation
    print("\n" + "="*50)
    print("RUNNING COMBINED SEASON EVALUATION")
    print("="*50)
    evaluate_combined_seasons()

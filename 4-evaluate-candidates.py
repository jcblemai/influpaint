# %% [markdown]
# InfluPaint vs FluSight Evaluation Orchestrator
# This script orchestrates the evaluation process by:
# - Reading job dates and collecting all forecast data
# - Using evaluation_module for core evaluation and plotting logic
# - Supporting different grouping and coloring schemes
# - Generating comprehensive evaluation reports

# %%
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Import our evaluation module
from evaluate_plot import (
    ModelEvaluator,
    PlotConfig,
    GroupConfig,
    create_influpaint_vs_flusight_config,
    create_scenario_based_config,
    create_config_based_config,
)
import evaluate_deprecated as evaluate_deprecated
from evaluation_module import ForecastRecord, ForecastDataset, score_dataset, compute_relative_scores

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
    
    # Plotting configuration
    PLOT_COLORS = {
        'influpaint': 'green',
        'flusight': 'blue', 
        'missing': 'red',
        'baseline': 'red'
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
                        group="groupB",
                        season=season,
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
                        group="groupA",
                        season=season,
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


def create_plots(season: str, all_scores_t: pd.DataFrame, all_scores_rel: pd.DataFrame, save_dir: str, 
                model_missing_counts: Dict[str, int], group_config: GroupConfig = None, 
                evaluator: ModelEvaluator = None):
    """
    Create visualization plots for the season results using the evaluation module.
    
    Args:
        season: Season identifier
        all_scores_t: Absolute scores dataframe
        all_scores_rel: Relative scores dataframe  
        save_dir: Directory to save plots
        model_missing_counts: Dictionary of missing counts per model
        group_config: Configuration for model grouping and colors
        evaluator: ModelEvaluator instance (creates default if None)
    """
    if evaluator is None:
        evaluator = ModelEvaluator()
    
    # 1a) Heatmap of absolute WIS - US only
    us_abs = all_scores_t[(all_scores_t["location"] == "US") & (all_scores_t["wis_type"] == "wis_total")]
    if len(us_abs) > 0:
        tp_us_abs = (
            us_abs
            .pivot_table(index=["model"], columns=["forecast_date", "target"], values="value", aggfunc="mean")
            .fillna(np.nan)
        )
        # Remove models that have no valid US data (all NaN rows)
        tp_us_abs = tp_us_abs.dropna(how='all')
        
        evaluator.create_heatmap_plot(tp_us_abs, f"{season}: Absolute WIS (US National only)", 
                                     f"{season}_absolute_wis_heatmap_US.png", save_dir, 
                                     model_missing_counts, group_config)

    # 1b) Heatmap of relative WIS - US only
    us_only = all_scores_rel[all_scores_rel["location"] == "US"]
    if len(us_only) > 0:
        tp_us = (
            us_only
            .pivot_table(index=["model"], columns=["forecast_date", "target"], values="value", aggfunc="mean")
            .fillna(np.nan)
        )
        # Remove models that have no valid US data (all NaN rows)
        tp_us = tp_us.dropna(how='all')
        
        evaluator.create_heatmap_plot(tp_us, f"{season}: Relative WIS vs Baseline (US National only)", 
                                     f"{season}_relative_wis_heatmap_US.png", save_dir, 
                                     model_missing_counts, group_config, center=1, vmin=0, vmax=2)

    # 1c) Heatmap of absolute WIS - All locations summed  
    abs_all = all_scores_t[all_scores_t["wis_type"] == "wis_total"]
    tp_abs_all = (
        abs_all
        .groupby(["model", "forecast_date", "target"], as_index=False)["value"].sum()
        .pivot_table(index=["model"], columns=["forecast_date", "target"], values="value", aggfunc="mean")
        .fillna(np.nan)
    )
    
    evaluator.create_heatmap_plot(tp_abs_all, f"{season}: Absolute WIS (All 51 locations summed)", 
                                 f"{season}_absolute_wis_heatmap_AllSum.png", save_dir, 
                                 model_missing_counts, group_config)

    # 1d) Heatmap of relative WIS - All locations summed
    tp_all = (
        all_scores_rel
        .groupby(["model", "forecast_date", "target"], as_index=False)["value"].sum()
        .pivot_table(index=["model"], columns=["forecast_date", "target"], values="value", aggfunc="mean")
        .fillna(np.nan)
    )
    
    evaluator.create_heatmap_plot(tp_all, f"{season}: Relative WIS vs Baseline (All 51 locations summed)", 
                                 f"{season}_relative_wis_heatmap_AllSum.png", save_dir, 
                                 model_missing_counts, group_config, center=1, vmin=0, vmax=2)

    # 2a) WIS components - US only 
    us_scores = all_scores_t[all_scores_t["location"] == "US"]
    if len(us_scores) > 0:
        evaluator.create_component_plots(us_scores, f"{season}", f"{season}_US", save_dir, 
                                        model_missing_counts, group_config)

    # 2b) WIS components - All locations summed
    evaluator.create_component_plots(all_scores_t, f"{season}", f"{season}_AllSum", save_dir, 
                                    model_missing_counts, group_config)
    
    # 3) Time series plots for this season
    # 3a) Absolute time series - US only
    us_timeseries = all_scores_t[
        (all_scores_t["location"] == "US") & 
        (all_scores_t["wis_type"] == "wis_total")
    ].copy()
    if not us_timeseries.empty:
        us_timeseries["season"] = season  # Add season column
        evaluator.create_time_series_plot(
            us_timeseries, 
            f"{season}: Absolute WIS Over Time (US National)",
            f"{season}_absolute_timeseries_US.png",
            save_dir, 
            model_missing_counts, 
            group_config, 
            is_relative=False
        )
    
    # 3b) Relative time series - US only
    us_rel_timeseries = all_scores_rel[all_scores_rel["location"] == "US"].copy()
    if not us_rel_timeseries.empty:
        us_rel_timeseries["season"] = season  # Add season column
        evaluator.create_time_series_plot(
            us_rel_timeseries,
            f"{season}: Relative WIS Over Time (US National)",
            f"{season}_relative_timeseries_US.png", 
            save_dir,
            model_missing_counts,
            group_config,
            is_relative=True
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

    # all_scores_t is already tidy from evaluation_module.score_dataset

    # Calculate relative scores vs baseline (use Flusight-Baseline specifically)
    baseline_candidates = [m for m in kept_models if m == "FluSight-baseline" or m.startswith("FluSight-baseline")]
    baseline_model = baseline_candidates[0] if baseline_candidates else "FluSight-baseline"
    try:
        all_scores_rel = compute_relative_scores(all_scores_t, baseline_model)
    except Exception:
        # Fallback: no baseline present, just copy wis_total
        all_scores_rel = all_scores_t[all_scores_t["wis_type"] == "wis_total"].copy()

    # Create plots with default grouping (InfluPaint vs FluSight)
    default_group_config = create_influpaint_vs_flusight_config()
    evaluator = ModelEvaluator()
    create_plots(season, all_scores_t, all_scores_rel, save_dir, model_missing_counts, 
                default_group_config, evaluator)

    return {"all_scores": all_scores_t, "all_scores_rel": all_scores_rel, "model_missing_counts": model_missing_counts}


# %%
def create_combined_plots(all_seasons_results: Dict[str, Dict], save_dir: str):
    """Create combined plots across all seasons."""
    evaluator = ModelEvaluator()
    evaluator.create_combined_plots(all_seasons_results, save_dir)
    return

def evaluate_with_custom_grouping(group_config: GroupConfig, suffix: str = ""):
    """
    Run evaluation with a custom grouping configuration.
    
    Args:
        group_config: Configuration for model grouping and colors
        suffix: Suffix to add to output directories
    
    Returns:
        Dictionary of evaluation results by season
    """
    jobs = read_jobs()
    by_season = season_dates_from_jobs(jobs)
    results = {}
    evaluator = ModelEvaluator()
    
    for season, dates in by_season.items():
        if season not in Config.FLUSIGHT_BASES:
            # Skip seasons not present locally
            continue
        
        # Create season-specific save directory with suffix
        save_dir = os.path.join("results", f"evaluate2_{season}{suffix}")
        print(f"Scoring season {season} on {len(dates)} dates → {save_dir}")
        
        # Use existing evaluate_season but with custom plotting
        season_results = evaluate_season(season, dates, save_dir)
        
        # Also create plots with custom grouping in separate directory
        custom_save_dir = os.path.join(save_dir, "custom_grouping")
        os.makedirs(custom_save_dir, exist_ok=True)
        
        # Get model missing counts from the season_results
        model_missing_counts = season_results.get("model_missing_counts", {})
        
        # Create custom plots
        create_plots(season, season_results["all_scores"], season_results["all_scores_rel"], 
                    custom_save_dir, model_missing_counts, group_config, evaluator)
        
        results[season] = season_results
    
    return results


def main():
    """
    Main function demonstrating different evaluation approaches.
    You can easily run different grouping schemes by calling evaluate_with_custom_grouping.
    """
    # Default evaluation (InfluPaint vs FluSight)
    jobs = read_jobs()
    by_season = season_dates_from_jobs(jobs)
    results = {}
    for season, dates in by_season.items():
        if season not in Config.FLUSIGHT_BASES:
            continue
        save_dir = os.path.join("results", f"evaluate2_{season}")
        print(f"Scoring season {season} on {len(dates)} dates → {save_dir}")
        results[season] = evaluate_season(season, dates, save_dir)


# Season-agnostic evaluation: evaluate an arbitrary list of dates spanning seasons
def evaluate_dates(dates: List[dt.date], save_dir: str, label: str = None) -> Dict[str, pd.DataFrame]:
    """
    Evaluate arbitrary forecast dates (potentially spanning seasons) using the
    same plotting logic. Dates are grouped by season internally; scoring is
    done per season and then concatenated.

    Args:
        dates: List of reference dates to evaluate
        save_dir: Output directory
        label: Optional label to use on plots; defaults to min-to-max date span

    Returns:
        Dict with keys: all_scores (tidy absolute), all_scores_rel (relative)
    """
    os.makedirs(save_dir, exist_ok=True)
    jobs = read_jobs()
    # Map provided dates to seasons via jobs file
    dates_set = set(dates)
    jobs_sub = jobs[jobs["date"].isin(dates_set)].copy()
    if jobs_sub.empty:
        raise RuntimeError("None of the provided dates matched the jobs configuration for season mapping.")

    by_season = {s: sorted(g["date"].unique().tolist()) for s, g in jobs_sub.groupby("season")}

    # Accumulate per-season results
    tidy_abs = []
    tidy_rel_parts = []
    combined_missing: Dict[str, int] = {}
    failures_all: Dict[str, Dict] = {}

    for season, season_dates in by_season.items():
        if season not in Config.FLUSIGHT_BASES:
            continue
        dataset, model_missing_counts, failures = build_dataset_for_season(season, season_dates)
        if not dataset.records:
            continue
        gt = load_ground_truth(season)
        scores_abs = score_dataset(dataset, gt)
        if scores_abs.empty:
            continue
        # Choose baseline
        kept_models = sorted(scores_abs["model"].unique())
        baseline_candidates = [m for m in kept_models if m == "FluSight-baseline" or m.startswith("FluSight-baseline")]
        baseline_model = baseline_candidates[0] if baseline_candidates else "FluSight-baseline"
        try:
            scores_rel = compute_relative_scores(scores_abs, baseline_model)
        except Exception:
            scores_rel = scores_abs[scores_abs["wis_type"] == "wis_total"].copy()

        tidy_abs.append(scores_abs)
        tidy_rel_parts.append(scores_rel)
        # Combine missing counts across seasons
        for k, v in model_missing_counts.items():
            combined_missing[k] = combined_missing.get(k, 0) + int(v)
        failures_all.update(failures)

    if not tidy_abs:
        raise RuntimeError("No scores could be computed for the provided dates.")

    all_scores_t = pd.concat(tidy_abs, ignore_index=True)
    all_scores_rel = pd.concat(tidy_rel_parts, ignore_index=True)

    # Plot with a neutral label
    if not label:
        dmin, dmax = min(dates), max(dates)
        label = f"custom-{dmin}_to_{dmax}"

    default_group_config = create_influpaint_vs_flusight_config()
    evaluator = ModelEvaluator()
    create_plots(label, all_scores_t, all_scores_rel, save_dir, combined_missing, default_group_config, evaluator)

    return {"all_scores": all_scores_t, "all_scores_rel": all_scores_rel, "model_missing_counts": combined_missing}
    
    # Create combined plots across all seasons
    if len(results) > 1:
        combined_save_dir = os.path.join("results", "evaluate2_combined")
        create_combined_plots(results, combined_save_dir)
    
    return results


def run_all_grouping_examples():
    """
    Example function showing how to run evaluation with different grouping schemes.
    """
    print("=== Running Scenario-Based Evaluation ===") 
    scenario_results = evaluate_with_custom_grouping(create_scenario_based_config(), "_scenarios")
    
    print("\\n=== Running Config-Based Evaluation ===")
    config_results = evaluate_with_custom_grouping(create_config_based_config(), "_configs")
    
    # Custom grouping example: Group by model architecture
    print("\\n=== Running Custom Architecture-Based Evaluation ===")
    def architecture_group_fn(model_name: str) -> str:
        if model_name.startswith('i') and '::' in model_name:
            parts = model_name.split("::")
            if len(parts) >= 2:
                # Extract model architecture (second part)
                arch = parts[1]
                if 'U500' in arch:
                    return 'unet_500'
                elif 'U1000' in arch:
                    return 'unet_1000'
                else:
                    return 'other_influpaint'
            else:
                return 'other_influpaint'
        else:
            return 'flusight'
    
    arch_config = GroupConfig(
        group_fn=architecture_group_fn,
        color_map={
            'unet_500': 'lightgreen', 
            'unet_1000': 'darkgreen',
            'other_influpaint': 'orange',
            'flusight': 'blue'
        }
    )
    arch_results = evaluate_with_custom_grouping(arch_config, "_architecture")
    
    return {
        'scenarios': scenario_results, 
        'configs': config_results,
        'architecture': arch_results
    }


if __name__ == "__main__":
    main()

# %% [markdown]
# InfluPaint vs FluSight Evaluation Orchestrator
# This notebook-style script orchestrates the evaluation process by:
# - Loading CSV forecast data for InfluPaint and FluSight models
# - Creating scoring_eval.ForecastDataset objects with proper grouping and display names
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

# Import our scoring package modules
import scoring.evaluation as scoring_eval
import scoring.plotting as scoring_plot

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


def collect_flusight_records(season: str, dates: List[dt.date]) -> Tuple[List[scoring_eval.ForecastRecord], Dict[str, int]]:
    """Collect FluSight forecasts into scoring_eval.ForecastRecord list and missing counts."""
    flusight_models = list_flusight_models(season)
    records: List[scoring_eval.ForecastRecord] = []
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
                    scoring_eval.ForecastRecord(
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



def collect_influpaint_records(season: str, dates: List[dt.date], season_jobs: pd.DataFrame) -> Tuple[List[scoring_eval.ForecastRecord], Dict[str, int]]:
    """Collect InfluPaint forecasts into scoring_eval.ForecastRecord list and missing counts."""
    configs = sorted(season_jobs["config"].unique().tolist())
    model_paths: Dict[str, Dict[dt.date, str]] = {}
    for config in configs:
        for d in dates:
            matches = find_influpaint_csvs(Config.INPAINT_RES_BASE, config, d)
            for run_name, path in matches:
                model_paths.setdefault(run_name, {})[d] = path

    records: List[scoring_eval.ForecastRecord] = []
    missing_counts: Dict[str, int] = {}

    for run_name, by_date in model_paths.items():
        present = 0
        for d in dates:
            if d not in by_date:
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
                    continue
                records.append(
                    scoring_eval.ForecastRecord(
                        model=run_name,
                        group="influpaint",
                        display_name=create_display_name(run_name),
                        forecast_date=pd.to_datetime(d),
                        df=df,
                    )
                )
                present += 1
            except Exception:
                continue

        missing = len(dates) - present
        missing_counts[run_name] = missing

    keep = {m for m, miss in missing_counts.items() if miss <= Config.ALLOW_MISSING_DATES_PER_MODEL}
    records = [r for r in records if r.model in keep]
    missing_counts = {m: c for m, c in missing_counts.items() if m in keep}
    return records, missing_counts


def build_dataset_for_season(season: str, dates: List[dt.date]) -> Tuple[scoring_eval.ForecastDataset, Dict[str, int]]:
    """Load forecasts for both groups and return a scoring_eval.ForecastDataset plus missing counts."""
    jobs = read_jobs()
    season_jobs = jobs[jobs["season"] == season]
    flusight_recs, flusight_missing = collect_flusight_records(season, dates)
    influpaint_recs, influpaint_missing = collect_influpaint_records(season, dates, season_jobs)
    dataset = scoring_eval.ForecastDataset(records=flusight_recs + influpaint_recs)
    missing_counts = {**flusight_missing, **influpaint_missing}
    return dataset, missing_counts



def create_plots(season: str, results: scoring_eval.ScoringResults, all_scores_rel: pd.DataFrame, 
                dataset: scoring_eval.ForecastDataset, save_dir: str, missing_counts: Dict[str, int]):
    """
    Create visualization plots for the season results using evaluation_module.
    
    Args:
        season: Season identifier
        results: ScoringResults with forecast_metrics and model_metrics
        all_scores_rel: Relative scores dataframe
        dataset: scoring_eval.ForecastDataset with model info
        save_dir: Directory to save plots
        missing_counts: Dictionary of missing counts per model
    """
    all_scores_t = results.forecast_metrics
    # 1a) Heatmap of absolute WIS - US only
    scoring_plot.forecast_scores_heatmap(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: Absolute WIS (US National only)", 
        f"{season}_absolute_wis_heatmap_US.png", 
        save_dir, missing_counts,
        location_filter="US", scoring_metric="wis_total"
    )

    # 1b) Heatmap of relative WIS - US only  
    scoring_plot.forecast_scores_heatmap(
        all_scores_rel, dataset, Config.GROUP_COLORS,
        f"{season}: Relative WIS vs Baseline (US National only)",
        f"{season}_relative_wis_heatmap_US.png", 
        save_dir, missing_counts,
        center=1, vmin=0, vmax=2, location_filter="US", scoring_metric="wis_total"
    )

    # 1c) Heatmap of absolute WIS - All locations summed
    scoring_plot.forecast_scores_heatmap(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: Absolute WIS (All 51 locations summed)",
        f"{season}_absolute_wis_heatmap_AllSum.png", 
        save_dir, missing_counts,
        location_filter="ALL", scoring_metric="wis_total"
    )

    # 1d) Heatmap of relative WIS - All locations summed
    scoring_plot.forecast_scores_heatmap(
        all_scores_rel, dataset, Config.GROUP_COLORS,
        f"{season}: Relative WIS vs Baseline (All 51 locations summed)",
        f"{season}_relative_wis_heatmap_AllSum.png", 
        save_dir, missing_counts,
        center=1, vmin=0, vmax=2, location_filter="ALL", scoring_metric="wis_total"
    )

    # 2a) WIS components - US only
    scoring_plot.forecast_components_breakdown(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: WIS Components (US National only)",
        f"{season}_wis_components_US.png",
        save_dir, missing_counts, location_filter="US"
    )

    # 2b) WIS components - All locations summed  
    scoring_plot.forecast_components_breakdown(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: WIS Components (All 51 locations summed)",
        f"{season}_wis_components_AllSum.png",
        save_dir, missing_counts, location_filter="ALL"
    )
    
    # 3a) Absolute time series - US only
    scoring_plot.forecast_performance_timeseries(
        all_scores_t, dataset, Config.GROUP_COLORS,
        f"{season}: Absolute WIS Over Time (US National)",
        f"{season}_absolute_timeseries_US.png",
        save_dir, missing_counts,
        location_filter="US", scoring_metric="wis_total", is_relative=False
    )
    
    # 3b) Relative time series - US only
    scoring_plot.forecast_performance_timeseries(
        all_scores_rel, dataset, Config.GROUP_COLORS,
        f"{season}: Relative WIS Over Time (US National)",
        f"{season}_relative_timeseries_US.png", 
        save_dir, missing_counts,
        location_filter="US", scoring_metric="wis_total", is_relative=True
    )
    
    # 4. Per-model metrics plots
    if not results.model_metrics.empty:
        # Model performance summary
        scoring_plot.model_performance_summary(
            results, dataset, Config.GROUP_COLORS,
            f"{season}: Model Performance Summary",
            f"{season}_model_performance_summary.png",
            save_dir
        )
        
        # Coverage metrics heatmaps (separate for 95% and 50%)
        scoring_plot.model_horizon_heatmap(
            results, dataset, Config.GROUP_COLORS,
            f"{season}: 95% Coverage by Horizon", 
            f"{season}_coverage_95_heatmap.png",
            save_dir, metric="coverage_95"
        )
        
        scoring_plot.model_horizon_heatmap(
            results, dataset, Config.GROUP_COLORS,
            f"{season}: 50% Coverage by Horizon",
            f"{season}_coverage_50_heatmap.png", 
            save_dir, metric="coverage_50"
        )
        
        # Completion rate heatmap (separate from coverage)
        scoring_plot.model_horizon_heatmap(
            results, dataset, Config.GROUP_COLORS,
            f"{season}: Completion Rate by Horizon",
            f"{season}_completion_rate_heatmap.png",
            save_dir, metric="completion_rate"
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
        dataset, model_missing_counts = build_dataset_for_season(season, dates)
        if not dataset.records:
            raise RuntimeError(f"No forecasts found for season {season}")
        gt = load_ground_truth(season)
        
        # Import per-model metrics from MetricRegistry
        from scoring.evaluation import MetricRegistry
        per_model_metrics = [
            MetricRegistry.COVERAGE_95,
            MetricRegistry.COVERAGE_95_GAP,
            MetricRegistry.COVERAGE_50,
            MetricRegistry.COVERAGE_50_GAP,
            MetricRegistry.COMPLETION_RATE
        ]
        
        results = scoring_eval.score_dataset(dataset, gt, dates, metrics=per_model_metrics)
        all_scores_t = results.forecast_metrics
        missing_counts = results.meta['missing_counts']
        
        if all_scores_t.empty:
            raise RuntimeError(f"Scoring produced no results for season {season}")
        logging.info(f"Successfully scored {len(all_scores_t['model'].unique())} models for season {season}")
    except Exception as e:
        logging.error(f"Error evaluating models for season {season}: {str(e)}")
        raise

    kept_models = sorted(all_scores_t["model"].unique())
    print(f"Kept models ({season}): {len(kept_models)}")

    # Calculate relative scores vs baseline (use FluSight-baseline specifically)
    baseline_candidates = [m for m in kept_models if m == "FluSight-baseline" or m.startswith("FluSight-baseline")]
    baseline_model = baseline_candidates[0] if baseline_candidates else "FluSight-baseline"
    try:
        all_scores_rel = scoring_eval.compute_relative_scores(all_scores_t, baseline_model)
    except Exception:
        # Fallback: no baseline present, just copy wis_total
        all_scores_rel = all_scores_t[all_scores_t["scoring_metric"] == "wis_total"].copy()

    # Create plots using evaluation_module
    create_plots(season, results, all_scores_rel, dataset, save_dir, missing_counts)

    return {"results": results, "all_scores_rel": all_scores_rel, "dataset": dataset, "model_missing_counts": model_missing_counts}


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
    all_expected_dates = []
    
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
        influpaint_recs, influpaint_missing = collect_influpaint_records(season, dates, season_jobs)
        
        # Add to combined collections
        all_records.extend(flusight_recs + influpaint_recs)
        all_expected_dates.extend(dates)
        
        # Combine missing counts (prefix with season to avoid conflicts)
        for model, count in flusight_missing.items():
            combined_missing_counts[f"{season}_{model}"] = count
        for model, count in influpaint_missing.items():
            combined_missing_counts[f"{season}_{model}"] = count
    
    if not all_records:
        raise RuntimeError("No forecast records found across all seasons")
    
    # Create combined dataset
    combined_dataset = scoring_eval.ForecastDataset(records=all_records)
    
    # Use ground truth from 2024-2025 season only
    combined_gt_df = load_ground_truth("2024-2025")
    
    # Score the combined dataset
    print("Computing WIS scores...")
    
    # Import per-model metrics from MetricRegistry
    from scoring.evaluation import MetricRegistry
    per_model_metrics = [
        MetricRegistry.COVERAGE_95,
        MetricRegistry.COVERAGE_95_GAP,
        MetricRegistry.COVERAGE_50,
        MetricRegistry.COVERAGE_50_GAP,
        MetricRegistry.COMPLETION_RATE
    ]
    
    results = scoring_eval.score_dataset(combined_dataset, combined_gt_df, all_expected_dates, metrics=per_model_metrics)
    all_scores_t = results.forecast_metrics
    validation_missing_counts = results.meta['missing_counts']
    
    if all_scores_t.empty:
        raise RuntimeError("Scoring produced no results")
    
    kept_models = sorted(all_scores_t["model"].unique())
    print(f"Kept models (combined): {len(kept_models)}")
    
    # Calculate relative scores vs baseline
    baseline_candidates = [m for m in kept_models if m == "FluSight-baseline" or m.startswith("FluSight-baseline")]
    baseline_model = baseline_candidates[0] if baseline_candidates else "FluSight-baseline"
    
    try:
        all_scores_rel = scoring_eval.compute_relative_scores(all_scores_t, baseline_model)
    except Exception:
        # Fallback: no baseline present, just copy wis_total
        all_scores_rel = all_scores_t[all_scores_t["scoring_metric"] == "wis_total"].copy()
    
    # Create plots for combined seasons
    season_label = "_".join(seasons)
    print(f"Creating plots → {save_dir}")
    create_plots(f"Combined ({season_label})", results, all_scores_rel, 
                combined_dataset, save_dir, validation_missing_counts)
    
    return {
        "results": results, 
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

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
from evaluation_module import (
    ModelEvaluator, PlotConfig, GroupConfig,
    create_influpaint_vs_flusight_config,
    create_scenario_based_config,
    create_config_based_config
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
# WIS scoring (adapted from evaluate.py)
def weighted_interval_score_fast(
    observations,
    alphas,
    q_dict,
    weights=None,
    percent=False,
    check_consistency=True,
):
    if weights is None:
        weights = np.array(alphas) / 2

    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        raise ValueError("Alpha values must be sorted in ascending order.")

    reversed_weights = list(reversed(weights))

    lower_quantiles = [q_dict.get(alpha / 2) for alpha in alphas]
    upper_quantiles = [q_dict.get(1 - (alpha / 2)) for alpha in reversed(alphas)]
    if any(q is None for q in lower_quantiles) or any(q is None for q in upper_quantiles):
        raise ValueError("Quantile dictionary does not include all necessary quantiles.")

    lower_quantiles = np.vstack(lower_quantiles)
    upper_quantiles = np.vstack(upper_quantiles)

    if check_consistency and np.any(np.diff(np.vstack((lower_quantiles, upper_quantiles)), axis=0) < 0):
        raise ValueError("Quantiles are not consistent.")

    lower_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    upper_q_alphas = (2 / np.array(list(reversed(alphas)))).reshape((-1, 1))

    sharpnesses = np.flip(upper_quantiles, axis=0) - lower_quantiles

    lower_calibrations = np.clip(lower_quantiles - observations, a_min=0, a_max=None) * lower_q_alphas
    upper_calibrations = np.clip(observations - upper_quantiles, a_min=0, a_max=None) * upper_q_alphas
    calibrations = lower_calibrations + np.flip(upper_calibrations, axis=0)
    upper_calibrations = np.flip(upper_calibrations, axis=0)

    if percent:
        # Not supported here to keep parity with evaluate.py
        raise ValueError("percent=True not supported with calibration split")

    totals = sharpnesses + calibrations

    weights = np.array(weights).reshape((-1, 1))
    sharpnesses_weighted = sharpnesses * weights
    calibrations_weighted = calibrations * weights
    upper_calibrations_weighted = upper_calibrations * weights
    lower_calibrations_weighted = lower_calibrations * weights
    totals_weighted = totals * weights

    weights_sum = np.sum(weights)
    sharpnesses_final = np.sum(sharpnesses_weighted, axis=0) / weights_sum
    calibrations_final = np.sum(calibrations_weighted, axis=0) / weights_sum
    upper_calibrations_final = np.sum(upper_calibrations_weighted, axis=0) / weights_sum
    lower_calibrations_final = np.sum(lower_calibrations_weighted, axis=0) / weights_sum
    totals_final = np.sum(totals_weighted, axis=0) / weights_sum

    return (
        totals_final,
        sharpnesses_final,
        calibrations_final,
        lower_calibrations_final,
        upper_calibrations_final,
    )


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


def score_Nwk_forecasts_hub(gt: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compute WIS and components for specified horizons in hub-format forecasts."""
    # Restrict GT to locations/dates present in forecasts
    f = filter_forecast_df(forecasts)
    locations = sorted(f["location"].unique())
    target_dates = sorted(f["target_end_date"].unique())

    gt2 = gt[(gt["location"].isin(locations)) & (gt["date"].isin(target_dates))].copy()
    gt_piv = gt2.pivot(index="date", columns="location", values="value").sort_index()

    # alphas from available lower quantiles
    qvals = sorted(f["output_type_id"].unique())
    lower = [q for q in qvals if q <= 0.5]
    alphas = np.array(lower) * 2

    # Use only target dates present in ground truth
    gt_dates = set(gt_piv.index)
    available_dates = [d for d in target_dates if d in gt_dates]
    if not available_dates:
        # Helpful diagnostics
        raise RuntimeError(
            "No valid target dates aligned with ground truth. "
            f"Forecast dates: {len(target_dates)}, GT dates intersect: 0"
        )
    all_targets = []
    for target_date in available_dates:
        sub = f[f["target_end_date"] == target_date]
        q_dict = {}
        q_levels = sorted(sub["output_type_id"].unique())
        for q in q_levels:
            vals = (
                sub[sub["output_type_id"] == q]
                .pivot(index="target_end_date", columns="location", values="value")
                .reindex(columns=gt_piv.columns)
                .loc[target_date]
                .to_numpy()
            )
            q_dict[float(q)] = vals

        obs = gt_piv.loc[target_date].to_numpy()
        # mask to drop locations with missing obs or quantiles
        masks = [~pd.isna(obs)]
        for q in q_levels:
            masks.append(~pd.isna(q_dict[float(q)]))
        valid_mask = np.logical_and.reduce(masks)
        if not np.any(valid_mask):
            # Nothing valid for this date, skip
            continue
        obs_v = obs[valid_mask]
        q_dict_v = {float(q): q_dict[float(q)][valid_mask] for q in q_levels}
        (wis_total, wis_sharpness, wis_calibration, underprediction, overprediction) = weighted_interval_score_fast(
            observations=obs_v,
            alphas=alphas,
            q_dict=q_dict_v,
            weights=alphas / 2,
        )

        # Build a human-readable target label from the unique horizon
        try:
            uniq_h = pd.unique(sub["horizon"]).tolist()
            h_label = int(uniq_h[0]) if len(uniq_h) == 1 else None
        except Exception:
            h_label = None

        df = pd.DataFrame(
            [wis_total, wis_sharpness, wis_calibration, underprediction, overprediction],
            index=["wis_total", "wis_sharpness", "wis_calibration", "wis_underprediction", "wis_overprediction"],
            columns=np.array(gt_piv.columns)[valid_mask],
        )
        df["target"] = f"{h_label} wk ahead" if h_label is not None else ""
        df["target_end_date"] = target_date
        all_targets.append(df)

    if not all_targets:
        raise RuntimeError("No valid target dates aligned with ground truth.")
    return pd.concat(all_targets).reset_index(names="wis_type").set_index(["target", "target_end_date"])


# %%
# Plotting helper functions
def determine_model_color(model_name: str) -> str:
    """Determine the color for a model based on its type."""
    if model_name.startswith('i') and '::' in model_name:
        return Config.PLOT_COLORS['influpaint']
    else:
        return Config.PLOT_COLORS['flusight']


def create_model_labels_with_missing(models: List[str], model_missing_counts: Dict[str, int]) -> List[str]:
    """Create y-tick labels with missing counts."""
    labels = []
    for model in models:
        missing_count = model_missing_counts.get(model, 0)
        if missing_count > 0:
            if len(model) > 60:
                # Split on "::" and wrap across lines for long model names
                parts = model.split("::")
                if len(parts) >= 6:
                    line1 = "::".join(parts[:2])
                    line2 = "::".join(parts[2:5]) 
                    line3 = "::".join(parts[5:])
                    labels.append(f"{line1}\n{line2}\n{line3} missing:{missing_count}")
                else:
                    labels.append(f"{model} missing:{missing_count}")
            else:
                labels.append(f"{model} missing:{missing_count}")
        else:
            if len(model) > 60:
                parts = model.split("::")
                if len(parts) >= 6:
                    line1 = "::".join(parts[:2])
                    line2 = "::".join(parts[2:5]) 
                    line3 = "::".join(parts[5:])
                    labels.append(f"{line1}\n{line2}\n{line3}")
                else:
                    labels.append(model)
            else:
                labels.append(model)
    return labels


def apply_model_colors_to_labels(ax, model_names: List[str], model_missing_counts: Dict[str, int]):
    """Apply colors to y-tick labels based on model type and missing data."""
    for i, label in enumerate(ax.get_yticklabels()):
        model_name = model_names[i]
        text = label.get_text()
        
        if "missing:" in text:
            label.set_color(Config.PLOT_COLORS['missing'])
        else:
            model_color = determine_model_color(model_name)
            label.set_color(model_color)


def add_missing_count_annotations(ax, model_names: List[str], model_missing_counts: Dict[str, int]):
    """Add red missing count annotations to the right of y-labels."""
    for i, label in enumerate(ax.get_yticklabels()):
        model_name = model_names[i]
        text = label.get_text()
        
        if "missing:" in text:
            # Split into lines and color the missing line red
            lines = text.split('\n')
            clean_text = '\n'.join([line for line in lines if not line.startswith('missing:')])
            missing_line = next((line for line in lines if line.startswith('missing:')), None)
            
            label.set_text(clean_text)
            if missing_line:
                # Add red text for missing count
                ax.text(-0.02, i, missing_line, color=Config.PLOT_COLORS['missing'], 
                       va='center', fontsize=label.get_fontsize(), ha='right', 
                       transform=ax.get_yaxis_transform())


def create_heatmap_plot(data: pd.DataFrame, title: str, filename: str, save_dir: str, 
                       model_missing_counts: Dict[str, int], cmap: str = "viridis", 
                       center: float = None, vmin: float = None, vmax: float = None):
    """Create a standardized heatmap plot."""
    if data.empty or data.shape[1] == 0:
        return
        
    # Sort by total score
    order = data.sum(axis=1, skipna=True).sort_values().index
    data_sorted = data.loc[order]
    
    # Create y-tick labels with missing counts
    ytick_labels = create_model_labels_with_missing(data_sorted.index.tolist(), model_missing_counts)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(12, data_sorted.shape[1] * 0.5), max(8, data_sorted.shape[0] * 0.4)))
    
    # Create heatmap
    if center is not None:
        cmap_obj = sns.diverging_palette(150, 300, as_cmap=True, center="light")
        sns.heatmap(data_sorted, annot=False, fmt=".2f", linewidths=0.5, ax=ax, 
                   center=center, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(data_sorted, annot=False, fmt=".2f", linewidths=0.5, ax=ax, cmap=cmap)
    
    # Set custom y-tick labels
    ax.set_yticklabels([label.replace(" missing:", "\nmissing:") for label in ytick_labels])
    
    # Apply colors and missing annotations
    apply_model_colors_to_labels(ax, data_sorted.index.tolist(), model_missing_counts)
    add_missing_count_annotations(ax, data_sorted.index.tolist(), model_missing_counts)
    
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig)


def evaluate_flusight_models(season: str, dates: List[dt.date], gt: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Evaluate FluSight models for a season."""
    flusight_models = list_flusight_models(season)
    scores = {}
    
    for model in flusight_models:
        present_dates = []
        model_scores = {}
        
        for d in dates:
            try:
                df = load_flusight_forecast(season, model, d)
                df = filter_forecast_df(df)
                
                # Check quantile completeness
                required_q = Config.REQUIRED_QUANTILES
                have = sorted(df["output_type_id"].unique().tolist())
                missing = sorted(set(np.round(required_q, 6)) - set(np.round(have, 6)))
                if missing:
                    continue
                    
                wis_all = score_Nwk_forecasts_hub(gt, df)
                model_scores[d] = wis_all
                present_dates.append(d)
                
            except (FileNotFoundError, RuntimeError, AssertionError):
                continue
        
        # Keep model if within missing date threshold
        missing_count = len(dates) - len(present_dates)
        if missing_count <= Config.ALLOW_MISSING_DATES_PER_MODEL and present_dates:
            key = f"{model}!{missing_count}!"
            scores[key] = pd.concat(model_scores, names=["forecast_date", "target", "target_end_date"])
        else:
            print(f"Dropping FluSight model {model}: missing={missing_count} (> {Config.ALLOW_MISSING_DATES_PER_MODEL}) or no valid scores")
    
    return scores


def evaluate_influpaint_models(season: str, dates: List[dt.date], gt: pd.DataFrame, season_jobs: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Evaluate InfluPaint models for a season. Returns (scores, failures)."""
    configs = sorted(season_jobs["config"].unique().tolist())
    scores = {}
    failures = {}
    
    # Find what CSVs actually exist and build proper run names
    actual_models = {}  # proper_run_name -> {scores, failures, scenario_id, config, run_id}
    
    for config in configs:
        for d in dates:
            matches = find_influpaint_csvs(Config.INPAINT_RES_BASE, config, d)
            for run_name, path in matches:
                if run_name not in actual_models:
                    # Extract scenario_id from run_name to link back to expected jobs
                    try:
                        scenario_id = int(run_name.split("::")[0][1:])  # Remove 'i' prefix
                        config_name = run_name.split("::")[-1]
                        
                        # Find matching run_id from jobs
                        matching = season_jobs[
                            (season_jobs["scenario_id"] == scenario_id) & 
                            (season_jobs["config"] == config_name)
                        ]
                        run_id = matching.iloc[0]["run_id"] if not matching.empty else None
                        
                        actual_models[run_name] = {
                            "scores": {}, 
                            "failures": {},
                            "scenario_id": scenario_id,
                            "config": config_name, 
                            "run_id": run_id
                        }
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse scenario_id from {run_name}")
                        continue
                
                try:
                    df = pd.read_csv(path)
                    # Normalize column names
                    if "output_type" not in df.columns and "type" in df.columns:
                        df = df.rename(columns={"type": "output_type"})
                    if "output_type_id" not in df.columns and "quantile" in df.columns:
                        df = df.rename(columns={"quantile": "output_type_id"})
                    
                    df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
                    df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
                    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
                    df = filter_forecast_df(df)
                    
                    # Check quantile completeness
                    required_q = Config.REQUIRED_QUANTILES
                    have = sorted(df["output_type_id"].unique().tolist())
                    missing = sorted(set(np.round(required_q, 6)) - set(np.round(have, 6)))
                    if missing:
                        actual_models[run_name]["failures"][d] = f"Missing quantiles {missing[:3]}{'...' if len(missing) > 3 else ''}"
                        continue
                    
                    wis_all = score_Nwk_forecasts_hub(gt, df)
                    actual_models[run_name]["scores"][d] = wis_all
                    
                except Exception as e:
                    actual_models[run_name]["failures"][d] = f"Error: {str(e)[:50]}..."
    
    # Add completely missing scenarios
    all_expected_jobs = season_jobs[["scenario_id", "config"]].drop_duplicates()
    found_scenarios = set()
    for run_name in actual_models.keys():
        try:
            scenario_id = int(run_name.split("::")[0][1:])
            config_name = run_name.split("::")[-1] 
            found_scenarios.add((scenario_id, config_name))
        except:
            continue
    
    for _, row in all_expected_jobs.iterrows():
        scenario_id = row["scenario_id"]
        config = row["config"]
        
        if (scenario_id, config) not in found_scenarios:
            scenario_jobs_subset = season_jobs[
                (season_jobs["scenario_id"] == scenario_id) & 
                (season_jobs["config"] == config)
            ]
            if not scenario_jobs_subset.empty:
                run_id = scenario_jobs_subset.iloc[0]["run_id"]
                missing_run_name = f"i{scenario_id}::missing_model::{config}"
                
                failure_details = {}
                for d in dates:
                    if d in scenario_jobs_subset["date"].values:
                        failure_details[d] = f"No CSV found for scenario {scenario_id}, config {config} on {d}"
                
                if failure_details:
                    actual_models[missing_run_name] = {
                        "scores": {},
                        "failures": failure_details,
                        "scenario_id": scenario_id,
                        "config": config,
                        "run_id": run_id
                    }
    
    # Process each model: add to scores if good enough, track all failures
    for run_name, data in actual_models.items():
        present_count = len(data["scores"])
        total_expected = len([d for d in dates if d in season_jobs[
            (season_jobs["scenario_id"] == data["scenario_id"]) & 
            (season_jobs["config"] == data["config"])
        ]["date"].values])
        missing_count = total_expected - present_count
        
        # Track failures for ALL models (both kept and dropped)
        if missing_count > 0 or data["failures"]:
            failure_details = data["failures"].copy()
            # Add missing dates that had no CSV found and no recorded failure
            expected_dates_for_model = season_jobs[
                (season_jobs["scenario_id"] == data["scenario_id"]) & 
                (season_jobs["config"] == data["config"]) &
                (season_jobs["season"] == season)
            ]["date"].values
            
            for d in expected_dates_for_model:
                if d not in data["scores"] and d not in failure_details:
                    failure_details[d] = f"No CSV found for {run_name} on {d}"
            
            failures[run_name] = {
                'present_count': present_count,
                'missing_count': missing_count,
                'present_dates': sorted(data["scores"].keys()),
                'failure_details': failure_details,
                'kept': missing_count <= Config.ALLOW_MISSING_DATES_PER_MODEL and present_count > 0
            }
        
        # Keep model if within threshold
        if missing_count <= Config.ALLOW_MISSING_DATES_PER_MODEL and present_count > 0:
            key = f"{run_name}!{missing_count}!"
            scores[key] = pd.concat(data["scores"], names=["forecast_date", "target", "target_end_date"])
        else:
            print(f"Dropping InfluPaint model {run_name}: present={present_count}, missing={missing_count} (> {Config.ALLOW_MISSING_DATES_PER_MODEL})")
    
    return scores, failures


def evaluate_models(season: str, dates: List[dt.date]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """Evaluate all models (FluSight + InfluPaint) for a season. Returns (scores, failures)."""
    gt = load_ground_truth(season)
    
    # Get InfluPaint configs for this season
    jobs = read_jobs()
    season_jobs = jobs[jobs["season"] == season]
    
    # Evaluate FluSight models
    flusight_scores = evaluate_flusight_models(season, dates, gt)
    
    # Evaluate InfluPaint models  
    influpaint_scores, influpaint_failures = evaluate_influpaint_models(season, dates, gt, season_jobs)
    
    # Combine results
    scores = {}
    scores.update(flusight_scores)
    scores.update(influpaint_scores)
    
    return scores, influpaint_failures


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
        # Evaluate all models
        scores, failures = evaluate_models(season, dates)
        
        if not scores:
            raise RuntimeError(f"No models scored for season {season}")
        
        logging.info(f"Successfully scored {len(scores)} models for season {season}")
    except Exception as e:
        logging.error(f"Error evaluating models for season {season}: {str(e)}")
        raise

    # Combine all scores
    all_scores = pd.concat(scores, names=["model", "forecast_date", "target", "target_end_date"])
    
    # Log model counts
    kept_models = sorted(set(all_scores.reset_index()["model"]))
    print(f"Kept models ({season}): {len(kept_models)}")
    
    # Report InfluPaint failures and create failed jobs file
    influpaint_failures = {k: v for k, v in failures.items() if k.startswith('i')}
    print(f"\nInfluPaint failures found: {len(influpaint_failures)} models")
    if influpaint_failures:
        print(f"\nDetailed InfluPaint failure report for {season}:")
        create_failed_jobs_file(season, influpaint_failures, save_dir)

    # Clean up model names and extract missing counts for display
    model_missing_counts = {}
    clean_scores = {}
    for model_key, data in scores.items():
        # Extract missing count from model key (format: "model_name!missing_count!")
        if "!" in model_key:
            parts = model_key.split("!")
            clean_model_name = parts[0]
            missing_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        else:
            clean_model_name = model_key
            missing_count = 0
        
        model_missing_counts[clean_model_name] = missing_count
        clean_scores[clean_model_name] = data

    # Combine all scores with clean names
    all_scores = pd.concat(clean_scores, names=["model", "forecast_date", "target", "target_end_date"])
    
    # Transform to long format for plotting
    all_scores_t = all_scores.reset_index()
    id_vars = ["model", "forecast_date", "target", "target_end_date", "wis_type"]
    location_columns = [c for c in all_scores_t.columns if c not in id_vars]
    all_scores_t = pd.melt(
        all_scores_t,
        id_vars=id_vars,
        value_vars=location_columns,
        var_name="location",
        value_name="value",
    )

    # Calculate relative scores vs baseline (use Flusight-Baseline specifically)
    baseline_keys = sorted([m for m in all_scores_t["model"].unique() if "Flusight-baseline" in m or "FluSight-baseline" in m])
    baseline_model = baseline_keys[0] if baseline_keys else None
    if baseline_model is not None:
        baseline = all_scores_t[all_scores_t["model"] == baseline_model]
        all_scores_rel = all_scores_t[all_scores_t["wis_type"] == "wis_total"].copy()
        baseline_cols = ["forecast_date", "target", "target_end_date", "wis_type", "location"]
        # Use left merge to keep all rows from models even when baseline missing
        all_scores_rel = pd.merge(
            all_scores_rel,
            baseline[baseline_cols + ["value"]],
            on=baseline_cols,
            suffixes=("", "_baseline"),
            how="left",
        )
        all_scores_rel["value"] = all_scores_rel["value"] / all_scores_rel["value_baseline"]
        all_scores_rel = all_scores_rel.drop(columns=["value_baseline"])
    else:
        all_scores_rel = all_scores_t[all_scores_t["wis_type"] == "wis_total"].copy()

    # Create plots with default grouping (InfluPaint vs FluSight)
    default_group_config = create_influpaint_vs_flusight_config()
    evaluator = ModelEvaluator()
    create_plots(season, all_scores_t, all_scores_rel, save_dir, model_missing_counts, 
                default_group_config, evaluator)

    return {"all_scores": all_scores_t, "all_scores_rel": all_scores_rel}


# %%
def create_combined_plots(all_seasons_results: Dict[str, Dict], save_dir: str):
    """Create combined plots across all seasons."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine data from all seasons
    combined_scores = []
    combined_rel_scores = []
    all_model_missing_counts = {}
    
    for season, season_data in all_seasons_results.items():
        scores_df = season_data["all_scores"].copy()
        rel_df = season_data["all_scores_rel"].copy()
        
        # Add season column
        scores_df["season"] = season
        rel_df["season"] = season
        
        combined_scores.append(scores_df)
        combined_rel_scores.append(rel_df)
        
        # Extract missing counts from season results (from the individual season processing)
        # We'll need to track these from the individual season results
        pass
    
    # Combine all data
    all_scores_combined = pd.concat(combined_scores, ignore_index=True)
    all_scores_rel_combined = pd.concat(combined_rel_scores, ignore_index=True)
    
    # For combined plots, we'll aggregate missing counts across seasons
    # Count total missing dates per model across all seasons
    # We need to get this information from the original evaluation
    model_total_missing = {}
    
    # Get total expected dates across all seasons
    jobs = read_jobs()
    total_expected_dates_by_season = {}
    for season in all_seasons_results.keys():
        season_jobs = jobs[jobs["season"] == season]
        total_expected_dates_by_season[season] = len(season_jobs["date"].unique())
    
    # For each model that appears in the results, calculate missing dates
    all_models = set()
    for season_data in all_seasons_results.values():
        all_models.update(season_data["all_scores"]["model"].unique())
    
    for model in all_models:
        total_missing = 0
        total_expected = 0
        
        for season in all_seasons_results.keys():
            season_data = all_seasons_results[season]["all_scores"]
            season_models = season_data["model"].unique()
            expected_dates_this_season = total_expected_dates_by_season[season]
            
            if model in season_models:
                # Count actual dates for this model in this season
                model_dates = len(season_data[season_data["model"] == model]["forecast_date"].unique())
                missing_this_season = expected_dates_this_season - model_dates
                total_missing += missing_this_season
                total_expected += expected_dates_this_season
            else:
                # Model completely missing from this season
                total_missing += expected_dates_this_season
                total_expected += expected_dates_this_season
        
        model_total_missing[model] = total_missing
    
    print(f"Creating combined plots across {len(all_seasons_results)} seasons...")
    
    # 1a) Combined Absolute Heatmap - All locations summed, all seasons
    abs_combined = all_scores_combined[all_scores_combined["wis_type"] == "wis_total"]
    tp_abs_combined = (
        abs_combined
        .groupby(["model", "season", "forecast_date", "target"], as_index=False)["value"].sum()
        .pivot_table(index=["model"], columns=["season", "forecast_date", "target"], values="value", aggfunc="mean")
        .fillna(np.nan)
    )
    
    if tp_abs_combined.shape[1] > 0:
        order = tp_abs_combined.sum(axis=1, skipna=True).sort_values().index
        tp_abs_combined = tp_abs_combined.loc[order]
        
        # Create y-tick labels with combined missing counts
        ytick_labels = []
        for model in tp_abs_combined.index:
            missing_count = model_total_missing.get(model, 0)
            if missing_count > 0:
                ytick_labels.append(f"{model} missing:{missing_count}")
            else:
                ytick_labels.append(model)
        
        f, ax = plt.subplots(figsize=(max(16, tp_abs_combined.shape[1] * 0.3), max(10, tp_abs_combined.shape[0] * 0.4)))
        sns.heatmap(tp_abs_combined, annot=False, fmt=".2f", linewidths=0.5, ax=ax, cmap="viridis")
        
        # Color model names by type and add red missing counts
        ax.set_yticklabels([label.replace(" missing:", "\nmissing:") for label in ytick_labels])
        
        for i, label in enumerate(ax.get_yticklabels()):
            model_name = tp_abs_combined.index[i]
            text = label.get_text()
            
            # Determine model color: InfluPaint (green) vs FluSight (blue)
            model_color = 'green' if model_name.startswith('i') and '::' in model_name else 'blue'
            label.set_color(model_color)
            
            if "missing:" in text:
                # Split into lines and color the missing line red
                lines = text.split('\n')
                clean_text = '\n'.join([line for line in lines if not line.startswith('missing:')])
                missing_line = next((line for line in lines if line.startswith('missing:')), None)
                
                label.set_text(clean_text)
                if missing_line:
                    # Add red text for missing count
                    ax.text(-0.02, i, missing_line, color='red', va='center', 
                           fontsize=label.get_fontsize(), ha='right', transform=ax.get_yaxis_transform())
        
        ax.set_title("All Seasons Combined: Absolute WIS (All locations summed)")
        plt.tight_layout()
        f.savefig(os.path.join(save_dir, "all_seasons_absolute_wis_heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close(f)
    
    # 1b) Combined Relative Heatmap - All locations summed, all seasons
    tp_combined = (
        all_scores_rel_combined
        .groupby(["model", "season", "forecast_date", "target"], as_index=False)["value"].sum()
        .pivot_table(index=["model"], columns=["season", "forecast_date", "target"], values="value", aggfunc="mean")
        .fillna(np.nan)
    )
    
    if tp_combined.shape[1] > 0:
        order = tp_combined.sum(axis=1, skipna=True).sort_values().index
        tp_combined = tp_combined.loc[order]
        
        # Create y-tick labels with combined missing counts
        ytick_labels = []
        for model in tp_combined.index:
            missing_count = model_total_missing.get(model, 0)
            if missing_count > 0:
                ytick_labels.append(f"{model} missing:{missing_count}")
            else:
                ytick_labels.append(model)
        
        f, ax = plt.subplots(figsize=(max(16, tp_combined.shape[1] * 0.3), max(10, tp_combined.shape[0] * 0.4)))
        cmap = sns.diverging_palette(150, 300, as_cmap=True, center="light")
        sns.heatmap(tp_combined, annot=False, fmt=".2f", linewidths=0.5, ax=ax, center=1, cmap=cmap, vmin=0, vmax=2)
        
        # Color model names by type and add red missing counts
        ax.set_yticklabels([label.replace(" missing:", "\nmissing:") for label in ytick_labels])
        
        for i, label in enumerate(ax.get_yticklabels()):
            model_name = tp_combined.index[i]
            text = label.get_text()
            
            # Determine model color: InfluPaint (green) vs FluSight (blue)
            model_color = 'green' if model_name.startswith('i') and '::' in model_name else 'blue'
            label.set_color(model_color)
            
            if "missing:" in text:
                # Split into lines and color the missing line red
                lines = text.split('\n')
                clean_text = '\n'.join([line for line in lines if not line.startswith('missing:')])
                missing_line = next((line for line in lines if line.startswith('missing:')), None)
                
                label.set_text(clean_text)
                if missing_line:
                    # Add red text for missing count
                    ax.text(-0.02, i, missing_line, color='red', va='center', 
                           fontsize=label.get_fontsize(), ha='right', transform=ax.get_yaxis_transform())
        
        ax.set_title("All Seasons Combined: Relative WIS vs Baseline (All locations summed)")
        plt.tight_layout()
        f.savefig(os.path.join(save_dir, "all_seasons_relative_wis_heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close(f)
    
    # 2) Combined WIS Components - All locations summed, all seasons
    comp_combined = (
        all_scores_combined.groupby(["model", "wis_type"], as_index=False)["value"].sum()
        .pivot(index="model", columns="wis_type", values="value")
        .fillna(0.0)
    )
    comp_combined = comp_combined.sort_values("wis_total") if "wis_total" in comp_combined.columns else comp_combined
    to_plot_combined = comp_combined[[c for c in ["wis_total", "wis_sharpness", "wis_calibration", "wis_overprediction", "wis_underprediction"] if c in comp_combined.columns]]
    
    if not to_plot_combined.empty:
        fig, axes = plt.subplots(1, to_plot_combined.shape[1], figsize=(4 * to_plot_combined.shape[1], max(10, 0.3 * len(to_plot_combined))))
        if to_plot_combined.shape[1] == 1:
            axes = [axes]
            
        for ax, col in zip(axes, to_plot_combined.columns):
            ax.scatter(to_plot_combined[col], np.arange(len(to_plot_combined)), s=8)
            ax.set_yticks(np.arange(len(to_plot_combined)))
            
            # Create labels with combined missing counts
            labels = []
            for model in to_plot_combined.index:
                missing_count = model_total_missing.get(model, 0)
                
                if missing_count > 0:
                    if len(model) > 60:
                        # Split on "::" and wrap across 2-3 lines
                        parts = model.split("::")
                        if len(parts) >= 6:  # Full model with all components
                            line1 = "::".join(parts[:2])
                            line2 = "::".join(parts[2:5]) 
                            line3 = "::".join(parts[5:])
                            wrapped_label = f"{line1}\n{line2}\n{line3} missing:{missing_count}"
                        else:
                            wrapped_label = f"{model} missing:{missing_count}"
                    else:
                        wrapped_label = f"{model} missing:{missing_count}"
                else:
                    if len(model) > 60:
                        # Split on "::" and wrap across 2-3 lines
                        parts = model.split("::")
                        if len(parts) >= 6:  # Full model with all components
                            line1 = "::".join(parts[:2])
                            line2 = "::".join(parts[2:5]) 
                            line3 = "::".join(parts[5:])
                            wrapped_label = f"{line1}\n{line2}\n{line3}"
                        else:
                            wrapped_label = model
                    else:
                        wrapped_label = model
                labels.append(wrapped_label)
            
            ax.set_yticklabels(labels, fontsize=6)
            
            # Color based on model type and missing data
            for i, label in enumerate(ax.get_yticklabels()):
                model_name = to_plot_combined.index[i]
                
                if "missing:" in label.get_text():
                    # Color the entire label red if it has missing data
                    label.set_color('red')
                else:
                    # Color based on model type: InfluPaint (green) vs FluSight (blue)
                    model_color = 'green' if model_name.startswith('i') and '::' in model_name else 'blue'
                    label.set_color(model_color)
                
            ax.set_title(col)
        
        fig.suptitle("All Seasons Combined: WIS Components (All locations summed)")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "all_seasons_wis_components.png"), dpi=200)
        plt.close(fig)
    
    # 3) Stacked plot - WIS components breakdown
    comp_combined_stack = (
        all_scores_combined.groupby(["model", "wis_type"], as_index=False)["value"].sum()
        .pivot(index="model", columns="wis_type", values="value")
        .fillna(0.0)
    )
    
    # Get components for stacking (excluding total and calibration since calibration includes both over/under)
    available_components = [c for c in Config.STACK_COMPONENTS if c in comp_combined_stack.columns]
    
    if available_components and not comp_combined_stack.empty:
        # Sort by total WIS
        if "wis_total" in comp_combined_stack.columns:
            comp_combined_stack = comp_combined_stack.sort_values("wis_total")
        
        # Select top models for readability (e.g., top 20)
        if len(comp_combined_stack) > 20:
            comp_combined_stack = comp_combined_stack.tail(20)  # Get worst performing (highest WIS)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(comp_combined_stack) * 0.4)))
        
        # Create stacked bar chart
        bottom = np.zeros(len(comp_combined_stack))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']  # Different colors for components
        
        for i, component in enumerate(available_components):
            values = comp_combined_stack[component].values
            bars = ax.barh(range(len(comp_combined_stack)), values, left=bottom, 
                          label=component.replace('wis_', '').title(), color=colors[i % len(colors)])
            bottom += values
        
        # Customize plot
        ax.set_yticks(range(len(comp_combined_stack)))
        
        # Create labels with missing counts
        labels = []
        for model in comp_combined_stack.index:
            missing_count = model_total_missing.get(model, 0)
            if len(model) > 60:
                # Simplified label for stacked plot
                parts = model.split("::")
                if len(parts) >= 2:
                    short_label = f"{parts[0]}::{parts[-1]}"  # scenario::config
                else:
                    short_label = model
            else:
                short_label = model
                
            if missing_count > 0:
                labels.append(f"{short_label} missing:{missing_count}")
            else:
                labels.append(short_label)
        
        ax.set_yticklabels(labels, fontsize=8)
        
        # Color labels
        for i, label in enumerate(ax.get_yticklabels()):
            model_name = comp_combined_stack.index[i]
            if "missing:" in label.get_text():
                label.set_color('red')
            else:
                model_color = 'green' if model_name.startswith('i') and '::' in model_name else 'blue'
                label.set_color(model_color)
        
        ax.set_xlabel('WIS Score')
        ax.set_title('All Seasons Combined: WIS Components Breakdown (Stacked)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "all_seasons_wis_stacked.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    # 4) Time series plot - Performance over forecast dates (all seasons, all dates)
    # Show complete time series for all seasons to properly visualize missing data
    timeseries_data = []
    all_forecast_dates = set()
    
    for season, season_data in all_seasons_results.items():
        season_dates = sorted(season_data["all_scores_rel"]["forecast_date"].unique())
        all_forecast_dates.update(season_dates)
        
        for date in season_dates:
            date_data = season_data["all_scores_rel"][
                (season_data["all_scores_rel"]["forecast_date"] == date) &
                (season_data["all_scores_rel"]["location"] == "US")  # US only for clarity
            ].copy()
            date_data["season"] = season
            timeseries_data.append(date_data)
    
    if timeseries_data:
        timeseries_combined = pd.concat(timeseries_data, ignore_index=True)
        
        # Get top models (by average performance)
        model_avg_performance = timeseries_combined.groupby("model")["value"].mean().sort_values()
        top_models = model_avg_performance.head(10).index.tolist()  # Best 10 models
        
        if top_models:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Create a complete timeline for all dates across all seasons
            all_dates_sorted = sorted(all_forecast_dates)
            date_to_index = {date: i for i, date in enumerate(all_dates_sorted)}
            
            # Create time series for each model with proper missing data handling
            for model in top_models:
                model_data = timeseries_combined[timeseries_combined["model"] == model].copy()
                if not model_data.empty:
                    # Create arrays for x and y values, allowing for gaps (missing data)
                    x_vals = []
                    y_vals = []
                    
                    # Group by season and forecast_date to get values
                    for _, row in model_data.iterrows():
                        date_idx = date_to_index[row["forecast_date"]]
                        x_vals.append(date_idx)
                        y_vals.append(row["value"])
                    
                    # Determine color and style
                    color = 'green' if model.startswith('i') and '::' in model else 'blue'
                    missing_count = model_total_missing.get(model, 0)
                    
                    # Create label with missing count
                    if missing_count > 0:
                        label = f"{model[:50]}... missing:{missing_count}" if len(model) > 50 else f"{model} missing:{missing_count}"
                        linestyle = '--'  # Dashed line for models with missing data
                        alpha = 0.7
                    else:
                        label = f"{model[:50]}..." if len(model) > 50 else model
                        linestyle = '-'
                        alpha = 1.0
                    
                    # Plot with gaps where data is missing (don't connect missing points)
                    ax.plot(x_vals, y_vals, 
                           marker='o', label=label, color=color, linestyle=linestyle, 
                           linewidth=2, alpha=alpha, markersize=4)
            
            # Customize plot
            # Create season boundaries and labels
            season_boundaries = []
            season_labels = []
            current_season = None
            season_start = 0
            
            for i, date in enumerate(all_dates_sorted):
                # Determine which season this date belongs to
                date_season = None
                for season in all_seasons_results.keys():
                    if date in all_seasons_results[season]["all_scores_rel"]["forecast_date"].values:
                        date_season = season
                        break
                
                if date_season != current_season:
                    if current_season is not None:
                        # Mark end of previous season
                        season_boundaries.append(i - 0.5)
                        # Add label at midpoint of season
                        mid_point = (season_start + i - 1) / 2
                        season_labels.append((mid_point, current_season))
                    current_season = date_season
                    season_start = i
            
            # Add final season label
            if current_season is not None:
                mid_point = (season_start + len(all_dates_sorted) - 1) / 2
                season_labels.append((mid_point, current_season))
            
            # Add vertical lines between seasons
            for boundary in season_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
            
            # Add season labels at top
            for pos, season in season_labels:
                ax.text(pos, ax.get_ylim()[1], season, ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            # Set x-axis ticks and labels (sample every ~10th date for readability)
            tick_indices = list(range(0, len(all_dates_sorted), max(1, len(all_dates_sorted)//20)))
            tick_labels = [str(all_dates_sorted[i]) for i in tick_indices]
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            ax.axhline(y=1, color='red', linestyle=':', alpha=0.7, label='Baseline (WIS=1)')
            ax.set_ylabel('Relative WIS (vs Flusight-Baseline)')
            ax.set_title('All Seasons: Relative Performance Over Time (US National)\nMissing data shown as gaps')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, "all_seasons_relative_time_series.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # Create absolute time series as well
            abs_timeseries_data = []
            for season, season_data in all_seasons_results.items():
                season_dates = sorted(season_data["all_scores"]["forecast_date"].unique())
                
                for date in season_dates:
                    date_data = season_data["all_scores"][
                        (season_data["all_scores"]["forecast_date"] == date) &
                        (season_data["all_scores"]["location"] == "US") &
                        (season_data["all_scores"]["wis_type"] == "wis_total")
                    ].copy()
                    date_data["season"] = season
                    abs_timeseries_data.append(date_data)
            
            if abs_timeseries_data:
                abs_timeseries_combined = pd.concat(abs_timeseries_data, ignore_index=True)
                abs_top_models = abs_timeseries_combined.groupby("model")["value"].mean().sort_values().head(10).index.tolist()
                
                if abs_top_models:
                    fig, ax = plt.subplots(figsize=(16, 10))
                    
                    # Create time series for each model with proper missing data handling
                    for model in abs_top_models:
                        model_data = abs_timeseries_combined[abs_timeseries_combined["model"] == model].copy()
                        if not model_data.empty:
                            # Create arrays for x and y values, allowing for gaps (missing data)
                            x_vals = []
                            y_vals = []
                            
                            # Group by season and forecast_date to get values
                            for _, row in model_data.iterrows():
                                date_idx = date_to_index[row["forecast_date"]]
                                x_vals.append(date_idx)
                                y_vals.append(row["value"])
                            
                            # Determine color and style
                            color = 'green' if model.startswith('i') and '::' in model else 'blue'
                            missing_count = model_total_missing.get(model, 0)
                            
                            # Create label with missing count
                            if missing_count > 0:
                                label = f"{model[:50]}... missing:{missing_count}" if len(model) > 50 else f"{model} missing:{missing_count}"
                                linestyle = '--'  # Dashed line for models with missing data
                                alpha = 0.7
                            else:
                                label = f"{model[:50]}..." if len(model) > 50 else model
                                linestyle = '-'
                                alpha = 1.0
                            
                            # Plot with gaps where data is missing (don't connect missing points)
                            ax.plot(x_vals, y_vals, 
                                   marker='o', label=label, color=color, linestyle=linestyle, 
                                   linewidth=2, alpha=alpha, markersize=4)
                    
                    # Add season boundaries and labels (same as relative plot)
                    for boundary in season_boundaries:
                        ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
                    
                    for pos, season in season_labels:
                        ax.text(pos, ax.get_ylim()[1], season, ha='center', va='bottom', 
                               fontweight='bold', fontsize=10)
                    
                    # Set x-axis ticks and labels
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                    
                    ax.set_ylabel('Absolute WIS')
                    ax.set_title('All Seasons: Absolute Performance Over Time (US National)\nMissing data shown as gaps')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    fig.savefig(os.path.join(save_dir, "all_seasons_absolute_time_series.png"), dpi=200, bbox_inches='tight')
                    plt.close(fig)
    
    print(f"Combined plots saved to: {save_dir}")


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
        
        # Get model missing counts - need to extract from the scoring process
        scores, failures = evaluate_models(season, dates)
        model_missing_counts = {}
        for model_key in scores.keys():
            if "!" in model_key:
                parts = model_key.split("!")
                clean_model_name = parts[0]
                missing_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            else:
                clean_model_name = model_key
                missing_count = 0
            model_missing_counts[clean_model_name] = missing_count
        
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
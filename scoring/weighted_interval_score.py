"""
Weighted Interval Score computation and hub-format scoring wrapper.

This module provides the core WIS implementation and hub-compatible scorer
moved from evaluate_deprecated.py.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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


def score_Nwk_forecasts_hub(gt: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WIS and components for specified horizons in hub-format forecasts.

    Expects forecasts with columns: target_end_date, location, output_type, output_type_id, value, target, horizon
    """
    f = forecasts.copy()
    # Standard filter: quantile forecasts only
    if "output_type" in f.columns:
        f = f[f["output_type"] == "quantile"]
    # Normalize types
    f["target_end_date"] = pd.to_datetime(f["target_end_date"]).dt.date
    f["output_type_id"] = pd.to_numeric(f["output_type_id"], errors="coerce")
    f["horizon"] = pd.to_numeric(f["horizon"], errors="coerce")
    f["location"] = f["location"].astype(str).str.strip()

    locations = sorted(f["location"].unique())
    target_dates = sorted(f["target_end_date"].unique())

    gt2 = gt[(gt["location"].isin(locations)) & (gt["date"].isin(target_dates))].copy()
    gt_piv = gt2.pivot(index="date", columns="location", values="value").sort_index()

    qvals = sorted(f["output_type_id"].unique())
    lower = [q for q in qvals if q <= 0.5]
    alphas = np.array(lower) * 2

    gt_dates = set(gt_piv.index)
    available_dates = [d for d in target_dates if d in gt_dates]
    if not available_dates:
        raise RuntimeError("No valid target dates aligned with ground truth.")

    all_targets = []
    for target_date in available_dates:
        sub = f[f["target_end_date"] == target_date]
        q_dict: Dict[float, np.ndarray] = {}
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
        masks = [~pd.isna(obs)]
        for q in q_levels:
            masks.append(~pd.isna(q_dict[float(q)]))
        valid_mask = np.logical_and.reduce(masks)
        if not np.any(valid_mask):
            continue
        obs_v = obs[valid_mask]
        q_dict_v = {float(q): q_dict[float(q)][valid_mask] for q in q_levels}
        (wis_total, wis_sharpness, wis_calibration, underprediction, overprediction) = weighted_interval_score_fast(
            observations=obs_v,
            alphas=alphas,
            q_dict=q_dict_v,
            weights=alphas / 2,
        )

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
    return pd.concat(all_targets).reset_index(names="scoring_metric").set_index(["target", "target_end_date"])
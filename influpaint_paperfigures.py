"""
Influpaint paper figures script (notebook-style).

Generates three groups of figures:
1) Unconditional generation: US grid and trajectories+mean heatmap for i804
2) Forecasts from CSV (4-week hubverse): overlay quantile fans on ground truth
3) Forecasts from NPY (full horizon): two-panel figure for two seasons
4) Mask experiments: ground truth in black and colored trajectories
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from influpaint.utils import SeasonAxis
import influpaint.utils.plotting as idplots
from influpaint.utils.helpers import flusight_quantile_pairs, flusight_quantiles
from influpaint.utils import ground_truth


# ==== Constants / Paths (edit as needed) ====
IMAGE_SIZE = 64
CHANNELS = 1

PLOT_MEDIAN = True

BEST_MODEL_ID = "i868"
BEST_CONFIG = "celebahq_noTTJ5"

def find_uncond_samples_path(model_id: str, base_dir: str = "from_longleaf/regen/samples_regen/") -> str:
    import glob
    pattern = os.path.join(base_dir, f"inverse_transformed_samples_{model_id}*.npy")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No inverse transformed samples found for model {model_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple samples found for model {model_id}: {matches}")
    return matches[0]

UNCOND_SAMPLES_PATH = find_uncond_samples_path(BEST_MODEL_ID)

INPAINTING_BASE = (
    "from_longleaf/influpaint_res/07b44fa_paper-2025-07-22_inpainting_2025-07-27"
)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
_MODEL_NUM = BEST_MODEL_ID.lstrip('i') if isinstance(BEST_MODEL_ID, str) else str(BEST_MODEL_ID)

# Fixed x-limits per season for publication-friendly alignment
SEASON_XLIMS = {
    '2023-2024': (dt.datetime(2023, 10, 7), dt.datetime(2024, 6, 1)),
    '2024-2025': (dt.datetime(2024, 11, 16), dt.datetime(2025, 5, 31)),
}

# Toggle: also show pre-forecast ("past") segments of NPY forecasts
SHOW_NPY_PAST = True


def forecast_week_saturdays(season: str, season_axis: SeasonAxis, max_weeks: int) -> pd.DatetimeIndex:
    """Return Saturday dates for forecast weeks for a given season string 'YYYY-YYYY'.

    Uses SeasonAxis.get_season_calendar under the hood.
    """
    season_year = int(str(season).split('-')[0])
    cal = season_axis.get_season_calendar(season_year)
    saturdays = pd.to_datetime(cal['saturday'])
    eff_len = min(max_weeks, len(saturdays))
    return saturdays[:eff_len]


def _format_date_axis(ax):
    """Apply YYYY-MM date format and tilt labels to avoid overlap."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')


# ==== Helpers ====
def load_unconditional_samples(path: str) -> np.ndarray:
    x = np.load(path)
    if x.ndim == 3:
        x = x[:, None, :, :]
    return x


def list_influpaint_csvs(base_dir: str, model_id: str, config: str):
    out = []
    for root, _, files in os.walk(base_dir):
        if model_id in root and f"conf_{config}" in root:
            for f in files:
                if f.endswith(".csv") and not f.endswith("-copaint.csv"):
                    out.append(os.path.join(root, f))
    return sorted(out)


def parse_date_from_folder(folder_name: str):
    try:
        d = folder_name.split("::")[-1]
        return pd.to_datetime(d).date()
    except Exception:
        return None


def list_inpainting_dirs(base_dir: str, model_id: str, config: str):
    out = []
    for d in os.listdir(base_dir):
        p = os.path.join(base_dir, d)
        if not os.path.isdir(p):
            continue
        if (d.startswith(model_id) and f"conf_{config}" in d):
            if os.path.exists(os.path.join(p, "fluforecasts_ti.npy")):
                out.append(p)
    return sorted(out)


def _state_to_code(state: str, season_axis: SeasonAxis) -> str:
    """Map 'US', FIPS code like '37', or abbrev like 'NC' to location_code string."""
    if state.upper() == 'US':
        return 'US'
    if state in set(season_axis.locations_df["location_code"].astype(str)):
        return str(state)
    m = season_axis.locations_df[season_axis.locations_df['abbreviation'].str.upper() == state.upper()]
    if not m.empty:
        return str(m.iloc[0]['location_code'])
    raise ValueError(f"Unknown state '{state}'")


# ==== 1) Unconditional Figures ====
season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
uncond = load_unconditional_samples(UNCOND_SAMPLES_PATH)

# US grid of several samples (like in batch/training.py)
fig, _ = idplots.plot_unconditional_us_map(
    inv_samples=uncond,
    season_axis=season_setup,
    sample_idx=list(np.arange(2, min(500, uncond.shape[0]), step=15)),
    multi_line=True,
    sharey=False,
    past_ground_truth=True,
)
plt.savefig(os.path.join(FIG_DIR, "unconditional_us_grid.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

# Trajectories + mean heatmap
fig, _ = idplots.fig_unconditional_trajectories_and_mean_heatmap(
    inv_samples=uncond,
    season_axis=season_setup,
    n_samples=12,
    save_path=os.path.join(FIG_DIR, "unconditional_trajs_and_mean_heatmap.png"),
)
plt.close(fig)


def plot_unconditional_states_quantiles_and_trajs(inv_samples: np.ndarray,
                                                  season_axis: SeasonAxis,
                                                  states: list[str],
                                                  n_sample_trajs: int = 10,
                                                  plot_median: bool = True,
                                                  save_path: str | None = None):
    """Plot unconditional sample trajectories and quantile fans for given states.

    - inv_samples: (N, 1, weeks, places) or (N, weeks, places)
    - states: list of state codes/abbrevs (expected ~5)
    - n_sample_trajs: number of light sample lines to overlay per state
    - plot_median: toggle median line overlay
    """
    # Normalize shape to (N, 1, W, P)
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    real_weeks = min(53, w)
    weeks = np.arange(1, real_weeks + 1)

    n_states = len(states)
    ncols = n_states
    nrows = 1
    if n_states > 5:
        nrows = 2
        ncols = int(np.ceil(n_states / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), dpi=200, sharey=False)
    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, st in enumerate(states):
        ax = axes_list[i]
        loc_code = _state_to_code(st, season_axis)
        place_idx = season_axis.locations.index(loc_code)
        ts = arr[:, 0, :real_weeks, place_idx]  # (N, W)

        # Color
        color = sns.color_palette('Set2', n_colors=n_states)[i % n_states]

        # Light sampled trajectories
        if n_sample_trajs and n_sample_trajs > 0:
            ns = min(n_sample_trajs, ts.shape[0])
            sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
            for si in sample_idxs:
                ax.plot(weeks, ts[si], color=color, alpha=0.25, lw=0.8, zorder=1)

        # Quantile bands
        for lo, hi in flusight_quantile_pairs:
            lo_curve = np.quantile(ts, lo, axis=0)
            hi_curve = np.quantile(ts, hi, axis=0)
            ax.fill_between(weeks, lo_curve, hi_curve, color=color, alpha=0.08, lw=0)

        # Median
        if plot_median:
            med = np.quantile(ts, 0.5, axis=0)
            ax.plot(weeks, med, color=color, lw=1.8, zorder=2)

        # Styling
        ax.text(0.02, 0.98, st.upper(), transform=ax.transAxes, va='top', ha='left',
                fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_xlim(1, real_weeks)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Epiweek')
        if i % ncols == 0:
            ax.set_ylabel('Incidence')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)

    # Hide any unused axes if states < grid size
    for j in range(len(axes_list)):
        if j >= n_states:
            axes_list[j].set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# New unconditional figure: 5 states quantiles + trajectories
fig = plot_unconditional_states_quantiles_and_trajs(
    inv_samples=uncond,
    season_axis=season_setup,
    states=['NC', 'CA', 'NY', 'TX', 'FL'],
    n_sample_trajs=10,
    plot_median=PLOT_MEDIAN,
    save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_states_quantiles_trajs.png"),
)
plt.close(fig)


def fig_unconditional_3d_heat_ridges(inv_samples: np.ndarray,
                                     season_axis: SeasonAxis,
                                     states: list[str] | None = None,
                                     stat: str = 'median',
                                     cmap: str = 'Reds',
                                     elev: float = 35,
                                     azim: float = -60,
                                     heatmap_mode: str = 'mean',
                                     sample_idx: int = 0,
                                     surface_alpha: float = 0.8,
                                     surface_zoffset_ratio: float = 0.0,
                                     ridge_offset_ratio: float = 0.005,
                                     location_stride: int = 1,
                                     fill_ridges: bool = True,
                                     fill_alpha: float = 0.35,
                                     save_path: str | None = None):
    """3D figure with bottom heatmap (mean across samples) and 3D ridgelines.

    - inv_samples: (N, 1, weeks, places) or (N, weeks, places)
    - states: optional list of state abbrevs/codes to overlay as ridges; if None, picks 8 evenly spaced locations
    - stat: 'median' or 'mean' used for ridge z-values across samples
    - cmap: colormap for bottom heatmap
    """
    # Normalize samples shape
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    P = len(season_axis.locations)
    real_weeks = min(53, w)

    # Compute heatmap (mean across samples or a single sample)
    if heatmap_mode == 'sample':
        sample_idx = int(np.clip(sample_idx, 0, n-1))
        heat = arr[sample_idx, 0, :real_weeks, :P]
    else:
        heat = arr[:, 0, :real_weeks, :P].mean(axis=0)  # (W, P)

    # Build grid for bottom surface
    x_vals = np.arange(1, real_weeks + 1)
    y_vals = np.arange(P)
    X, Y = np.meshgrid(x_vals, y_vals)  # shapes (P, W)
    # Optional tiny z-offset for surface (defaults to 0 to avoid visible shift)
    zmax_global = float(np.nanmax(heat)) if np.isfinite(heat).all() else 1.0
    zmax_global = max(1.0, zmax_global)
    Z0 = np.zeros_like(X, dtype=float) - (surface_zoffset_ratio * zmax_global if surface_zoffset_ratio else 0.0)
    Cdata = heat.T  # (P, W)

    # Normalize colors
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin=np.nanmin(Cdata), vmax=np.nanmax(Cdata) if np.nanmax(Cdata) > 0 else 1.0)
    facecolors = cmap_obj(norm(Cdata))

    # Figure and 3D axis
    fig = plt.figure(figsize=(12, 7), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Plot bottom colored surface (flat at z=0)
    surf = ax.plot_surface(X, Y, Z0, rstride=1, cstride=1,
                           facecolors=facecolors[:-1, :-1], shade=False,
                           linewidth=0, antialiased=False, alpha=surface_alpha)
    surf.set_zsort('max')

    # Choose locations for ridges
    if states:
        place_idxs = [season_axis.locations.index(_state_to_code(s, season_axis)) for s in states]
        labels = [s.upper() for s in states]
    else:
        stride = max(1, int(location_stride))
        place_idxs = list(range(0, P, stride))
        # readable labels using abbreviations if available
        locdf = season_axis.locations_df
        if 'abbreviation' in locdf.columns:
            abbr_map = locdf.set_index('location_code')['abbreviation']
            labels = [abbr_map.get(str(season_axis.locations[i]), str(season_axis.locations[i])) for i in place_idxs]
        else:
            labels = [str(season_axis.locations[i]) for i in place_idxs]

    # Palette for ridges
    ridge_colors = sns.color_palette('Set2', n_colors=len(place_idxs))

    # Plot ridges: x=weeks, y=place_idx, z=statistic over samples
    ridge_offset = ridge_offset_ratio * zmax_global
    for j, (pi, lab) in enumerate(zip(place_idxs, labels)):
        ts = arr[:, 0, :real_weeks, pi]  # (N, W)
        if stat == 'mean':
            z = np.nanmean(ts, axis=0)
        else:
            z = np.nanmedian(ts, axis=0)
        y_curve = np.full_like(x_vals, fill_value=pi)
        # Optional ribbon fill under the curve down to z=0
        if fill_ridges:
            Xf = np.vstack([x_vals, x_vals])
            Yf = np.vstack([y_curve, y_curve])
            Zf = np.vstack([z + ridge_offset, np.zeros_like(z)])
            ax.plot_surface(Xf, Yf, Zf,
                            color=ridge_colors[j], alpha=fill_alpha,
                            linewidth=0, antialiased=False, shade=False)
        ax.plot(x_vals, y_curve, z + ridge_offset, color=ridge_colors[j], lw=2.0, zorder=100-place_idxs[j])
        # Label near end of ridge
        ax.text(x_vals[-1]+0.5, pi, (z[-1] + ridge_offset), lab, color=ridge_colors[j], fontsize=9, ha='left', va='center')

    # Aesthetics
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('Epiweek')
    ax.set_ylabel('Location index')
    ax.set_zlabel('Incidence')
    ax.set_xlim(1, real_weeks)
    ax.set_ylim(0, P-1)
    ax.set_zlim(bottom=0)
    # no title per request

    # Light grid styling
    ax.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


# 3D ridge + heatmap illustration
fig3d, _ = fig_unconditional_3d_heat_ridges(
    inv_samples=uncond,
    season_axis=season_setup,
    states=None,
    stat='median',
    location_stride=5,
    elev=20,
    azim=-110,
    save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_3d_heat_ridges.png"),
)
plt.close(fig3d)

def fig_unconditional_3d_heat_ridges_plotly(inv_samples: np.ndarray,
                                            season_axis: SeasonAxis,
                                            states: list[str] | None = None,
                                            stat: str = 'median',
                                            heatmap_mode: str = 'mean',
                                            sample_idx: int = 0,
                                            surface_opacity: float = 0.6,
                                            ridge_lift: float = 0.0,
                                            location_stride: int = 1,
                                            camera_eye: tuple[float, float, float] | None = None,
                                            save_path_html: str | None = None):
    """Interactive Plotly version of 3D heatmap + ridgelines.

    Saves an interactive HTML if save_path_html is provided.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as e:
        raise RuntimeError("Plotly is required for this function. Please install plotly.") from e

    # Normalize shape
    if inv_samples.ndim == 4:
        arr = inv_samples
    elif inv_samples.ndim == 3:
        arr = inv_samples[:, None, :, :]
    else:
        raise ValueError("inv_samples must be (sample, feature, week, place) or (sample, week, place)")

    n, c, w, p = arr.shape
    P = len(season_axis.locations)
    real_weeks = min(53, w)

    # Heatmap data
    if heatmap_mode == 'sample':
        sample_idx = int(np.clip(sample_idx, 0, n-1))
        heat = arr[sample_idx, 0, :real_weeks, :P]
    else:
        heat = arr[:, 0, :real_weeks, :P].mean(axis=0)

    weeks = np.arange(1, real_weeks + 1)
    y_idx = np.arange(P)
    Z = heat.T  # (P, W)

    # Build figure with surface
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=weeks, y=y_idx,
                             colorscale='Reds', showscale=True,
                             opacity=surface_opacity))

    # Choose locations for ridges
    if states:
        place_idxs = [season_axis.locations.index(_state_to_code(s, season_axis)) for s in states]
        labels = [s.upper() for s in states]
    else:
        stride = max(1, int(location_stride))
        place_idxs = list(range(0, P, stride))
        locdf = season_axis.locations_df
        if 'abbreviation' in locdf.columns:
            abbr_map = locdf.set_index('location_code')['abbreviation']
            labels = [abbr_map.get(str(season_axis.locations[i]), str(season_axis.locations[i])) for i in place_idxs]
        else:
            labels = [str(season_axis.locations[i]) for i in place_idxs]

    # Add ridge lines
    for pi, lab in zip(place_idxs, labels):
        ts = arr[:, 0, :real_weeks, pi]
        if stat == 'mean':
            z = np.nanmean(ts, axis=0)
        else:
            z = np.nanmedian(ts, axis=0)
        fig.add_trace(go.Scatter3d(x=weeks, y=np.full_like(weeks, pi), z=z + ridge_lift,
                                   mode='lines', name=lab,
                                   line=dict(width=4)))

    # Layout and camera
    # Camera eye: lower and more to the left by default
    if camera_eye is None:
        camera_eye = (1.0, -1.6, 0.6)

    fig.update_layout(
        scene=dict(
            xaxis_title='Epiweek',
            yaxis_title='Location index',
            zaxis_title='Incidence',
            camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=None
    )

    if save_path_html:
        # Embed plotly.js for offline viewing
        pio.write_html(fig, file=save_path_html, include_plotlyjs=True, full_html=True)
        if os.path.exists(save_path_html):
            print(f"Saved Plotly HTML to {save_path_html}")
        else:
            print(f"Failed to save Plotly HTML to {save_path_html}")
    return fig


# Plotly interactive 3D version (HTML)
try:
    html_path = os.path.join(FIG_DIR, f"{_MODEL_NUM}_uncond_3d_heat_ridges_plotly.html")
    _ = fig_unconditional_3d_heat_ridges_plotly(
        inv_samples=uncond,
        season_axis=season_setup,
        states=None,
        stat='median',
        surface_opacity=0.6,
        ridge_lift=0.0,
        location_stride=3,
        camera_eye=(1.0, -1.6, 0.6),
        save_path_html=html_path,
    )
except RuntimeError as _e:
    print("Plotly not available; skipping interactive 3D figure.")

# ==== 2) Forecasts from CSVs (4-week hubverse) ====
def load_truth_for_season(season: str) -> pd.DataFrame:
    from prepare_dataset_for_scoringutils import ScoringutilsFullEvaluator
    ev = ScoringutilsFullEvaluator()
    gt = ev.load_ground_truth(season)
    gt["date"] = pd.to_datetime(gt["date"])  # ensure datetime
    return gt



def plot_csv_quantile_fans_for_season(season: str, base_dir: str, model_id: str, config: str,
                                      pick_every: int = 2, state='US',
                                      start_date: str = '2023-10-07',
                                      save_path: str | None = None,
                                      plot_median: bool = True):
    states = state if isinstance(state, (list, tuple)) else [state]
    n = len(states)
    # Layout: for readability, use 2 rows when many states
    if n >= 4:
        nrows = 2
        ncols = int(np.ceil(n / 2))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), dpi=200, sharey=False)
        axes_list = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=(4*n, 3.5), dpi=200, sharey=False)
        axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    csvs = list_influpaint_csvs(base_dir, model_id, config)
    if not csvs:
        print("No CSV forecasts found.")
        return None
    # Load all forecast CSVs once
    df_list = []
    for p in csvs:
        try:
            dfi = pd.read_csv(p, dtype={"location": str})
            if "reference_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["reference_date"]).dt.date
            elif "forecast_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["forecast_date"]).dt.date
            else:
                continue
            dfi["target_end_date"] = pd.to_datetime(dfi["target_end_date"]).dt.date
            dfi["q"] = pd.to_numeric(dfi.get("output_type_id", dfi.get("quantile")), errors="coerce")
            dfi["target"] = dfi.get("target", "wk inc flu hosp")
            df_list.append(dfi)
        except Exception:
            continue
    if not df_list:
        print("No valid CSV content parsed.")
        return None
    df_all = pd.concat(df_list, ignore_index=True)

    # Use fixed bounds if provided for season
    left_bound = SEASON_XLIMS.get(season, (pd.to_datetime(start_date), None))[0]
    # Default right bound is end-of-season if available, else 365 days after start
    default_right = pd.to_datetime(start_date) + pd.Timedelta(days=365)
    right_bound = SEASON_XLIMS.get(season, (None, default_right))[1] or default_right
    for i_ax, (ax, st) in enumerate(zip(axes_list, states)):
        loc_code = _state_to_code(st, season_setup)
        # Ground truth for state
        gt = load_truth_for_season(season)
        gt = gt[gt["location"].astype(str) == loc_code].sort_values('date')
        gt = gt[(gt['date'] >= left_bound) & (gt['date'] <= right_bound)]
        ax.plot(gt['date'], gt['value'], color='black', lw=2)

        # State forecasts
        df = df_all[(df_all["location"].astype(str) == loc_code) & (df_all["target"] == "wk inc flu hosp") & (df_all["output_type"] == "quantile")]
        refs = sorted(df["ref"].unique())
        refs = refs[::max(1, pick_every)]
        palette = sns.color_palette("Set2", n_colors=len(refs))
        for j, r in enumerate(refs):
            sub = df[df["ref"] == r]
            if sub.empty:
                continue
            for lo, hi in flusight_quantile_pairs:
                low = sub[np.isclose(sub["q"], lo)].sort_values("target_end_date")
                up = sub[np.isclose(sub["q"], hi)].sort_values("target_end_date")
                if len(low) and len(up):
                    x = pd.to_datetime(low["target_end_date"]).values
                    mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                    if np.any(mask):
                        ax.fill_between(x[mask], low["value"].values[mask], up["value"].values[mask],
                                        color=palette[j], alpha=0.08, lw=0)
            med = sub[np.isclose(sub["q"], 0.5)].sort_values("target_end_date")
            if plot_median and len(med):
                x = pd.to_datetime(med["target_end_date"]).values
                mask = (x >= np.datetime64(left_bound)) & (x <= np.datetime64(right_bound))
                if np.any(mask):
                    ax.plot(x[mask], med["value"].values[mask], color=palette[j], lw=2)
                rdt = pd.to_datetime(r)
                if left_bound <= rdt <= right_bound:
                    ax.axvline(rdt, color=palette[j], ls='--', lw=1)
                    # Add date label near the top like in multi-season plot
                    ymax = ax.get_ylim()[1]
                    ax.text(rdt, ymax*0.95, str(rdt.date()), color=palette[j], rotation=90,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        _add_corner = ax.text(0.02, 0.98, st.upper(), transform=ax.transAxes, va='top', ha='left',
                               fontsize=11, fontweight='bold',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_ylim(bottom=0)
        if i_ax == 0:
            ax.set_ylabel('Incident flu hospitalizations')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)
        ax.set_xlim(left_bound, right_bound)
        _format_date_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_csv_quantile_fans_multiseasons(seasons: list, base_dir: str, model_id: str, config: str,
                                        states: list, pick_every: int = 2,
                                        save_path: str | None = None,
                                        plot_median: bool = True):
    """Plot CSV forecast fans over full multi-season ground truth for multiple states.

    - seasons: list like ['2023-2024','2024-2025']
    - states: list like ['US','NC','CA','NY','TX']
    - x-limits: fixed from 2023-10-07 to 2025-05-31 for readability
    """
    # Build GT across requested seasons
    gt_all_list = []
    for s in seasons:
        g = load_truth_for_season(s)
        g['date'] = pd.to_datetime(g['date'])
        gt_all_list.append(g)
    gt_all = pd.concat(gt_all_list, ignore_index=True)

    # Load all forecast CSVs once
    csvs = list_influpaint_csvs(base_dir, model_id, config)
    if not csvs:
        print("No CSV forecasts found.")
        return None
    df_list = []
    for p in csvs:
        try:
            dfi = pd.read_csv(p, dtype={"location": str})
            if "reference_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["reference_date"]).dt.date
            elif "forecast_date" in dfi.columns:
                dfi["ref"] = pd.to_datetime(dfi["forecast_date"]).dt.date
            else:
                continue
            dfi["target_end_date"] = pd.to_datetime(dfi["target_end_date"]).dt.date
            dfi["q"] = pd.to_numeric(dfi.get("output_type_id", dfi.get("quantile")), errors="coerce")
            dfi["target"] = dfi.get("target", "wk inc flu hosp")
            df_list.append(dfi)
        except Exception:
            continue
    if not df_list:
        print("No valid CSV content parsed.")
        return None
    df_all = pd.concat(df_list, ignore_index=True)

    # Layout: two rows for readability when many states
    n = len(states)
    if n >= 4:
        nrows = 2
        ncols = int(np.ceil(n / 2))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), dpi=200, sharey=False)
        axes_list = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), dpi=200, sharey=False)
        axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    # Global x-lims spanning both seasons
    x_left = dt.datetime(2023, 10, 7)
    x_right = dt.datetime(2025, 5, 31)

    for ax, st in zip(axes_list, states):
        loc_code = _state_to_code(st, season_setup)
        # GT
        if loc_code == 'US':
            gt_us = gt_all[gt_all['location'].astype(str) == 'US'].sort_values('date')
            ax.plot(gt_us['date'], gt_us['value'], color='black', lw=2)
        else:
            gt_st = gt_all[gt_all['location'].astype(str) == loc_code].sort_values('date')
            ax.plot(gt_st['date'], gt_st['value'], color='black', lw=2)

        # Forecasts for this state across both seasons
        df = df_all[(df_all["location"].astype(str) == loc_code) & (df_all["target"] == "wk inc flu hosp") & (df_all["output_type"] == "quantile")]
        refs = sorted(df["ref"].unique())
        refs = refs[::max(1, pick_every)]
        palette = sns.color_palette("Set2", n_colors=len(refs))
        for i, r in enumerate(refs):
            sub = df[df["ref"] == r]
            if sub.empty:
                continue
            # quantile bands
            for lo, hi in flusight_quantile_pairs:
                low = sub[np.isclose(sub["q"], lo)].sort_values("target_end_date")
                up = sub[np.isclose(sub["q"], hi)].sort_values("target_end_date")
                if len(low) and len(up):
                    x = pd.to_datetime(low["target_end_date"]).values
                    ax.fill_between(x, low["value"].values, up["value"].values, color=palette[i], alpha=0.08, lw=0)
            # median
            med = sub[np.isclose(sub["q"], 0.5)].sort_values("target_end_date")
            if plot_median and len(med):
                x = pd.to_datetime(med["target_end_date"]).values
                ax.plot(x, med["value"].values, color=palette[i], lw=2)
                rdt = pd.to_datetime(r)
                ax.axvline(rdt, color=palette[i], ls='--', lw=1)
                ax.text(rdt, ax.get_ylim()[1]*0.95, str(r), color=palette[i], rotation=90,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Styling
        ax.text(0.02, 0.98, st.upper(), transform=ax.transAxes, va='top', ha='left', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_ylim(bottom=0)
        ax.set_xlim(x_left, x_right)
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)
        _format_date_axis(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


fig = plot_csv_quantile_fans_multiseasons(
    seasons=["2023-2024", "2024-2025"],
    base_dir=INPAINTING_BASE,
    model_id=BEST_MODEL_ID,
    config=BEST_CONFIG,
    states=['US','NC','CA','NY','TX','FL'],
    pick_every=2,
    save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_csv_fans_states_2023_2025.png"),
    plot_median=PLOT_MEDIAN,
)
if fig is not None:
    plt.close(fig)

# Also create per-season CSV fan plots for the same states
_csv_states = ['US','NC','CA','NY','TX','FL']
for _season in ["2023-2024", "2024-2025"]:
    _fig = plot_csv_quantile_fans_for_season(
        season=_season,
        base_dir=INPAINTING_BASE,
        model_id=BEST_MODEL_ID,
        config=BEST_CONFIG,
        pick_every=2,
        state=_csv_states,
        save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_csv_fans_states_{_season.replace('-', '_')}.png"),
        plot_median=PLOT_MEDIAN,
    )
    if _fig is not None:
        plt.close(_fig)


# ==== 3) NPY full-horizon forecasts: two-panel (two seasons) ====
def plot_npy_multi_date_two_seasons(base_dir: str, model_id: str, config: str,
                                    seasons=("2023-2024", "2024-2025"),
                                    per_season_pick=4,
                                    state=('US',),
                                    start_date: str = '2023-10-07',
                                    save_path: str | None = None,
                                    n_sample_trajs: int = 10,
                                    plot_median: bool = True):
    states = state if isinstance(state, (list, tuple)) else [state]
    nrows, ncols = len(seasons), len(states)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), dpi=200, sharey=False)
    # Normalize axes to 2D array shape (nrows, ncols)
    if nrows == 1 and ncols == 1:
        axes2 = np.array([[axes]])
    elif nrows == 1:
        axes2 = np.array(axes).reshape(1, ncols)
    elif ncols == 1:
        axes2 = np.array(axes).reshape(nrows, 1)
    else:
        axes2 = np.array(axes)
    for iax, season in enumerate(seasons):
        dirs = list_inpainting_dirs(base_dir, model_id, config)
        dirs_with_dates = []
        for d in dirs:
            dd = parse_date_from_folder(os.path.basename(d))
            if dd is None:
                continue
            y = season.split('-')[0]
            if SeasonAxis.for_flusight(remove_us=True, remove_territories=True).get_fluseason_year(pd.to_datetime(dd)) == int(y):
                dirs_with_dates.append((dd, d))
        dirs_with_dates = sorted(dirs_with_dates)[:]
        if not dirs_with_dates:
            for icol in range(ncols):
                axes[iax][icol].text(0.5, 0.5, f"{season}: no forecasts", transform=axes[iax][icol].transAxes, ha='center', va='center')
                axes[iax][icol].set_axis_off()
            continue
        step = max(1, len(dirs_with_dates) // max(1, per_season_pick))
        picked = dirs_with_dates[::step][:per_season_pick]
        min_dref = min(d for d,_ in picked)
        # Fixed bounds per season if configured
        default_left = pd.to_datetime(start_date)
        left_bound = SEASON_XLIMS.get(season, (default_left, None))[0]
        end_year = int(season.split('-')[1])
        default_right = dt.datetime(end_year, 5, 31)
        right_bound = SEASON_XLIMS.get(season, (None, default_right))[1] or default_right
        for icol, st in enumerate(states):
            ax = axes2[iax, icol]
            loc_code = _state_to_code(st, season_setup)
            # GT
            gt_df = load_truth_for_season(season)
            if loc_code == 'US':
                # Use national GT directly (do not sum states)
                gt_us = gt_df[gt_df['location'].astype(str) == 'US'].sort_values('date')
                gt_us = gt_us[(gt_us['date'] >= left_bound) & (gt_us['date'] <= right_bound)]
                x_dates = pd.to_datetime(gt_us['date'].values)
                ax.plot(x_dates, gt_us['value'].values, color='k', lw=2)
            else:
                gt = gt_df[gt_df['location'].astype(str) == loc_code].sort_values('date')
                gt = gt[(gt['date'] >= left_bound) & (gt['date'] <= right_bound)]
                x_dates = gt['date'].values
                ax.plot(x_dates, gt['value'], color='k', lw=2)
            # Forecasts
            palette = sns.color_palette('Dark2', n_colors=len(picked))
            for i, (dref, dpath) in enumerate(picked):
                arr = np.load(os.path.join(dpath, 'fluforecasts_ti.npy'))
                if loc_code == 'US':
                    ts = arr[:, 0, :, :len(season_setup.locations)].sum(axis=-1)
                else:
                    place_idx = season_setup.locations.index(loc_code)
                    ts = arr[:, 0, :, place_idx]
                # Get forecast Saturdays via SeasonAxis mapping
                x_weeks = forecast_week_saturdays(season, season_setup, ts.shape[1])
                eff_len = min(len(x_weeks), ts.shape[1])
                dates_plot = pd.to_datetime(x_weeks[:eff_len])
                ts = ts[:, :eff_len]
                # Light sampled trajectories
                if n_sample_trajs and n_sample_trajs > 0:
                    ns = min(n_sample_trajs, ts.shape[0])
                    sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
                    # Future trajectories
                    mask_fut = dates_plot >= pd.to_datetime(dref)
                    if np.any(mask_fut):
                        for si in sample_idxs:
                            ax.plot(dates_plot[mask_fut], ts[si, mask_fut], color=palette[i], alpha=0.25, lw=0.7, zorder=1)
                    # Past trajectories (optional)
                    if SHOW_NPY_PAST:
                        mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                        if np.any(mask_past):
                            for si in sample_idxs:
                                ax.plot(dates_plot[mask_past], ts[si, mask_past], color=palette[i], alpha=0.15, lw=0.6, ls=':', zorder=1)
                for lo, hi in flusight_quantile_pairs:
                    # Future (from forecast start)
                    mask_fut = dates_plot >= pd.to_datetime(dref)
                    x_plot_fut = dates_plot[mask_fut]
                    if len(x_plot_fut) > 0:
                        ylo_f = np.quantile(ts, lo, axis=0)[mask_fut]
                        yhi_f = np.quantile(ts, hi, axis=0)[mask_fut]
                        ax.fill_between(x_plot_fut, ylo_f, yhi_f, color=palette[i], alpha=0.05, lw=0)
                    # Past (before forecast start), optional
                    if SHOW_NPY_PAST:
                        mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                        x_plot_past = dates_plot[mask_past]
                        if len(x_plot_past) > 0:
                            ylo_p = np.quantile(ts, lo, axis=0)[mask_past]
                            yhi_p = np.quantile(ts, hi, axis=0)[mask_past]
                            ax.fill_between(x_plot_past, ylo_p, yhi_p, color=palette[i], alpha=0.03, lw=0)
                if plot_median:
                    med_all = np.quantile(ts, 0.5, axis=0)
                    # Future median
                    mask_med_f = dates_plot >= pd.to_datetime(dref)
                    x_plot_med_f = dates_plot[mask_med_f]
                    if len(x_plot_med_f) > 0:
                        ax.plot(x_plot_med_f, med_all[mask_med_f], color=palette[i], lw=1.6, zorder=2)
                    # Past median (optional)
                    if SHOW_NPY_PAST:
                        mask_med_p = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                        x_plot_med_p = dates_plot[mask_med_p]
                        if len(x_plot_med_p) > 0:
                            ax.plot(x_plot_med_p, med_all[mask_med_p], color=palette[i], lw=1.0, alpha=0.7, ls=':', zorder=2)
                rdt = pd.to_datetime(dref)
                ax.axvline(rdt, color=palette[i], ls='--', lw=1)
                # annotate forecast date on the dashed line
                ax.text(rdt, ax.get_ylim()[1]*0.95, str(rdt.date()), color=palette[i], rotation=90,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            # Style
            ax.text(0.02, 0.98, st.upper(), transform=ax.transAxes, va='top', ha='left', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax, trim=True)
            if icol == 0:
                ax.set_ylabel('Incident flu hospitalizations')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Date')
            # Use fixed bounds per season
            ax.set_xlim(left_bound, right_bound)
            _format_date_axis(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


fig = plot_npy_multi_date_two_seasons(
    base_dir=INPAINTING_BASE,
    model_id=BEST_MODEL_ID,
    config=BEST_CONFIG,
    seasons=("2023-2024", "2024-2025"),
    per_season_pick=4,
    state=['US','NC','CA','NY','TX'],
    save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_npy_two_panel_states.png"),
    plot_median=PLOT_MEDIAN,
)
plt.close(fig)


def plot_npy_two_panel_national(base_dir: str, model_id: str, config: str,
                                seasons=("2023-2024", "2024-2025"),
                                per_season_pick=4,
                                start_date: str = '2023-10-07',
                                save_path: str | None = None,
                                n_sample_trajs: int = 10,
                                plot_median: bool = True):
    fig, axes = plt.subplots(1, len(seasons), figsize=(14, 5), dpi=200, sharey=False)
    if len(seasons) == 1:
        axes = [axes]
    for iax, season in enumerate(seasons):
        ax = axes[iax]
        dirs = list_inpainting_dirs(base_dir, model_id, config)
        dirs_with_dates = []
        for d in dirs:
            dd = parse_date_from_folder(os.path.basename(d))
            if dd is None:
                continue
            y = season.split('-')[0]
            if SeasonAxis.for_flusight(remove_us=True, remove_territories=True).get_fluseason_year(pd.to_datetime(dd)) == int(y):
                dirs_with_dates.append((dd, d))
        dirs_with_dates = sorted(dirs_with_dates)[:]
        if not dirs_with_dates:
            ax.set_title(f"{season}: no forecasts found")
            continue
        step = max(1, len(dirs_with_dates) // max(1, per_season_pick))
        picked = dirs_with_dates[::step][:per_season_pick]
        min_dref = min(d for d,_ in picked)
        # national GT: use US row directly (no state summing)
        gt_df = load_truth_for_season(season)
        gt_us = gt_df[gt_df['location'].astype(str) == 'US'].sort_values('date')
        left_bound = SEASON_XLIMS.get(season, (pd.to_datetime(start_date), None))[0]
        right_bound = SEASON_XLIMS.get(season, (None, dt.datetime(int(season.split('-')[1]),5,31)))[1]
        gt_us = gt_us[(gt_us['date'] >= left_bound) & (gt_us['date'] <= right_bound)]
        x_dates = pd.to_datetime(gt_us['date'].values)
        ax.plot(x_dates, gt_us['value'].values, color='k', lw=2)
        palette = sns.color_palette('Dark2', n_colors=len(picked))
        min_dref = min(d for d, _ in picked)
        for i, (dref, dpath) in enumerate(picked):
            arr = np.load(os.path.join(dpath, 'fluforecasts_ti.npy'))
            nat = arr.sum(axis=-1)[:, 0, :]  # (n_samples, weeks)
            x_weeks = forecast_week_saturdays(season, season_setup, nat.shape[1])
            eff_len = min(len(x_weeks), nat.shape[1])
            dates_plot = pd.to_datetime(x_weeks[:eff_len])
            # Light sampled trajectories
            if n_sample_trajs and n_sample_trajs > 0:
                ns = min(n_sample_trajs, nat.shape[0])
                sample_idxs = np.linspace(0, nat.shape[0]-1, num=ns, dtype=int)
                mask_fut = dates_plot >= pd.to_datetime(dref)
                if np.any(mask_fut):
                    for si in sample_idxs:
                        ax.plot(dates_plot[mask_fut], nat[si, :eff_len][mask_fut], color=palette[i], alpha=0.25, lw=0.7, zorder=1)
                if SHOW_NPY_PAST:
                    mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                    if np.any(mask_past):
                        for si in sample_idxs:
                            ax.plot(dates_plot[mask_past], nat[si, :eff_len][mask_past], color=palette[i], alpha=0.15, lw=0.6, ls=':', zorder=1)
            for lo, hi in flusight_quantile_pairs:
                # Future
                mask_fut = dates_plot >= pd.to_datetime(dref)
                if np.any(mask_fut):
                    lo_curve = np.quantile(nat[:, :eff_len], lo, axis=0)[mask_fut]
                    hi_curve = np.quantile(nat[:, :eff_len], hi, axis=0)[mask_fut]
                    ax.fill_between(dates_plot[mask_fut], lo_curve, hi_curve, color=palette[i], alpha=0.05, lw=0)
                # Past (optional)
                if SHOW_NPY_PAST:
                    mask_past = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                    if np.any(mask_past):
                        lo_curve_p = np.quantile(nat[:, :eff_len], lo, axis=0)[mask_past]
                        hi_curve_p = np.quantile(nat[:, :eff_len], hi, axis=0)[mask_past]
                        ax.fill_between(dates_plot[mask_past], lo_curve_p, hi_curve_p, color=palette[i], alpha=0.03, lw=0)
            if plot_median:
                # Medians
                mask_med_f = dates_plot >= pd.to_datetime(dref)
                if np.any(mask_med_f):
                    med = np.quantile(nat[:, :eff_len], 0.5, axis=0)[mask_med_f]
                    ax.plot(dates_plot[mask_med_f], med, color=palette[i], lw=1.8, zorder=2)
                if SHOW_NPY_PAST:
                    mask_med_p = (dates_plot >= left_bound) & (dates_plot < pd.to_datetime(dref))
                    if np.any(mask_med_p):
                        med_p = np.quantile(nat[:, :eff_len], 0.5, axis=0)[mask_med_p]
                        ax.plot(dates_plot[mask_med_p], med_p, color=palette[i], lw=1.0, alpha=0.7, ls=':', zorder=2)
            rdt = pd.to_datetime(dref)
            ax.axvline(rdt, color=palette[i], ls='--', lw=1)
            ax.text(rdt, ax.get_ylim()[1]*0.95, str(rdt.date()), color=palette[i], rotation=90,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, trim=True)
        if iax == 0:
            ax.set_ylabel('Incidence')
        ax.set_xlabel('Date')
        ax.set_xlim(left_bound, right_bound)
        _format_date_axis(ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


fig = plot_npy_two_panel_national(
    base_dir=INPAINTING_BASE,
    model_id=BEST_MODEL_ID,
    config=BEST_CONFIG,
    seasons=("2023-2024", "2024-2025"),
    per_season_pick=4,
    save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_npy_two_panel_US.png"),
    plot_median=PLOT_MEDIAN,
)
plt.close(fig)
fig = plot_npy_multi_date_two_seasons(
    base_dir=INPAINTING_BASE,
    model_id=BEST_MODEL_ID,
    config=BEST_CONFIG,
    seasons=("2023-2024", "2024-2025"),
    per_season_pick=4,
    state='CA',
    save_path=os.path.join(FIG_DIR, f"{_MODEL_NUM}_forecast_npy_two_panel_state_CA.png"),
    plot_median=PLOT_MEDIAN,
)
plt.close(fig)


# ==== 4) Mask experiments: one figure per mask ====
def _recreate_mask(gt: ground_truth.GroundTruth, mask_name: str):
    mask = np.ones((CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    ss = gt.season_setup
    if mask_name == 'missing_half_subpop':
        half = len(ss.locations)//2
        mask[:, :, :half] = 0
    elif mask_name == 'missing_midseason':
        start = pd.to_datetime(f"{gt.season_first_year}-12-07")
        end = pd.to_datetime(f"{int(gt.season_first_year)+1}-01-07")
        w0 = ss.get_season_week(start)
        w1 = ss.get_season_week(end)
        mask[:, w0-1:w1, :] = 0
    elif mask_name == 'missing_midseason_peak':
        start = pd.to_datetime(f"{int(gt.season_first_year)+1}-02-01")
        end = pd.to_datetime(f"{int(gt.season_first_year)+1}-02-15")
        w0 = ss.get_season_week(start)
        w1 = ss.get_season_week(end)
        mask[:, w0-1:w1, :] = 0
    elif mask_name == 'missing_nc':
        code = '37'
        idx = ss.locations.index(code)
        mask[:, :, idx] = 0
    return mask


def plot_mask_experiments(mask_dir: str, forecast_date: str,
                          states=('NC', 'CA'),
                          n_sample_trajs: int = 10,
                          plot_median: bool = True):
    masks = [d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]
    if not masks:
        print("No mask experiment subfolders found.")
        return []

    outs = []
    for name in sorted(masks):
        # Determine season from folder name if present
        season_detect = None
        try:
            import re
            m = re.search(r"_season(\d{4})", name)
            if m:
                season_detect = m.group(1)
        except Exception:
            season_detect = None
        season_use = season_detect

        
        # Build GT for this season
        gt = ground_truth.GroundTruth(
            season_first_year=str(season_use),
            data_date=dt.datetime.today(),
            mask_date=pd.to_datetime(forecast_date),
            channels=CHANNELS,
            image_size=IMAGE_SIZE,
            nogit=True,
        )
        dates = pd.to_datetime(gt.gt_xarr['date'].values)

        # Load data
        subdir = os.path.join(mask_dir, name)
        f_path = os.path.join(subdir, 'fluforecasts_ti.npy')
        m_path = os.path.join(subdir, 'mask.npy')
        if not (os.path.exists(f_path) and os.path.exists(m_path)):
            continue
        arr = np.load(f_path)
        mk = np.load(m_path)

        # Choose locations: if exactly 5 masked locations -> plot those; else pick up to 5 masked
        p_len = len(gt.season_setup.locations)
        masked_any = (mk[0, :arr.shape[2], :p_len] == 0).any(axis=0)
        masked_idx = np.where(masked_any)[0].tolist()
        if len(masked_idx) == 5:
            plot_indices = masked_idx
        elif len(masked_idx) > 0:
            plot_indices = masked_idx[:5]
        else:
            # fallback to provided states
            plot_indices = []
            for st in (states if isinstance(states, (list, tuple)) else [states]):
                code = _state_to_code(st, gt.season_setup)
                plot_indices.append(gt.season_setup.locations.index(code))
            plot_indices = plot_indices[:5]

        # Labels
        locdf = gt.season_setup.locations_df
        abbr_map = None
        if 'abbreviation' in locdf.columns:
            abbr_map = locdf.set_index('location_code')['abbreviation']
        labels = [abbr_map.get(str(gt.season_setup.locations[i]), str(gt.season_setup.locations[i])) if abbr_map is not None else str(gt.season_setup.locations[i]) for i in plot_indices]

        ncols = 1 + len(plot_indices)
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4.5), dpi=200)
        if ncols == 2:
            axes = [axes[0], axes[1]]
        # Mask overlay
        base_crop = gt.gt_xarr.data[0][:52, :52]
        mask_crop = mk[0][:52, :52]
        axes[0].imshow(base_crop.T, cmap='Greys', aspect='equal')
        axes[0].imshow(mask_crop.T, alpha=.3, cmap='rainbow', aspect='equal')
        axes[0].set_aspect('equal')
        axes[0].set_axis_off()

        palette = sns.color_palette('Set1', n_colors=len(plot_indices))
        for j, (idx, lab) in enumerate(zip(plot_indices, labels)):
            ax = axes[j+1]
            gt_series = gt.gt_xarr.data[0, :, idx]
            ax.plot(dates, gt_series, color='k', lw=1.5)
            ts = arr[:, 0, :, idx]
            # Sample trajectories (only where masked)
            if n_sample_trajs and n_sample_trajs > 0:
                ns = min(n_sample_trajs, ts.shape[0])
                sample_idxs = np.linspace(0, ts.shape[0]-1, num=ns, dtype=int)
                keep = mk[0, :ts.shape[1], idx]
                for si in sample_idxs:
                    y = ts[si, :len(dates)].copy()
                    y[keep == 1] = np.nan
                    ax.plot(dates[:len(y)], y, color=palette[j], alpha=0.25, lw=0.7)
            # Quantile fans and median (only where masked)
            for lo, hi in flusight_quantile_pairs:
                lo_curve = np.quantile(ts, lo, axis=0)
                hi_curve = np.quantile(ts, hi, axis=0)
                keepw = mk[0, :len(lo_curve), idx]
                lo_curve = lo_curve.copy(); hi_curve = hi_curve.copy()
                lo_curve[keepw == 1] = np.nan
                hi_curve[keepw == 1] = np.nan
                ax.fill_between(dates[:len(lo_curve)], lo_curve, hi_curve, color=palette[j], alpha=0.06, lw=0)
            if plot_median:
                med = np.quantile(ts, 0.5, axis=0)
                med_masked = med.copy()
                med_masked[mk[0, :len(med), idx] == 1] = np.nan
                ax.plot(dates[:len(med_masked)], med_masked, color=palette[j], lw=1.8)
            # Corner label
            ax.text(0.02, 0.98, str(lab).upper(), transform=ax.transAxes, va='top', ha='left',
                    fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.set_ylabel('Incident flu hospitalizations')
            ax.set_xlabel('Date')
            sns.despine(ax=ax, trim=True)
        fig.tight_layout()
    
        os.makedirs(os.path.join(FIG_DIR, "mask_figures"), exist_ok=True)
        out_path = os.path.join(FIG_DIR, "mask_figures", f"{_MODEL_NUM}_mask_{name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        outs.append(out_path)
    return outs


MASK_RESULTS_DIR = "from_longleaf/mask_experiments_868_celebahq_noTTJ5/"
MASK_FORECAST_DATE = "2025-05-14"

if os.path.isdir(MASK_RESULTS_DIR):
    outputs = plot_mask_experiments(
        mask_dir=MASK_RESULTS_DIR,
        forecast_date=MASK_FORECAST_DATE,
        states=('NC', 'CA'),
        plot_median=False,
    )
    print("Mask figures:", outputs)

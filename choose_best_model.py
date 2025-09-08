# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: diffusion_torch
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import forestplot as fp
from influpaint.batch.config import CONFIG_BASELINE
from influpaint.batch.scenarios import get_training_scenario

# Load data
leaderboards = pd.read_csv('results/leaderboards/leaderboard_full.csv')

# %% Basic data exploration
leaderboard_piv = leaderboards.pivot(index=['season', 'model'], columns='metric', values='score').reset_index()

# %%
seasons = ['Combined', '2023-2024', '2024-2025']
fig, axes = plt.subplots(3, 1, figsize=(5, 15))

for ax, season in zip(axes, seasons):
    sub = leaderboard_piv[leaderboard_piv.season == season]
    sub = sub[sub.relative_wis < 1.5]  # filter out relative_wis > 2
    sns.scatterplot(data=sub, x='wis', y='relative_wis', hue='model', ax=ax, legend=False, s=50)
    ax.set_title(season)
    ax.set_xlabel('wis')
    ax.set_ylabel('relative_wis')

plt.tight_layout()

# %%
# Initialize empty list to collect data
model_data = []

for season in ['2023-2024', '2024-2025']:
    sub = leaderboard_piv[leaderboard_piv.season == season]
    # Use all models for this season
    all_models_season = sub.copy()
    all_models_season['rank_relative_wis'] = all_models_season['relative_wis'].rank()
    all_models_season['rank_wis'] = all_models_season['wis'].rank()
    all_models_season['sum_rank'] = all_models_season['rank_relative_wis'] + all_models_season['rank_wis']
    
    # Add season-specific data
    season_data = all_models_season[['model', 'rank_relative_wis', 'rank_wis', 'relative_wis', 'wis', 'sum_rank']].copy()
    season_data['season'] = season
    model_data.append(season_data)

# Combine all data
all_models = pd.concat(model_data).reset_index(drop=True)

# Get unique models across all seasons
unique_models = all_models['model'].unique()

# Create final dataframe with seasons as columns
model_rank = pd.DataFrame({'model': unique_models})

# Add columns for each season
for season in ['2023-2024', '2024-2025']:
    season_data = all_models[all_models.season == season].set_index('model')
    season_short = season.replace('202', '2')
    
    # Add rank columns for this season
    model_rank[f'rk_rel_wis_{season_short}'] = model_rank['model'].map(season_data['rank_relative_wis'])
    model_rank[f'rk_wis_{season_short}'] = model_rank['model'].map(season_data['rank_wis'])
    model_rank[f'rel_wis_{season_short}'] = model_rank['model'].map(season_data['relative_wis'])
    model_rank[f'wis_{season_short}'] = model_rank['model'].map(season_data['wis'])
    model_rank[f'sum_rk_{season_short}'] = model_rank['model'].map(season_data['sum_rank'])

# Remove models that don't appear in any season (shouldn't happen with this logic, but just in case)
model_rank["sum_rk_all"] = model_rank[[col for col in model_rank.columns if col.startswith('rk')]].sum(axis=1)

# %%
model_rank[["model"]+[col for col in model_rank.columns if 'rk_' in col and 'sum_rk_2' not in col]].sort_values(by='sum_rk_all').reset_index(drop=True)

# %% Parameter effect analysis
def parse_scenario_id(model_name):
    """Extract scenario ID from model name"""
    match = re.match(r'i(\d+)::', model_name)
    return int(match.group(1))

# Parse all models and add parameter columns
parsed_models = []
for _, row in leaderboards.iterrows():
    scenario_id = parse_scenario_id(row['model'])
    scenario_spec = get_training_scenario(scenario_id)
    parsed_models.append({
        **row,
        "scenario_id": scenario_id,
        "ddpm_name": scenario_spec.ddpm_name,
        "unet_name": scenario_spec.unet_name,
        "dataset_name": scenario_spec.dataset_name,
        "transform_name": scenario_spec.transform_name,
        "enrich_name": scenario_spec.enrich_name,
    })

df_parsed = pd.DataFrame(parsed_models)

# Find baseline model (matching CONFIG_BASELINE parameters)
baseline_models = df_parsed[
    (df_parsed['ddpm_name'] == CONFIG_BASELINE['ddpm_name']) &
    (df_parsed['unet_name'] == CONFIG_BASELINE['unet_name']) &
    (df_parsed['dataset_name'] == CONFIG_BASELINE['dataset_name']) &
    (df_parsed['transform_name'] == CONFIG_BASELINE['transform_name']) &
    (df_parsed['enrich_name'] == CONFIG_BASELINE['enrich_name'])
]

# Get best performing model for baseline configuration
baseline_combined_wis = baseline_models[
    (baseline_models['season'] == 'Combined') & 
    (baseline_models['metric'] == 'wis') & 
    (baseline_models['aggregation'] == 'sum')
]
baseline_model = baseline_combined_wis.loc[baseline_combined_wis['score'].idxmin()]
baseline_wis = baseline_model['score']

# Get corresponding relative WIS for the same model
baseline_combined_rel = baseline_models[
    (baseline_models['season'] == 'Combined') & 
    (baseline_models['metric'] == 'relative_wis') & 
    (baseline_models['aggregation'] == 'mean') &
    (baseline_models['model'] == baseline_model['model'])
]
baseline_rel_wis = baseline_combined_rel['score'].iloc[0]

print(f"Baseline model: {baseline_model['model']}")
print(f"Baseline WIS: {baseline_wis:.0f}")
print(f"Baseline Relative WIS: {baseline_rel_wis:.3f}")

# %%
def extract_inpaint_config(model_name):
    """Extract and clean inpainting config from model name"""
    match = re.search(r'::inpaint_CoPaint::(\w+)', model_name)
    if match:
        config = match.group(1)
        return config.replace('celebahq_', '')  # Remove prefix
    return 'unknown'
# Analyze parameter effects for both WIS and relative WIS
def analyze_parameter_effect(param_name, baseline_value, baseline_wis, baseline_rel_wis):
    """Analyze effect of changing one parameter from baseline for both WIS and relative WIS"""
    results = []
    
    # Filter to Combined season
    combined_wis = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'wis') & 
        (df_parsed['aggregation'] == 'sum')
    ]
    
    combined_rel_wis = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'relative_wis') & 
        (df_parsed['aggregation'] == 'mean')
    ]
    
    # Get unique values for this parameter
    param_values = combined_wis[param_name].unique()
    
    for value in param_values:
        if value == baseline_value:
            continue
            
        # Find models that differ only in this parameter (WIS)
        matching_wis = combined_wis[
            (combined_wis['ddpm_name'] == (CONFIG_BASELINE['ddpm_name'] if param_name != 'ddpm_name' else value)) &
            (combined_wis['unet_name'] == (CONFIG_BASELINE['unet_name'] if param_name != 'unet_name' else value)) &
            (combined_wis['dataset_name'] == (CONFIG_BASELINE['dataset_name'] if param_name != 'dataset_name' else value)) &
            (combined_wis['transform_name'] == (CONFIG_BASELINE['transform_name'] if param_name != 'transform_name' else value)) &
            (combined_wis['enrich_name'] == (CONFIG_BASELINE['enrich_name'] if param_name != 'enrich_name' else value)) &
            (combined_wis[param_name] == value)
        ]
        
        # Find corresponding relative WIS models
        matching_rel_wis = combined_rel_wis[
            (combined_rel_wis['ddpm_name'] == (CONFIG_BASELINE['ddpm_name'] if param_name != 'ddpm_name' else value)) &
            (combined_rel_wis['unet_name'] == (CONFIG_BASELINE['unet_name'] if param_name != 'unet_name' else value)) &
            (combined_rel_wis['dataset_name'] == (CONFIG_BASELINE['dataset_name'] if param_name != 'dataset_name' else value)) &
            (combined_rel_wis['transform_name'] == (CONFIG_BASELINE['transform_name'] if param_name != 'transform_name' else value)) &
            (combined_rel_wis['enrich_name'] == (CONFIG_BASELINE['enrich_name'] if param_name != 'enrich_name' else value)) &
            (combined_rel_wis[param_name] == value)
        ]
        
        if len(matching_wis) > 0 and len(matching_rel_wis) > 0:
            # Take best scores for this parameter value
            best_wis = matching_wis['score'].min()
            best_rel_wis = matching_rel_wis['score'].min()
            
            wis_improvement = (baseline_wis - best_wis) / baseline_wis * 100
            rel_wis_improvement = (baseline_rel_wis - best_rel_wis) / baseline_rel_wis * 100
            
            results.append({
                'parameter': param_name,
                'value': value,
                'wis': best_wis,
                'relative_wis': best_rel_wis,
                'wis_improvement_pct': wis_improvement,
                'rel_wis_improvement_pct': rel_wis_improvement,
                'n_models': len(matching_wis)
            })
    
    return pd.DataFrame(results)

# Analyze all parameters including inpainting config
all_effects = []
for param, baseline_val in CONFIG_BASELINE.items():
    param_key = f"{param}"
    effects = analyze_parameter_effect(param_key, baseline_val, baseline_wis, baseline_rel_wis)
    all_effects.append(effects)

# Add inpainting config analysis - need to get baseline inpaint config first
baseline_inpaint_config = extract_inpaint_config(baseline_model['model'])
print(f"Baseline Inpaint Config: {baseline_inpaint_config}")

# Analyze inpainting config effects
inpaint_effects = []
unique_configs = df_parsed['model'].apply(extract_inpaint_config).unique()

for config in unique_configs:
    if config == baseline_inpaint_config or config == 'unknown':
        continue
    
    # Find models with this inpaint config
    config_models_wis = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'wis') & 
        (df_parsed['aggregation'] == 'sum') &
        (df_parsed['model'].apply(extract_inpaint_config) == config)
    ]
    
    config_models_rel = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'relative_wis') & 
        (df_parsed['aggregation'] == 'mean') &
        (df_parsed['model'].apply(extract_inpaint_config) == config)
    ]
    
    if len(config_models_wis) > 0 and len(config_models_rel) > 0:
        best_wis = config_models_wis['score'].min()
        best_rel_wis = config_models_rel['score'].min()
        
        wis_improvement = (baseline_wis - best_wis) / baseline_wis * 100
        rel_wis_improvement = (baseline_rel_wis - best_rel_wis) / baseline_rel_wis * 100
        
        inpaint_effects.append({
            'parameter': 'inpaint_config',
            'value': config,
            'wis': best_wis,
            'relative_wis': best_rel_wis,
            'wis_improvement_pct': wis_improvement,
            'rel_wis_improvement_pct': rel_wis_improvement,
            'n_models': len(config_models_wis)
        })

if inpaint_effects:
    inpaint_df = pd.DataFrame(inpaint_effects)
    all_effects.append(inpaint_df)

param_effects = pd.concat(all_effects, ignore_index=True)
param_effects["wis_improvement_pct"] = param_effects["wis_improvement_pct"].round(2)

# Create forest plot
# Prepare data for forest plot with clean category labels
def clean_parameter_name(param_name, baseline_inpaint_config=None):
    """Convert parameter names to clean display names with baseline reference"""
    name_map = {
        'ddpm_name': 'DDPM (ref: U500c)',
        'unet_name': 'U-Net (ref: Rx124)',
        'dataset_name': 'Dataset (ref: 30S70M)', 
        'transform_name': 'Transform (ref: Sqrt)',
        'enrich_name': 'Enrichment (ref: No)',
        'inpaint_config': f'Inpaint Config (ref: {baseline_inpaint_config or "noTTJ5"})'
    }
    return name_map.get(param_name, param_name)

def create_elegant_forest_plot(data, title="", figsize=(12, 8)):
    """
    Create an elegant forest plot for parameter effects.
    
    Parameters:
    -----------
    data : DataFrame with columns estimate, lower, upper, label, category
    title : str, plot title
    figsize : tuple, figure size
    
    Returns:
    --------
    matplotlib axes object
    """
    # Create the plot
    ax = fp.forestplot(
        data,
        estimate='estimate',
        ll='lower', 
        hl='upper',
        varlabel='label',
        annote=['estimate'],
        annoteheaders=['Effect (%)'],
        groupvar='category',
        xlabel='Improvement over Baseline (%)',
        figsize=figsize,
        flush=True,
        color_alt_rows=True,
        decimal_precision=2,
        xline=0,  # Reference line at 0
        xlinestyle='--',
        xlinecolor='red',
        **{'fontfamily': 'sans-serif'}
    )
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return ax

# Prepare data for forest plot
forest_data = []
for _, row in param_effects.iterrows():
    forest_data.append({
        'label': f"{clean_parameter_name(row['parameter'], baseline_inpaint_config)} = {row['value']}",
        'estimate': row['wis_improvement_pct'],
        'lower': row['wis_improvement_pct'],  # Simple confidence interval
        'upper': row['wis_improvement_pct'],
        'category': clean_parameter_name(row['parameter'], baseline_inpaint_config)
    })



forest_df = pd.DataFrame(forest_data)

# Create elegant forest plot
ax = create_elegant_forest_plot(forest_df)
plt.show()

# Print summary table
print("\nParameter Effects Summary:")
print("=" * 80)
param_effects_sorted = param_effects.sort_values('wis_improvement_pct', ascending=False)
for _, row in param_effects_sorted.iterrows():
    print(f"{row['parameter']:>15} = {row['value']:<15} | WIS: {row['wis_improvement_pct']:>6.1f}% | RelWIS: {row['rel_wis_improvement_pct']:>6.1f}% | N: {row['n_models']}")

# %%

# %%
forest_df

# %%
# Load MLflow loss data and analyze relationship with forecast performance
mlflow_losses = pd.read_csv('mlflow_losses.csv')
mlflow_timeseries = pd.read_csv('mlflow_loss_timeseries.csv')

# Get models that are in the leaderboard
leaderboard_scenario_ids = df_parsed['scenario_id'].unique()
filtered_losses = mlflow_losses[mlflow_losses['scenario_id'].isin(leaderboard_scenario_ids)]
filtered_timeseries = mlflow_timeseries[mlflow_timeseries['scenario_id'].isin(leaderboard_scenario_ids)]

# %%
# Plot all loss timeseries together
# Set publication style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 0.8,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(12, 6))

# Best model is i804 - highlight it
best_model_id = 804

for scenario_id in filtered_timeseries['scenario_id'].unique():
    scenario_data = filtered_timeseries[filtered_timeseries['scenario_id'] == scenario_id]
    scenario_name = scenario_data['scenario_string'].iloc[0]
    if scenario_id != best_model_id:
        ax.plot(scenario_data['step'], scenario_data['loss'], 
                alpha=0.6, linewidth=1, label=f'{scenario_name}')
        
for scenario_id in filtered_timeseries['scenario_id'].unique():
    scenario_data = filtered_timeseries[filtered_timeseries['scenario_id'] == scenario_id]
    scenario_name = scenario_data['scenario_string'].iloc[0]
    if scenario_id == best_model_id:
        # Highlight best model with thicker line
        ax.plot(scenario_data['step'], scenario_data['loss'], 
                linewidth=2.5, alpha=0.9, color='k', label=f'{scenario_name} (Chosen Model)')

ax.set_xlabel('Training Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show legend inside the plot (upper right corner)
ax.legend(loc='upper right', frameon=True, fontsize=10, title="Model")

plt.tight_layout()
plt.show()

# %%
def prepare_loss_performance_data(df_parsed, filtered_losses):
    """Merge leaderboard performance data with MLflow training losses"""
    # Get Combined season performance metrics
    combined_wis = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'wis') & 
        (df_parsed['aggregation'] == 'sum')
    ][['scenario_id', 'model', 'score', 'dataset_name']].rename(columns={'score': 'wis'})

    combined_rel_wis = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'relative_wis') & 
        (df_parsed['aggregation'] == 'mean')
    ][['scenario_id', 'model', 'score']].rename(columns={'score': 'relative_wis'})

    # Merge data
    loss_performance = filtered_losses.merge(combined_wis, on='scenario_id', how='inner')
    loss_performance = loss_performance.merge(combined_rel_wis[['scenario_id', 'model', 'relative_wis']], 
                                            on=['scenario_id', 'model'], how='inner')
    
    # Add inpainting config
    loss_performance['inpaint_config'] = loss_performance['model'].apply(extract_inpaint_config)
    
    return loss_performance

# Prepare merged dataset
loss_performance = prepare_loss_performance_data(df_parsed, filtered_losses)

# Exclude models with relative WIS > 2.0
excluded_models = loss_performance[loss_performance['relative_wis'] > 1.5]
loss_performance_filtered = loss_performance[loss_performance['relative_wis'] <= 1.5]

print(f"Filtered {len(loss_performance_filtered)} models (excluded {len(excluded_models)} with relative WIS > 2.0)")

# %%
def plot_scatter(data, x_col, y_col, xlabel, ylabel, scale_y=False, ax=None, show_legend=True):
    """Create scatter plot with all model labels"""
    # Separate datasets
    ds_30S70M_mask = data['dataset_name_y'] == '30S70M'
    main_dataset = data[ds_30S70M_mask]
    other_datasets = data[~ds_30S70M_mask]
    
    # Identify best model
    best_model_mask = ((data['scenario_id'] == 804) & 
                      (data['inpaint_config'] == 'noTTJ5'))
    best_model_data = data[best_model_mask]
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        standalone = True
    else:
        fig = ax.figure
        standalone = False
    
    # Add scatter points for different inpainting configs with unique markers
    config_markers = {
        'noTTJ5': 'D',      # Large cross
        'try3': 'o ',        # Plus
        'celebahq': 'X'     # Diamond
    }
    
    # Plot each config with its specific marker
    for config, marker in config_markers.items():
        config_data = data[data['inpaint_config'] == config]
        if not config_data.empty:
            # Separate by dataset for consistent coloring
            config_main = config_data[config_data['dataset_name_y'] == '30S70M']
            config_other = config_data[config_data['dataset_name_y'] != '30S70M']
            

            y_config_main = config_main[y_col] / 1000 if scale_y else config_main[y_col]
            ax.scatter(config_main[x_col], y_config_main, 
                        marker=marker, s=80, color='steelblue', alpha=0.8,
                        #edgecolors='black', linewidth=1, 
                        label=f'{config} (30S70M)' if show_legend else None)
            
            y_config_other = config_other[y_col] / 1000 if scale_y else config_other[y_col]
            ax.scatter(config_other[x_col], y_config_other, 
                        marker=marker, s=80, color='lightcoral', alpha=0.8,
                        #edgecolors='black', linewidth=1,
                        label=f'{config} (Other)' if show_legend else None)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_legend:
        ax.legend(fontsize=10, frameon=True)

    # Label all models
    for _, row in data.iterrows():
        y_val = row[y_col] / 1000 if scale_y else row[y_col]
        label = f"i{row['scenario_id']}_{row['inpaint_config']}"
        if ((row['scenario_id'] == 804) and (row['inpaint_config'] == 'noTTJ5')):
            label += " (CHOICE)"
            ax.annotate(label, (row[x_col], y_val), 
                       xytext=(8, 8), textcoords='offset points', 
                       fontsize=9, alpha=1.0, fontweight='bold', color='red')
        else:
            ax.annotate(label, (row[x_col], y_val), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, alpha=0.8)
    
    if standalone:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

# Create subplot figure with A and B panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Average loss vs Relative WIS
plot_scatter(loss_performance_filtered, 'avg_loss_last_100', 'relative_wis',
            'Average Loss (Last 100 Steps)', 'Relative WIS', ax=ax1, show_legend=False)

# Panel B: Average loss vs WIS  
plot_scatter(loss_performance_filtered, 'avg_loss_last_100', 'wis',
            'Average Loss (Last 100 Steps)', 'WIS (×1000)', scale_y=True, ax=ax2, show_legend=True)

# Add panel labels
ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.show()

# Print correlation analysis
print("\nCorrelation Analysis (filtered data, relative WIS ≤ 2.0):")
print("=" * 60)
print(f"Final Loss vs Relative WIS: {loss_performance_filtered['final_loss'].corr(loss_performance_filtered['relative_wis']):.3f}")
print(f"Final Loss vs WIS: {loss_performance_filtered['final_loss'].corr(loss_performance_filtered['wis']):.3f}")
print(f"Avg Loss (Last 100) vs Relative WIS: {loss_performance_filtered['avg_loss_last_100'].corr(loss_performance_filtered['relative_wis']):.3f}")
print(f"Avg Loss (Last 100) vs WIS: {loss_performance_filtered['avg_loss_last_100'].corr(loss_performance_filtered['wis']):.3f}")

# %%
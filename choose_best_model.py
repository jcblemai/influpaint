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
baseline_combined = baseline_models[
    (baseline_models['season'] == 'Combined') & 
    (baseline_models['metric'] == 'wis') & 
    (baseline_models['aggregation'] == 'sum')
]
baseline_model = baseline_combined.loc[baseline_combined['score'].idxmin()]
baseline_wis = baseline_model['score']

print(f"Baseline model: {baseline_model['model']}")
print(f"Baseline WIS: {baseline_wis:.0f}")

# %%
# Analyze parameter effects
def analyze_parameter_effect(param_name, baseline_value):
    """Analyze effect of changing one parameter from baseline"""
    results = []
    
    # Filter to Combined season, WIS metric, sum aggregation
    combined_wis = df_parsed[
        (df_parsed['season'] == 'Combined') & 
        (df_parsed['metric'] == 'wis') & 
        (df_parsed['aggregation'] == 'sum')
    ]
    
    # Get unique values for this parameter
    param_values = combined_wis[param_name].unique()
    
    for value in param_values:
        if value == baseline_value:
            continue
            
        # Find models that differ only in this parameter
        matching_models = combined_wis[
            (combined_wis['ddpm_name'] == (CONFIG_BASELINE['ddpm_name'] if param_name != 'ddpm_name' else value)) &
            (combined_wis['unet_name'] == (CONFIG_BASELINE['unet_name'] if param_name != 'unet_name' else value)) &
            (combined_wis['dataset_name'] == (CONFIG_BASELINE['dataset_name'] if param_name != 'dataset_name' else value)) &
            (combined_wis['transform_name'] == (CONFIG_BASELINE['transform_name'] if param_name != 'transform_name' else value)) &
            (combined_wis['enrich_name'] == (CONFIG_BASELINE['enrich_name'] if param_name != 'enrich_name' else value)) &
            (combined_wis[param_name] == value)
        ]
        
        # Take best score for this parameter value
        best_score = matching_models['score'].min()
        improvement = (baseline_wis - best_score) / baseline_wis * 100
        
        results.append({
            'parameter': param_name,
            'value': value,
            'wis': best_score,
            'improvement_pct': improvement,
            'n_models': len(matching_models)
        })
    
    return pd.DataFrame(results)

# Analyze all parameters
all_effects = []
for param, baseline_val in CONFIG_BASELINE.items():
    param_key = f"{param}"
    effects = analyze_parameter_effect(param_key, baseline_val)
    all_effects.append(effects)

param_effects = pd.concat(all_effects, ignore_index=True)

# Create forest plot
# Prepare data for forest plot
forest_data = []
for _, row in param_effects.iterrows():
    forest_data.append({
        'label': f"{row['parameter']} = {row['value']}",
        'estimate': row['improvement_pct'],
        'lower': row['improvement_pct'] - 1,  # Simple confidence interval
        'upper': row['improvement_pct'] + 1,
        'category': row['parameter']
    })

forest_df = pd.DataFrame(forest_data)

# Create forest plot
fig = fp.forestplot(
    forest_df,
    estimate='estimate',
    varlabel='label',
    xlabel='Improvement over Baseline (%)',
    color_alt_rows=True,
    groupvar='category',
    figsize=(12, 8),
    annote=['estimate'],
    annoteheaders=['Effect (%)'],
    rightannote=None
)

plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.title(f'Parameter Effects on WIS Performance\n(Baseline: {baseline_model["model"]})', fontsize=14, pad=20)
plt.show()

# Print summary table
print("\nParameter Effects Summary:")
print("=" * 80)
param_effects_sorted = param_effects.sort_values('improvement_pct', ascending=False)
for _, row in param_effects_sorted.iterrows():
    print(f"{row['parameter']:>15} = {row['value']:<15} | {row['improvement_pct']:>6.1f}% | WIS: {row['wis']:>8.0f} | N: {row['n_models']}")

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

ax.set_xlabel('Training Step', fontsize=12)
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
def extract_inpaint_config(model_name):
    """Extract and clean inpainting config from model name"""
    match = re.search(r'::inpaint_CoPaint::(\w+)', model_name)
    if match:
        config = match.group(1)
        return config.replace('celebahq_', '')  # Remove prefix
    return 'unknown'

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
excluded_models = loss_performance[loss_performance['relative_wis'] > 2.0]
loss_performance_filtered = loss_performance[loss_performance['relative_wis'] <= 2.0]

print(f"Filtered {len(loss_performance_filtered)} models (excluded {len(excluded_models)} with relative WIS > 2.0)")

# %%
def plot_scatter(data, x_col, y_col, xlabel, ylabel, scale_y=False):
    """Create scatter plot with all model labels"""
    # Separate datasets
    ds_30S70M_mask = data['dataset_name_y'] == '30S70M'
    main_dataset = data[ds_30S70M_mask]
    other_datasets = data[~ds_30S70M_mask]
    
    # Identify best model
    best_model_mask = ((data['scenario_id'] == 804) & 
                      (data['inpaint_config'] == 'noTTJ5'))
    best_model_data = data[best_model_mask]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot datasets
    y_main = main_dataset[y_col] / 1000 if scale_y else main_dataset[y_col]
    y_other = other_datasets[y_col] / 1000 if scale_y else other_datasets[y_col]
    
    ax.scatter(main_dataset[x_col], y_main, 
              alpha=0.6, color='steelblue', s=50, label='30S70M dataset', 
              edgecolors='white', linewidth=0.5)
    ax.scatter(other_datasets[x_col], y_other, 
              alpha=0.6, color='lightcoral', s=50, label='Other datasets', 
              edgecolors='white', linewidth=0.5)
    
    # Highlight best model
    if not best_model_data.empty:
        y_best = best_model_data[y_col] / 1000 if scale_y else best_model_data[y_col]
        ax.scatter(best_model_data[x_col], y_best, 
                  marker='o', s=120, color='red', label='Best Model (i804_noTTJ5)', 
                  edgecolors='black', linewidth=3, alpha=0.8)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, frameon=True)
    
    # Label all models
    for _, row in data.iterrows():
        y_val = row[y_col] / 1000 if scale_y else row[y_col]
        label = f"i{row['scenario_id']}_{row['inpaint_config']}"
        if ((row['scenario_id'] == 804) and (row['inpaint_config'] == 'noTTJ5')):
            label += " (CHOICE)"
            ax.annotate(label, (row[x_col], y_val), 
                       xytext=(8, 8), textcoords='offset points', 
                       fontsize=8, alpha=1.0, fontweight='bold', color='red')
        else:
            ax.annotate(label, (row[x_col], y_val), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=7, alpha=0.8)
    
    plt.tight_layout()
    plt.show()

# Create scatter plots
plot_scatter(loss_performance_filtered, 'final_loss', 'relative_wis', 
            'Final Training Loss', 'Relative WIS')

plot_scatter(loss_performance_filtered, 'final_loss', 'wis', 
            'Final Training Loss', 'WIS (×1000)', scale_y=True)

plot_scatter(loss_performance_filtered, 'avg_loss_last_100', 'relative_wis',
            'Average Loss (Last 100 Steps)', 'Relative WIS')

plot_scatter(loss_performance_filtered, 'avg_loss_last_100', 'wis',
            'Average Loss (Last 100 Steps)', 'WIS (×1000)', scale_y=True)

# Print correlation analysis
print("\nCorrelation Analysis (filtered data, relative WIS ≤ 2.0):")
print("=" * 60)
print(f"Final Loss vs Relative WIS: {loss_performance_filtered['final_loss'].corr(loss_performance_filtered['relative_wis']):.3f}")
print(f"Final Loss vs WIS: {loss_performance_filtered['final_loss'].corr(loss_performance_filtered['wis']):.3f}")
print(f"Avg Loss (Last 100) vs Relative WIS: {loss_performance_filtered['avg_loss_last_100'].corr(loss_performance_filtered['relative_wis']):.3f}")
print(f"Avg Loss (Last 100) vs WIS: {loss_performance_filtered['avg_loss_last_100'].corr(loss_performance_filtered['wis']):.3f}")

# %%
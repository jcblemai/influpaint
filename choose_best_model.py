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
leaderboards = pd.read_csv('results/leaderboards/leaderboard_full.csv')

# %%
leaderboards

# %%
print(leaderboards.metric.unique())
print(leaderboards.aggregation.unique())
print(leaderboards.season.unique())

# %%
leaderboard_piv = leaderboards.pivot(index=['season', 'model'], columns='metric', values='score').reset_index()
leaderboard_piv

# %%
leaderboard_piv

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
    # set x axis as log scale
    #ax.set_xscale('log')

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


# %%

model_rank[["model"]+[col for col in model_rank.columns if 'rk_' in col and 'sum_rk_2' not in col]].sort_values(by='sum_rk_all').reset_index(drop=True)


# %%

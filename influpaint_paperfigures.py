# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
from influpaint.utils.plotting import show_tensor_image
from influpaint.utils.helpers import flusight_quantile_pairs, flusight_quantiles
from influpaint.utils import ground_truth
import datetime
import os


image_size = 64
channels = 1
batch_size=512

do_inpainting = True
do_training = False

# %% [markdown]
# Notebook forked from influpaint.ipynb, and modified to do the paper figures
# ## Load forecasts from best model

# %%
def load_training_samples():

    """Load training samples from the best model for Figure 1 (unconditional generation)."""
    import glob
    
    # Training results directory

    samples_path = os.path.join("from_longleaf/regen/samples_regen/inverse_transformed_samples_i804::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_No.npy")

    training_samples = np.load(samples_path)
    
    return training_samples

def load_all_inpainting_forecasts():
    """Load all inpainting forecasts for the best model from all dates."""
    import glob
    
    # Best model identified from choose_best_model.py analysis
    best_model_id = "i804"
    best_config = "celebahq_noTTJ5"
    
    # Inpainting results directory
    inpainting_base = "from_longleaf/influpaint_res/07b44fa_paper-2025-07-22_inpainting_2025-07-27"
    
    # Find all directories matching best model and config
    pattern = f"{inpainting_base}/{best_model_id}*conf_{best_config}*"
    matching_dirs = glob.glob(pattern)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No inpainting directories found matching pattern: {pattern}")
    
    # Load fluforecasts_ti.npy from all directories
    all_forecasts = []
    forecast_dates = []
    
    for forecast_dir in sorted(matching_dirs):
        fluforecasts_ti_path = os.path.join(forecast_dir, "fluforecasts_ti.npy")
        
        if not os.path.exists(fluforecasts_ti_path):
            print(f"Warning: fluforecasts_ti.npy not found in {forecast_dir}")
            continue
            
        fluforecasts_ti = np.load(fluforecasts_ti_path)
        all_forecasts.append(fluforecasts_ti)
        
        # Extract date from folder name
        folder_name = os.path.basename(forecast_dir)
        date_str = folder_name.split("::")[-1]  # Last part is the date
        forecast_dates.append(date_str)
        
    print(f"Loaded {len(all_forecasts)} forecast sets from {len(matching_dirs)} directories")
    if all_forecasts:
        print(f"Each forecast shape: {all_forecasts[0].shape}")
    
    return all_forecasts, forecast_dates

# Load training samples for Figure 1
training_samples = load_training_samples()

# Load all inpainting forecasts
all_inpainting_forecasts, forecast_dates = load_all_inpainting_forecasts()

# Create figures directory
os.makedirs('figures', exist_ok=True)


# %% [markdown]
# ## Display loaded forecasts
# 
# Display the forecasts loaded from the best model:
# `fluforecasts_ti.shape` is (batch_size, channels, epiweek, locations)

# %%
# Display training samples (unconditional generation)
random_index = 0
fig, axes = plt.subplots(1, 1, figsize=(16,3), dpi=100)
ax = axes
show_tensor_image(training_samples[random_index], ax = ax)
plt.show()
plt.imshow(training_samples[random_index].reshape(image_size, image_size, channels))
plt.show()

# %%
fig, axes = plt.subplots(1, 1, figsize=(16,3), dpi=100)
ax = axes
for i in range(min(batch_size, training_samples.shape[0])):
    show_tensor_image(training_samples[i], ax = ax)

# %%
# histogram of peaks for training samples. In the US historically it's from 13k to 34k
plt.hist(training_samples[:,0,:,:].sum(axis=2).max(axis=1), bins=30)
print(f"mean peak is {training_samples[:,0,:,:].sum(axis=2).max(axis=1).mean()/1000:.1f}k (US historically it's from 13k to 34k)")

# %%
# %% [markdown]
# ## Paper Figures

# %%


# Custom with specific labels for Figure 1 - training samples (unconditional generation)
indices_lscustom = [10, 25, 50]
labels_custom = ['(a) Curve A', '(b) Curve B', '(c) Curve C']

def create_figure_1_unconditional(samples, indices, labels, save_path=None):
    """Create Figure 1 showing unconditional generation samples."""
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    
    for i, (idx, label) in enumerate(zip(indices, labels)):
        ax = axes[i]
        # Plot time series for a specific sample and location
        sample_data = samples[idx, 0, :, :].sum(axis=1)  # Sum over locations, plot over time
        ax.plot(sample_data)
        ax.set_title(label)
        ax.set_xlabel('Week')
        ax.set_ylabel('Cases')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 1 to: {save_path}")
    
    plt.show()
    return fig

# Create Figure 1
create_figure_1_unconditional(training_samples, indices_lscustom, labels_custom, 
                             save_path='figures/fig1_unconditional.png')

# %%
def load_forecast_csv(csv_path):
    """Load and process forecast CSV file"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df

def load_truth_data(truth_path="/users/c/h/chadi/influpaint/Flusight/2022-2023/FluSight-forecast-hub-official/data-truth/truth-Incident Hospitalizations.csv"):
    """Load ground truth data from FluSight"""
    import pandas as pd
    import os
    if os.path.exists(truth_path):
        truth_df = pd.read_csv(truth_path)
        truth_df['date'] = pd.to_datetime(truth_df['date'])
        return truth_df
    else:
        print(f"Truth file not found: {truth_path}")
        return None

def create_figure_2_forecasting(csv_files, states, save_path=None):
    """
    Figure 2: Forecasting case studies showing full flu season with forecasts overlaid
    • Shows complete 2022-2023 flu season with ground truth
    • Overlays forecast fans at their respective forecast dates
    • Demonstrates different forecasting scenarios across the season
    """
    import pandas as pd
    import datetime as dt
    
    # Load ground truth data
    truth_df = load_truth_data()
    
    # Set paper-ready style
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Create single large plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=300)
    
    # Define colors for different forecasts
    forecast_colors = ['#FF4444', '#4444FF', '#44AA44', '#AA44AA']
    
    # Plot full season ground truth first
    if truth_df is not None:
        # Use first available state for demonstration
        state = states[0] if states else '01'
        state_truth = truth_df[truth_df['location'] == state]
        
        if len(state_truth) > 0:
            # Get full 2022-2023 flu season data
            season_start = pd.to_datetime('2022-10-01')
            season_end = pd.to_datetime('2023-05-31')
            
            season_truth = state_truth[
                (state_truth['date'] >= season_start) & 
                (state_truth['date'] <= season_end)
            ].sort_values('date')
            
            if len(season_truth) > 0:
                print(f"Ground truth dates: {season_truth['date'].min().date()} to {season_truth['date'].max().date()}")
                
                # Plot full season ground truth using actual dates
                ax.plot(season_truth['date'], season_truth['value'], 
                       color='black', linewidth=3, marker='o', markersize=4,
                       label='Observed Ground Truth', markerfacecolor='white', 
                       markeredgecolor='black', markeredgewidth=1)
    
    # Now overlay each forecast at its respective date
    legend_labels = []
    for i, (csv_file, state, color) in enumerate(zip(csv_files, states, forecast_colors)):
        # Load forecast data
        df = load_forecast_csv(csv_file)
        print(f"Processing forecast {i+1}: {csv_file.split('/')[-1]}")
        
        # Filter for the state
        state_data = df[df['location'] == state] if 'location' in df.columns else df
        
        if len(state_data) == 0:
            available_states = df['location'].unique()[:5]
            if len(available_states) > 0:
                state = available_states[0]
                state_data = df[df['location'] == state]
        
        if len(state_data) == 0:
            continue
            
        # Sort by target_end_date
        state_data = state_data.sort_values('target_end_date')
        
        # Extract forecast date from CSV data (more reliable than filename)
        if 'forecast_date' in state_data.columns:
            forecast_date = pd.to_datetime(state_data['forecast_date'].iloc[0])
            print(f"  Forecast date from CSV: {forecast_date.date()}")
        else:
            # Fallback to filename parsing
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})\.csv$', csv_file)
            if not date_match:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', csv_file.split('/')[-1])
            
            if not date_match:
                print(f"  Could not extract date from filename: {csv_file}")
                continue
                
            forecast_date = pd.to_datetime(date_match.group(1))
            print(f"  Forecast date from filename: {forecast_date.date()}")
        
        # Get quantiles
        available_quantiles = sorted(state_data['quantile'].unique())
        quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
        quantiles = [q for q in quantiles if q in available_quantiles]
        
        # Plot forecast uncertainty bands
        for q_idx in range(len(quantiles)//2):
            lower_q = quantiles[q_idx]
            upper_q = quantiles[-(q_idx+1)]
            
            lower_data = state_data[state_data['quantile'] == lower_q].sort_values('target_end_date')
            upper_data = state_data[state_data['quantile'] == upper_q].sort_values('target_end_date')
            
            if len(lower_data) > 0 and len(upper_data) > 0:
                # Use actual target_end_dates for x-axis
                forecast_dates = pd.to_datetime(lower_data['target_end_date'])
                alpha = 0.3 - q_idx * 0.05
                
                ax.fill_between(forecast_dates, 
                              lower_data['value'].values, 
                              upper_data['value'].values, 
                              alpha=alpha, color=color, edgecolor='none')
        
        # Plot median forecast
        median_data = state_data[state_data['quantile'] == 0.5].sort_values('target_end_date')
        if len(median_data) > 0:
            forecast_dates = pd.to_datetime(median_data['target_end_date'])
            
            ax.plot(forecast_dates, median_data['value'], 
                   color=color, linewidth=3, linestyle='-',
                   label=f'Forecast {i+1} ({forecast_date.strftime("%b %d")})')
        
        # Add vertical line at forecast date
        ax.axvline(forecast_date, color=color, linestyle='--', alpha=0.7, linewidth=2)
    
    # Formatting
    ax.set_xlabel('Date (2022-2023 Flu Season)', fontsize=14)
    ax.set_ylabel('Flu Hospitalizations', fontsize=14)
    ax.set_title('Figure 2: Forecasting Case Studies - Full Season Context\nInfluPaint Inpainting Results', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Final layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')

    plt.show()
    return fig

# Example usage for Figure 2 - Load forecast CSVs and create case studies
forecast_base_path = "/users/c/h/chadi/influpaint_res/3d47f4a_2023-11-07/forecasts_noTT/test::model_MyUnet200::dataset_R1Fv::trans_Sqrt::enrich_PoisPadScaleSmall::800::inpaint_CoPaint::conf_celebahq_noTT2/"

# Select representative dates and states for the four case studies
case_study_files = [
    forecast_base_path + "test::model_MyUnet200::dataset_RFv::trans_Sqrt::enrich_PoisPadScaleSmall::800::inpaint_CoPaint::conf_celebahq_noTT2-2022-11-14.csv",
    forecast_base_path + "test::model_MyUnet200::dataset_R1Fv::trans_Sqrt::enrich_PoisPadScaleSmall::800::inpaint_CoPaint::conf_celebahq_noTT2-2022-10-17.csv", 
    forecast_base_path + "test::model_MyUnet200::dataset_R1Fv::trans_Sqrt::enrich_PoisPadScaleSmall::800::inpaint_CoPaint::conf_celebahq_noTT2-2023-02-06.csv",
    forecast_base_path + "test::model_MyUnet200::dataset_R1Fv::trans_Sqrt::enrich_PoisPadScaleSmall::800::inpaint_CoPaint::conf_celebahq_noTT2-2022-12-12.csv"
]

# Representative states for each case study (using numeric FIPS codes)
case_study_states = ["01", "02", "04", "05"]  # Alabama, Alaska, Arizona, Arkansas

# Check if files exist and create the figure
import os
existing_files = [f for f in case_study_files if os.path.exists(f)]

if len(existing_files) >= 2:  # Need at least 2 files for meaningful comparison
    create_figure_2_forecasting(existing_files[:4], case_study_states[:len(existing_files[:4])], 
                               save_path='figure_2_forecasting.png')
else:
    print(f"Found {len(existing_files)} forecast files. Need at least 2 for Figure 2.")
    print("Available files:")
    for f in existing_files:
        print(f"  {f}")



# %%
# fig, axes = plt.subplots(8, 7, figsize=(16,16), dpi=100)
# 
# for ipl in range(51):
#     ax = axes.flat[ipl]
#     for i in range(batch_size):
#         show_tensor_image(dataset.apply_transform_inv(samples[-1][i]), ax = ax, place=ipl, multi=True)

# %%
animate = False
if animate:
    import matplotlib.animation as animation

    random_index = 53
# TODO: the reshape shuffles the information
    fig = plt.figure()
    ims = []
    for i in range(ddpm1.timesteps):
        im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')
    plt.show()
if animate:
    plt.ioff()
    for ts in tqdm(range(0, ddpm1.timesteps, 5)):
        fig, axes = plt.subplots(8, 8, figsize=(10,10))
        for ipl in range(51):
            ax = axes.flat[ipl]
            for i in range(0,batch_size, 2):
                show_tensor_image(dataset.apply_transform_inv(samples[ts][i]), ax = ax, place=ipl)
        plt.savefig(f'results/{ts}.png')
        plt.close(fig)



# %%
if do_inpainting:
    gt1 = ground_truth.GroundTruth(season_first_year="2024", 
                                data_date=datetime.datetime.today(),
                                mask_date=datetime.datetime.today(),
                                channels=channels,
                                image_size=image_size
                                )
    gt1.plot_mask()
    gt1.plot()

# %%
channels



# %%


# %% [markdown]
# ## Plot forecasts from best model



# %%
def plot_all_inpainting_forecasts_with_ground_truth():
    """Plot all inpainting forecasts against ground truth."""
    
    # Load ground truth for the latest forecast date to get the season
    if not forecast_dates:
        print("No forecast dates available")
        return
        
    # Use the latest date to determine season
    latest_date = sorted(forecast_dates)[-1]
    latest_datetime = pd.to_datetime(latest_date)
    
    # Load ground truth
    from influpaint.utils.season_axis import SeasonAxis
    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    season_first_year = str(season_setup.get_fluseason_year(latest_datetime))
    
    gt1 = ground_truth.GroundTruth(
        season_first_year=season_first_year,
        data_date=datetime.datetime.today(),
        mask_date=latest_datetime,
        channels=channels,
        image_size=image_size,
        nogit=True
    )
    
    # Plot setup
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # Plot all forecasts
    for i, (forecasts, date_str) in enumerate(zip(all_inpainting_forecasts, forecast_dates)):
        # Compute national forecasts (sum across locations)
        forecasts_national = forecasts.sum(axis=-1)
        
        # Plot median forecast
        median_forecast = np.median(forecasts_national, axis=0)[0]  # [0] for first channel
        
        for iax in range(2):
            ax = axes[iax]
            
            # Plot quantile bands for this forecast
            for iqt in range(11):
                ax.fill_between(np.arange(64), 
                              np.quantile(forecasts_national, flusight_quantile_pairs[iqt,0], axis=0)[0], 
                              np.quantile(forecasts_national, flusight_quantile_pairs[iqt,1], axis=0)[0], 
                              alpha=0.05, color='darkred')
            
            # Plot median
            ax.plot(np.arange(64), 
                   np.quantile(forecasts_national, flusight_quantiles[12], axis=0)[0], 
                   color='red', alpha=0.7, linewidth=1)
    
    # Plot ground truth on both axes
    for iax in range(2):
        ax = axes[iax]
        ax.plot(gt1.gt_xarr.data[0,:gt1.inpaintfrom_idx].sum(axis=1), color='k', marker='.', ls='', markersize=3, label='Ground Truth')
        ax.axvline(gt1.inpaintfrom_idx-1, c='k', linestyle='--', alpha=0.7)
        
        if iax == 0:
            ax.set_xlim(0, 52)
            ax.set_ylim(bottom=0)
            ax.set_title('Full Season')
        else:
            ax.set_xlim(gt1.inpaintfrom_idx-4, gt1.inpaintfrom_idx+4)
            ax.set_ylim(bottom=0, top=1500)
            ax.set_title('Forecast Region (Zoomed)')
        
        ax.set_xlabel('Week')
        ax.set_ylabel('Hospitalizations')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'All Inpainting Forecasts vs Ground Truth\n{season_first_year}-{int(season_first_year)+1} Season', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/all_inpainting_forecasts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plotted {len(all_inpainting_forecasts)} forecast sets against ground truth")
    return fig

# Create the comprehensive forecasting plot
plot_all_inpainting_forecasts_with_ground_truth()

# %%
# fig, axes = plt.subplots(1, 1, figsize=(5,3), dpi=200)
# 
# ax = axes
# for i in range(batch_size)[::2]:
#     ax.plot(gt1.gt_xarr.data[0,:gt1.inpaintfrom_idx].sum(axis=1), color='k', marker='.', ls='')
#     ax.plot(dataset.apply_transform_inv(samples[-1][i])[0].sum(axis=1) , lw=.5, marker='.', markersize=1, markerfacecolor='black', markeredgecolor='black', color="lightcoral")
#     show_tensor_image(samples[-1][i], ax = ax, multi=True,)
#     ax.axvline(gt1.inpaintfrom_idx-1,  c='k', lw=1, ls='-.')
#     ax.set_xlim(0,52)
#     #ax.set_ylim(bottom=0, auto=True)
#     #ax.grid(visible = True)
#     ax.set_title("National")
#     sns.despine(ax = ax, trim = True, offset=4)
# fig.tight_layout()
# plt.show()

# %%
#fig, axes = plt.subplots(8, 7, figsize=(16,16), dpi=200, sharex=True)

#for ipl in range(51):
#    ax = axes.flat[ipl]
#    for i in range(min(50, batch_size)):  # print max 50 sims
#        ax.plot(gt_xarr.data[0,:inpaintfrom_idx, ipl], color='k')
#        show_tensor_image(samples[-1][i], ax = ax, place=ipl, multi=True)
#        ax.axvline(inpaintfrom_idx-1, c='k', lw=.7, ls='-.')
#        ax.set_xlim(0,52)
#        ax.set_ylim(bottom=0, auto=True)
#        ax.grid()
#        #ax.set_title(get_state_name(places[ipl]))
#fig.tight_layout()
##plt.savefig("inpainting.pdf")

# %%
plt.plot(gt1.gt_xarr.data[0,:gt1.inpaintfrom_idx].sum(axis=-1))

# %%


# This section has been replaced by the comprehensive plotting function above

# %%


# %%
import datetime

# Get today's date
today = datetime.datetime.today()

# Find the next Saturday
days_until_saturday = (5 - today.weekday()) % 7
next_saturday = today + datetime.timedelta(days=days_until_saturday)

print("Next Saturday's date is:", next_saturday.strftime("%Y-%m-%d"))
forecast_date = next_saturday.date()
team_abbrv = "UNC_IDD-InfluPaint"
forecast_date_str=str(forecast_date)

# %%
%rm -r output
%mkdir output

# %%
gt1.gt_df_final

# %%
gt1.gt_df_final




# %%
import importlib
ground_truth = importlib.reload(ground_truth)
gt1 = ground_truth.GroundTruth(season_first_year="2024", 
                            data_date=datetime.datetime.today(),#datetime.datetime(2024, 12, 3),
                            mask_date=datetime.datetime.today(),
                            channels=channels,
                            image_size=image_size
                            )

# %%
gt1.export_forecasts(fluforecasts_ti=fluforecasts_ti,
                        forecasts_national=forecasts_national,
                        directory="output",#f'output_{forecast_date_str}',
                        prefix=team_abbrv,
                        forecast_date=forecast_date,
                        save_plot=True)

# %%
# from importlib import reload
# ground_truth = reload(ground_truth)
# gt1 = ground_truth.GroundTruth(season_first_year="2023", 
#                                data_date=datetime.datetime.today(), #datetime.datetime(2023,10,25)
#                                mask_date=datetime.datetime.today(),
#                                channels=channels,
#                                image_size=image_size
#                                )

# %%
gt1.export_forecasts_2023(fluforecasts_ti=fluforecasts_ti,
                        forecasts_national=forecasts_national,
                        directory='output',
                        prefix=team_abbrv,
                        forecast_date=forecast_date,
                        save_plot=True)

# %% [markdown]
# ## Data Export

# %%
import requests
requests.post("https://ntfy.sh/chadi_modeling",
     data="Notebook finshed running !",
     headers={
         "Title": "Inpainting-diffusion",
         "Priority": "urgent",
         "Tags": "warning,tada"
     })



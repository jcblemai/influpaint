# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import epiframework
import numpy as np
import pandas as pd
import nn_blocks, idplots, ddpm, myutils, inpaint, ground_truth, epitransforms
import build_dataset, training_datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import datetime

import sys
sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler
from guided_diffusion import unet
from utils import config

image_size = 64
channels = 1
batch_size=512
epoch = 800
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device == "cuda":
    print(myutils.cuda_mem_info())
    torch.cuda.empty_cache() # make sure we don't keep old stuff
    print(myutils.cuda_mem_info())

do_inpainting = True
do_training = False

# %% [markdown]
# Notebook forked from influpaint.ipynb, and modified to do the paper figures
# ## Prepare ground-truth for inpainting

# %%
%ls /users/c/h/chadi/influpaint_res/3d47f4a_2023-11-07/*.pth

# %%
checkpoint_fn = "/users/c/h/chadi/influpaint_res/3d47f4a_2023-11-07/test::model_MyUnet500::dataset_R1::trans_Sqrt::enrich_No::800.pth"
#work with noTT2
checkpoint_fn = "/users/c/h/chadi/influpaint_res/3d47f4a_2023-11-07/test::model_MyUnet200::dataset_R1Fv::trans_Sqrt::enrich_PoisPadScaleSmall::800.pth"

# %%
model_str = checkpoint_fn.split('/')[-1]
for part in model_str.split('::')[1:-1]:
    print(part)
    tp, val = part.split('_')
    if tp == 'model':
        unet_spec = epiframework.model_libary(image_size=image_size, channels=channels, epoch=epoch, device=device, batch_size=batch_size)
        ddpm1 = unet_spec[val]
    elif tp == "dataset":
        dataset_spec = epiframework.dataset_library(gt1=gt1, channels=channels)
        dataset = dataset_spec[val]
    elif tp == "trans":
        scaling_per_channel = np.array(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])))
        transforms_spec, transform_enrich = epiframework.transform_library(scaling_per_channel=scaling_per_channel)
        transform = transforms_spec[val]
    elif tp == "enrich":
        scaling_per_channel = np.array(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])))
        transforms_spec, transform_enrich = epiframework.transform_library(scaling_per_channel=scaling_per_channel)
        enrich = transform_enrich[val]
dataset.add_transform(transform=transform["reg"], transform_inv=transform["inv"], transform_enrich=enrich, bypass_test=False)

# %%
ddpm1.load_model_checkpoint(checkpoint_fn)


# %% [markdown]
# ## Sampling (inference)
# 
# To sample from the model, we can just use our sample function defined above:
# `samples[-1][:,0,:,:].shape` is (512, 64, 64) which is batchsize, epiweek, locations

# %%
do_sampling=True
if do_sampling: 
    samples = ddpm1.sample()
    random_index = 0
    fig, axes = plt.subplots(1, 1, figsize=(16,3), dpi=100)
    ax = axes # es.flat[i]
    idplots.show_tensor_image(dataset.apply_transform_inv(samples[-1][random_index]), ax = ax)
    plt.show()
    plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels))
    plt.show()

# %%

if do_sampling:
    fig, axes = plt.subplots(1, 1, figsize=(16,3), dpi=100)
    ax = axes # es.flat[i]
    for i in range(batch_size):
        idplots.show_tensor_image(dataset.apply_transform_inv(samples[-1][i]), ax = ax)

    #ax.set_ylim(0,10000)

# %%
#plt.hist(samples[-1].flatten(), bins = 100);
#plt.plot(transform_inv(samples[-1][:,0,:,:].sum(axis=2)).T);

# %%
if do_sampling:
    # histogram of peaks. In the US historically it's from 13k to 34k
    plt.hist(dataset.transform_inv(samples[-1][:,0,:,:].sum(axis=2)).max(axis=1), bins=30)
print(f"mean peak is {dataset.transform_inv(samples[-1][:,0,:,:].sum(axis=2)).max(axis=1).mean()/1000:.1f}k (US historically it's from 13k to 34k)")

# %%
# %% [markdown]
# ## Paper Figures

# %%


if do_sampling:
    # Custom with specific labels for Figure 1
    indices_custom = [10, 25, 50]
    labels_custom = ['(a) Curve A', '(b) Curve B', '(c) Curve C']
    create_figure_2(samples, dataset, indices=indices_custom, season_labels=labels_custom, 
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
#         idplots.show_tensor_image(dataset.apply_transform_inv(samples[-1][i]), ax = ax, place=ipl, multi=True)

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
                idplots.show_tensor_image(dataset.apply_transform_inv(samples[ts][i]), ax = ax, place=ipl)
        plt.savefig(f'results/{ts}.png')
        plt.close(fig)



# %%
from importlib import reload
ground_truth = reload(ground_truth)
if do_inpainting:
    if False:       # if True, take a training data sample as gt:
        inpaintfrom_idx = 19
        gt_keep_mask = np.ones((channels,image_size,image_size))
        gt_keep_mask[:,inpaintfrom_idx:,:] = 0
        # mask is ones for the known pixels, and zero for the ones to be infered
        gt = data.getitem_nocast(4)
        print(gt.shape)
        show_tensor_image(gt, place=ipl)
        plt.show()

    else:
        gt1 = ground_truth.GroundTruth(season_first_year="2024", 
                                    data_date=datetime.datetime.today(),#datetime.datetime(2024, 12, 3),
                                    mask_date=datetime.datetime.today(),
                                    channels=channels,
                                    image_size=image_size
                                    )
        gt1.plot_mask()
        gt1.plot()

# %%
channels

# %% [markdown]
# ## Define a PyTorch Dataset + DataLoader

# %%
#dataset = training_datasets.FluDataset.from_fluview(
#                                         flusetup=gt1.flusetup,
#                                         download=False, 
#                                         #transform=transforms.transform_randomscale
#                                         )
dataset = training_datasets.FluDataset.from_SMHR1_fluview(season_setup=gt1.season_setup, download=False)
#data = training_datasets.FluDataset.from_csp_SMHR1('Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc', channels=channels)
#dataset = training_datasets.FluDataset.from_synthetic_dataset("synthetic_dataset/sir_dataset_1.nc", channels=channels)
dataset.test(6)

# %%
if do_inpainting:
    scaling_per_channel = np.array(np.sqrt(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])).data))
else:
    scaling_per_channel = np.array(np.sqrt(max(dataset.max_per_feature)))
scaling_per_channel

# %%
## define image transformations (e.g. using torchvision)

transform_enrich = transforms.Compose([
                        transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
                        transforms.Lambda(lambda t: epitransforms.transform_random_padintime(t, min_shift = -15, max_shift = 15)),
                        #transforms.Lambda(lambda t: epitransforms.transform_randomscale(t, min=.1, max=1.9)),
        ])
#                         transforms.Lambda(lambda t: epitransforms.transform_skewednoise(t, scale=.4, a=-1.8))
transform = transforms.Compose([
                        epitransforms.transform_sqrt,
                        transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale = 1/scaling_per_channel)),
                        transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale = 2)),

        ])

# TODO: scaling on incident hops scale.

transform_inv = transforms.Compose([
                    epitransforms.transform_sqrt_inv,
                    transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale = 1/scaling_per_channel)),
                    transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale = 2)),
        ][::-1])      # important reverse the sequence

# %%
dataset.add_transform(transform=transform, transform_inv=transform_inv, transform_enrich=transform_enrich, bypass_test=False)

# %%
sample = dataset[6]
print(f"There are {len(dataset)} samples in the dataset, and a single sample has shape {sample.shape}")

# %%
# Dataset is shuffled, but the last incompleted batch is dropped
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
batch = next(iter(dataloader))
print(f"batch shape {batch.shape}")

# %% [markdown]
# Next, we define a function which we'll apply on-the-fly on the entire dataset. We use the `with_transform` [functionality](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/main_classes#datasets.Dataset.with_transform) for that. The function just applies some basic image preprocessing: random horizontal flips, rescaling and finally make them have values in the $[-1,1]$ range.

# %%
fig, axes = plt.subplots(1,3, figsize=(7,2))
axes.flat[0].hist(dataset.get_sample_raw(4).flatten(), bins=15);
axes.flat[1].hist(dataset.get_sample_transformed(4).flatten(), bins=15);
axes.flat[2].hist(dataset.get_sample_transformed_enriched(4).flatten(), bins=15);

# %%
# fig, axes = plt.subplots(8, 7, figsize=(16,16), dpi=100, sharex=True, sharey=True)
# for ipl in range(51):
#     ax = axes.flat[ipl]
#     for i in range(batch_size):
#         ax.plot(batch[i][0][:,ipl], lw=0.5)

# %% [markdown]
# ## Initialize the DDPM class

# %%
model = nn_blocks.Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    use_convnext=False
)

ddpm1 = ddpm.DDPM(model=model, 
                    image_size=image_size, 
                    channels=channels, 
                    batch_size=batch_size, 
                    epochs=epoch, 
                    timesteps=200,
                    device=device)


# %%


# %%
if do_training:
    model = unet.UNetModel(
            image_size=image_size,
            in_channels=1,
            out_channels=1,
            num_res_blocks=3,
            model_channels=3,
            attention_resolutions=2,
        )

# %%
dataset
if device == "cuda":
    print(myutils.cuda_mem_info())
    torch.cuda.empty_cache() # make sure we don't keep old stuff
    print(myutils.cuda_mem_info())

# %% [markdown]
# ## Forward diffusion sample

# %%
# use seed for reproducability
#torch.manual_seed(0)
x_start = next(iter(dataloader))
print(f"batch shape {batch.shape}")

def plot(imgs, row_title=None, col_title=None, with_histogram=True, plot_size=2, **imshow_kwargs): # with_orig=False,
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    
    mult_rows=1
    if with_histogram:
        mult_rows=2

    place = 4
    color="lightcoral"#'slategray'
    print(gt1.season_setup.get_location_name(gt1.season_setup.locations[place]))
    
    num_rows = len(imgs)*mult_rows
    num_cols = len(imgs[0]) #+ with_orig
    fig, axs = plt.subplots(figsize=(plot_size*num_cols,plot_size*num_rows), nrows=num_rows, ncols=num_cols, squeeze=False, dpi=100)
    for row_idx, row in enumerate(imgs):
        # row = [transform_inv(a[place,:])] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx*mult_rows, col_idx]
            ax.plot(np.arange(52), transform_inv(np.asarray(img))[0,:52,place], **imshow_kwargs, color=color, lw=3, marker='.', markersize=5, markerfacecolor='black', markeredgecolor='black')
            ax.set_xticks(np.arange(0,53,13))
            sns.despine(ax = ax,  offset = 10, trim = True)
            if col_idx>0:
                ax.set(yticklabels=[], yticks=[])
                sns.despine(ax = ax,  offset = 10, left=True, trim = True)

            
            #ax.set_ylim(0)
            #ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if with_histogram:
                ax = axs[row_idx*mult_rows+1, col_idx]
                ax.hist(np.asarray(img).flatten(), bins=20, color='slategray', alpha=0.7, lw=1, edgecolor='k', histtype='stepfilled')
                ax.set_xticks([-1,0,1])
                ax.set(yticklabels=[], yticks=[])
                sns.despine(ax = ax,  offset = 4, left=True, trim = True)
            

    #if with_orig:
    #    axs[0, 0].set(title='Original')
    #    axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
    if col_title is not None:
        for col_idx in range(num_cols):
            axs[0, col_idx].set(title=col_title[col_idx])
            axs[0, col_idx].title.set_size(16)

    plt.tight_layout()

#diffused_curves = [[q_sample(x_start[7,:], torch.tensor([t])) for t in [0, 50, 100, 150, 199]] ,
#                   [q_sample(x_start[9,:], torch.tensor([t])) for t in [0, 50, 100, 150, 199]]]#
#plot(diffused_curves, with_histogram=True)

def plot_foward_diffusion(x_start, ts, with_histogram=True, plot_size=2, **imshow_kwargs):
    if not isinstance(x_start, list):
        # Make a 2d grid even if there's just 1 row
        x_start = [x_start]
    diffused_curves = [[x] + [ddpm1.q_sample(x, torch.tensor([t])) for t in ts] for x in x_start]
    col_title = ['t = 0 (original)'] + [f't = {t}/{ddpm1.timesteps-1}' for t in ts]

    plot(diffused_curves, with_histogram=with_histogram, plot_size=plot_size, col_title=col_title, **imshow_kwargs)


plot_foward_diffusion([x_start[0,:]], ts=[3,8, 50, 100, 150, 199], with_histogram=True, plot_size= 2.5)

# %% [markdown]
# This means that we can now define the loss function given the model as follows:

# %% [markdown]
# ## Train the model
# 

# %%
if False:
    ddpm1.train(dataloader=dataloader)



# %%
if False:
    ddpm1.write_train_checkpoint()


# %% [markdown]
# ## Inpainting (general)

# %%
gt = dataset.apply_transform(np.nan_to_num(gt1.gt_xarr.data, nan=0.0)) # data.apply_transform TODO: This is bad
gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(device)
gt = torch.from_numpy(gt).type(torch.FloatTensor).to(device)

# %% [markdown]
# ## Inpatinting from copaint

# %%
conf = config.Config(default_config_dict=
                    {
                        "respace_interpolate": False,
                            "ddim": {
                                "ddim_sigma": 0.0,
                                "schedule_params": {
                                    "ddpm_num_steps": 200,
                                    "jump_length": 10,
                                    "jump_n_sample": 2,
                                    "num_inference_steps": 200,
                                    "schedule_type": "linear",
                                    "time_travel_filter_type": "none",
                                    "use_timetravel": True
                                }
                            },
                            "optimize_xt": {
                                "coef_xt_reg": 0.0001,
                                "coef_xt_reg_decay": 1.01,
                                "filter_xT": False,
                                "lr_xt": 0.02,
                                "lr_xt_decay": 1.012,
                                "mid_interval_num": 1,
                                "num_iteration_optimize_xt": 2,
                                "optimize_before_time_travel": True,
                                "optimize_xt": True,
                                "use_adaptive_lr_xt": True,
                                "use_smart_lr_xt_decay": True
                            
                            },
                        "debug":False
                    },  use_argparse=False)

conf = epiframework.copaint_config_library(timesteps=ddpm1.timesteps)["celebahq_noTT"]
sampler = O_DDIMSampler(use_timesteps=np.arange(ddpm1.timesteps), 
                conf=conf,
                betas=ddpm1.betas, 
                model_mean_type=None,
                model_var_type=None,
                loss_type=None)


# %%
a = sampler.p_sample_loop(model_fn=ddpm1.model, 
                     shape=(batch_size, channels, image_size, image_size),
                     conf=conf,
                     model_kwargs={"gt": gt.repeat(batch_size, 1, 1, 1),
                                    "gt_keep_mask":gt_keep_mask.repeat(batch_size, 1, 1, 1),
                                    "mymodel":True, 
                                }
                     )
fluforecasts = np.array(a['sample'].cpu())

# %% [markdown]
# ## Let's add repainting from the RePaint paper
# 
# Dimensions are chanel, time, place

# %%
#   inpaint1 = inpaint.REpaint(ddpm=ddpm1, gt=gt, gt_keep_mask=gt_keep_mask, resampling_steps=2)
#   n_samples = batch_size
#   all_samples = []
#   for i in range(max(n_samples//batch_size,1)):
#       samples = inpaint1.sample_paint()
#       all_samples.append(samples)
#   
#   fluforecasts = -1*np.ones((batch_size*max(n_samples//batch_size,1), 1, 64, 64))
#   for i in range(max(n_samples//batch_size,1)):
#       fluforecasts[i*batch_size:i*batch_size+batch_size] = all_samples[i][-1]

# %% [markdown]
# ## plot inpainted forecasts

# %%
#fluforecasts = np.load("/Users/chadi/Research/influpaint/2023-11-18_UNC_IDD-InfluPaint_forecast.npy")

# %%
fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)


# compute the national quantiles, important as sum of quantiles >> quantiles of sum
forecasts_national = fluforecasts_ti.sum(axis=-1)
forecasts_national.shape

# %%
# fig, axes = plt.subplots(1, 1, figsize=(5,3), dpi=200)
# 
# ax = axes
# for i in range(batch_size)[::2]:
#     ax.plot(gt1.gt_xarr.data[0,:gt1.inpaintfrom_idx].sum(axis=1), color='k', marker='.', ls='')
#     ax.plot(dataset.apply_transform_inv(samples[-1][i])[0].sum(axis=1) , lw=.5, marker='.', markersize=1, markerfacecolor='black', markeredgecolor='black', color="lightcoral")
#     idplots.show_tensor_image(samples[-1][i], ax = ax, multi=True,)
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


# %%
fig, axes = plt.subplots(1, 2, figsize=(12,4), dpi=100)

for iax in range(2):
  ax = axes[iax]
  for iqt in range(11):
      #print(flusight_quantile_pairs[iqt,0], flusight_quantile_pairs[iqt,1])
      ax.fill_between(np.arange(64), 
                      np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0], 
                      np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0], alpha=.1, color='darkred')
      
  ax.plot(np.arange(64), 
          np.quantile(forecasts_national, myutils.flusight_quantiles[12], axis=0)[0], color='r')
  #ax.plot(np.arange(64), fluforecasts_ti[:,0].sum(axis=-1).T,lw=0.1, alpha=.1,  color='k')
  ax.plot(gt1.gt_xarr.data[0,:gt1.inpaintfrom_idx].sum(axis=1), color='k', marker='.', ls='')
  ax.axvline(gt1.inpaintfrom_idx-1, c='k')
  if iax==0:
    ax.set_xlim(0,52)
    ax.set_ylim(bottom=0, auto=True)
  if iax==1:
    ax.set_xlim(gt1.inpaintfrom_idx-4,gt1.inpaintfrom_idx+4)
    ax.set_ylim(bottom=0, top=1500)
  
  ax.grid(visible = True)
  ax.set_title("National")
fig.tight_layout()
plt.show()

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



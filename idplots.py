import math
from inspect import isfunction
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import pandas as pd
import xarray as xr

import read_datasources, training_datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import Adam
import datetime

import myutils

def plot_to_ax(array, ax=None, place=None, multi=False, channels=1):
    c = ["firebrick", "slateblue", "orange"]
    # (3, 48, 51)
    if len(array.shape) == 4:  # it's a batch
        array = array[0, :, :, :]
    if ax is None:
        ax = plt.gca()  # get current ax

    if place is None:
        array_to_plot = array.sum(axis=2)
    else:
        array_to_plot = array[:, :, place]
    for k in range(channels):
        if multi:  # several lines will be plotted:
            ax.plot(array_to_plot[k], c=c[k], lw=0.5, alpha=0.5)
        else:
            ax.plot(array_to_plot[k], c=c[k], lw=2, alpha=1)


def show_tensor_image(inv_transd_image, ax=None, place=None, multi=False):
    plot_to_ax(inv_transd_image, ax=ax, place=place, multi=multi)


def plot_timeseries_grid(df, season_axis, value_col='value', location_col='location_code', 
                        date_col='week_enddate', title_func=None):
    """
    Plot time series data in a grid of subplots, one per location.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot
    season_axis : SeasonAxis
        SeasonAxis object for getting location names
    value_col : str
        Column name for values to plot
    location_col : str  
        Column name for location codes
    date_col : str
        Column name for dates
    title_func : callable, optional
        Function to generate subplot titles. If None, uses season_axis.get_location_name()
        
    Returns:
    --------
    fig, axes : matplotlib objects
        Figure and axes objects
    """
    # Pivot data for easier plotting
    df_piv = df.pivot(columns=location_col, values=value_col, index=date_col)
    
    # Calculate grid dimensions
    n_locations = len(df_piv.columns)
    n_cols = int(np.ceil(np.sqrt(n_locations)))
    n_rows = int(np.ceil(n_locations / n_cols))
    
    # Infer figure size based on grid dimensions
    figsize = (n_cols * 3, n_rows * 2.5)
    
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=figsize)
    
    # Handle case where there's only one subplot
    if n_locations == 1:
        axes = [axes]
    else:
        axes = axes.flat
    
    for idx, location in enumerate(df_piv.columns):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        ax.plot(df_piv[location])
        
        # Set title
        if title_func is not None:
            title = title_func(location)
        else:
            title = season_axis.get_location_name(location)
        ax.set_title(title)
        ax.grid()
    
    # Hide unused subplots
    for idx in range(n_locations, len(axes)):
        axes[idx].set_visible(False)
        
    fig.tight_layout()
    fig.autofmt_xdate()
    
    return fig, axes


def plot_season_overlap_grid(df, season_axis, value_col='value', location_col='location_code',
                           season_col='fluseason', season_week_col='season_week',
                           title_func=None, line_color='k', line_width=1.2,
                           despine=True):
    """
    Plot overlapping seasons in a grid of subplots, one per location.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to plot
    season_axis : SeasonAxis
        SeasonAxis object for getting location names
    value_col : str
        Column name for values to plot
    location_col : str
        Column name for location codes
    season_col : str
        Column name for season/year
    season_week_col : str
        Column name for season week (integer 1-53)
    title_func : callable, optional
        Function to generate subplot titles. If None, uses season_axis.get_location_name()
    line_color : str
        Color for the season lines
    line_width : float
        Width of the season lines
    despine : bool
        Whether to remove top and right spines
        
    Returns:
    --------
    fig, axes : matplotlib objects
        Figure and axes objects
    """
    # Pivot data using multi-index with integer weeks
    df_piv = df.pivot(columns=location_col, values=value_col, 
                     index=[season_col, season_week_col])
    
    # Calculate grid dimensions
    n_locations = len(df_piv.columns)
    n_cols = int(np.ceil(np.sqrt(n_locations)))
    n_rows = int(np.ceil(n_locations / n_cols))
    
    # Infer figure size based on grid dimensions
    figsize = (n_cols * 3, n_rows * 2.5)
    
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=figsize)
    
    # Handle case where there's only one subplot
    if n_locations == 1:
        axes = [axes]
    else:
        axes = axes.flat
    
    for idx, location in enumerate(df_piv.columns):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Plot each season as a separate line
        for season in df_piv.index.unique(level=season_col):
            try:
                season_data = df_piv.loc[season, location]
                if len(season_data) > 0:
                    ax.plot(season_data, c=line_color, lw=line_width)
                else:
                    print(f"Empty data for {location} and season {season}")
            except KeyError:
                # Handle missing data for this season/location combination
                continue
        
        # Set title
        if title_func is not None:
            title = title_func(location)
        elif hasattr(season_axis, 'get_location_name'):
            title = season_axis.get_location_name(location)
        else:
            title = location
        ax.set_title(title)
        
        if not despine:
            ax.grid()
        
        if despine:
            sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_locations, len(axes)):
        axes[idx].set_visible(False)
        
    fig.tight_layout()
    fig.autofmt_xdate()
    
    return fig, axes



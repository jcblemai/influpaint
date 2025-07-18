
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime
import itertools as it
import xarray as xr





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


def plot_us_grid(data, season_axis, value_col='value', location_col='location_code', 
                 date_col='week_enddate', colors=None, line_width=2.5, 
                 alpha_fill=0.2, quantile_cols=None, title_suffix='', 
                 date_range=None, y_ticks=None, date_format='%Y', sample_idx=None, 
                 multi_line=False, show_us_summary=True, sharey=True):
    """
    Plot time series data in a US state grid layout.
    
    Parameters:
    -----------
    data : pd.DataFrame or xr.DataArray
        DataFrame containing the data to plot, or xarray with dimensions:
        - (feature, date, place) for general use
        - (sample, feature, season_week, place) for create_datasets.py format
    season_axis : SeasonAxis
        SeasonAxis object containing location information from influpaint_locations.csv
    value_col : str
        Column name for values to plot (used for DataFrame input)
    location_col : str
        Column name for location codes (used for DataFrame input)
    date_col : str
        Column name for dates (used for DataFrame input)
    colors : list, optional
        List of colors to use for different series
    line_width : float
        Width of the main line (for single line) or base width for multiple lines
    alpha_fill : float
        Alpha for fill_between areas
    quantile_cols : dict, optional
        Dictionary with quantile columns for confidence intervals
        Format: {'q025': 'col_name', 'q975': 'col_name', 'q25': 'col_name', 'q75': 'col_name'}
    title_suffix : str
        Suffix to add to subplot titles
    date_range : tuple, optional
        Tuple of (start_date, end_date) for x-axis limits
    y_ticks : list, optional
        List of y-axis tick positions
    date_format : str
        Format string for date labels
    sample_idx : int or list, optional
        For create_datasets.py format: sample index(es) to use. If None, uses all samples.
        For general xarray format: feature index(es) to use. If None, uses all features.
        If int, uses single item. If list, uses multiple items for multi-line plot.
    multi_line : bool
        If True, plots multiple lines with lighter weight and transparency
    show_us_summary : bool
        If True, shows a summary plot for all US locations combined
    sharey : bool
        If True, shares y-axis across all subplots
        
    Returns:
    --------
    fig, axes : matplotlib objects
        Figure and axes objects
    """
    # US state grid layout
    state_posx = {
        'ak': (0, 0), 'me': (0, 10),
        'vt': (1, 9), 'nh': (1, 10),
        'wa': (2, 0), 'id': (2, 1), 'mt': (2, 2), 'nd': (2, 3), 'mn': (2, 4),
        'il': (2, 5), 'wi': (2, 6), 'mi': (2, 7), 'ny': (2, 8), 'ri': (2, 9), 'ma': (2, 10),
        'or': (3, 0), 'nv': (3, 1), 'wy': (3, 2), 'sd': (3, 3), 'ia': (3, 4), 
        'in': (3, 5), 'oh': (3, 6), 'pa': (3, 7), 'nj': (3, 8), 'ct': (3, 9),
        'ca': (4, 0), 'ut': (4, 1), 'co': (4, 2), 'ne': (4, 3), 'mo': (4, 4),
        'ky': (4, 5), 'wv': (4, 6), 'va': (4, 7), 'md': (4, 8), 'de': (4, 9),
        'az': (5, 1), 'nm': (5, 2), 'ks': (5, 3), 'ar': (5, 4), 'tn': (5, 5), 
        'nc': (5, 6), 'sc': (5, 7), 'dc': (5, 8), 'ok': (6, 3), 'la': (6, 4), 
        'ms': (6, 5), 'al': (6, 6), 'ga': (6, 7), 'hi': (6, 0), 'tx': (7, 3), 'fl': (7, 7)
    }
    
    # Detect input type and convert to common format
    if isinstance(data, xr.DataArray):
        # Handle xarray input
        xarr = data
        
        # Check if this is the create_datasets.py format (sample, feature, season_week, place)
        if 'season_week' in xarr.dims and 'sample' in xarr.dims:
            # This is the create_datasets.py format
            # Only use the first 53 weeks (real data, not padded)
            real_weeks = min(53, len(xarr.season_week))
            dates = np.arange(1, real_weeks + 1)  # season weeks 1-53
            places = xarr.place.values
            
            # Determine which samples to use
            if sample_idx is None:
                # Use all samples
                sample_indices = list(range(len(xarr.sample)))
                multi_line = True
            elif isinstance(sample_idx, int):
                # Use single sample
                sample_indices = [sample_idx]
            elif isinstance(sample_idx, list):
                # Use specified samples
                sample_indices = sample_idx
                multi_line = True
            else:
                raise ValueError("sample_idx must be None, int, or list")
            
            # Validate sample indices
            for idx in sample_indices:
                if idx >= len(xarr.sample):
                    raise ValueError(f"sample_idx {idx} out of range for {len(xarr.sample)} samples")
            
            # Create DataFrame-like structure for plotting
            if len(sample_indices) == 1:
                # Single sample case
                plot_data = {}
                for i, place in enumerate(places):
                    if isinstance(place, str) and place:  # Skip empty strings
                        # Use first feature, only real weeks
                        plot_data[place] = xarr[sample_indices[0], 0, :real_weeks, i].values
                df_plot = pd.DataFrame(plot_data, index=dates)
            else:
                # Multiple samples case - store as dict with 2D arrays
                plot_data = {}
                for i, place in enumerate(places):
                    if isinstance(place, str) and place:  # Skip empty strings
                        # Extract data for all samples for this place: (season_week, sample)
                        # Use first feature, only real weeks
                        place_data = xarr[sample_indices, 0, :real_weeks, i].values.T
                        plot_data[place] = place_data
                df_plot = pd.Series(plot_data, name='data')
                
        else:
            # This is the original format (feature, date, place)
            dates = pd.to_datetime(xarr.date.values)
            places = xarr.place.values
            
            # Determine which features to use
            if sample_idx is None:
                # Use all features
                feature_indices = list(range(len(xarr.feature)))
                multi_line = True
            elif isinstance(sample_idx, int):
                # Use single feature
                feature_indices = [sample_idx]
            elif isinstance(sample_idx, list):
                # Use specified features
                feature_indices = sample_idx
                multi_line = True
            else:
                raise ValueError("sample_idx must be None, int, or list")
            
            # Validate feature indices
            for idx in feature_indices:
                if idx >= len(xarr.feature):
                    raise ValueError(f"sample_idx {idx} out of range for {len(xarr.feature)} features")
            
            # Create DataFrame-like structure for plotting
            if len(feature_indices) == 1:
                # Single feature case
                plot_data = {}
                for i, place in enumerate(places):
                    if isinstance(place, str):
                        plot_data[place] = xarr[feature_indices[0], :, i].values
                df_plot = pd.DataFrame(plot_data, index=dates)
            else:
                # Multiple features case - store as dict with 2D arrays
                plot_data = {}
                for i, place in enumerate(places):
                    if isinstance(place, str):
                        # Extract data for all features for this place: (date, feature)
                        place_data = xarr[feature_indices, :, i].values.T
                        plot_data[place] = place_data
                df_plot = pd.Series(plot_data, name='data')
        
    elif isinstance(data, pd.DataFrame):
        # Handle DataFrame input
        df = data.copy()
        
        # Convert dates if needed
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Pivot DataFrame for easier plotting
        df_plot = df.pivot(columns=location_col, values=value_col, index=date_col)
        
        # Handle quantile columns if provided
        if quantile_cols:
            quantile_data = {}
            for q_name, q_col in quantile_cols.items():
                quantile_data[q_name] = df.pivot(columns=location_col, values=q_col, index=date_col)
    else:
        raise ValueError("Data must be either a pandas DataFrame or xarray DataArray")
    
    # Get state names from season_axis
    state_names = {}
    # season_axis.locations is always a list, locations_df is the DataFrame
    for _, row in season_axis.locations_df.iterrows():
        if row['abbreviation'].lower() in state_posx:
            state_names[row['abbreviation'].lower()] = row['location_name']
    
    # Grid dimensions
    w = 2.95 - 0.4
    h = 2.25 - 0.4
    ncols = 11
    nrows = 8
    
    # Create figure
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*w, nrows*h), dpi=200, sharey=sharey)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Get all grid positions
    tups = list(it.product(range(nrows), range(ncols)))
    
    # Default colors
    if colors is None:
        colors = ['teal', 'goldenrod', 'firebrick']
    
    # Plot each state
    for st, po in state_posx.items():
        st_upper = st.upper()
        stlab = state_names.get(st, st.upper())
        
        # Map state abbreviation to location code if needed
        location_key = st_upper
        if isinstance(data, xr.DataArray) and 'season_week' in data.dims:
            # For create_datasets.py format, need to map abbreviation to location_code
            location_key = None
            for _, row in season_axis.locations_df.iterrows():
                if row['abbreviation'].upper() == st_upper:
                    location_key = row['location_code']
                    break
            if location_key is None:
                continue
        
        # Check if state data exists
        if isinstance(df_plot, pd.DataFrame):
            # DataFrame case (single sample or DataFrame input)
            if location_key in df_plot.columns:
                state_series = df_plot[location_key].dropna()
                
                if not state_series.empty:
                    dates_plot = state_series.index
                    values_plot = state_series.values
                else:
                    continue
            else:
                continue
        else:
            # Series case (multiple samples from xarray)
            if location_key in df_plot.index:
                state_data = df_plot[location_key]
                
                if state_data is not None and len(state_data) > 0:
                    dates_plot = dates  # Use original dates
                    values_plot = state_data  # This is a 2D array (date, sample)
                else:
                    continue
            else:
                continue
        
        # Determine if we have multiple lines (check if values is 2D or multi_line is True)
        if multi_line and len(values_plot.shape) > 1 and values_plot.shape[1] > 1:
            # Multiple lines - plot each with lighter weight
            multi_line_width = line_width * 0.5
            multi_alpha = 0.5
            
            for i in range(values_plot.shape[1]):
                color_idx = i % len(colors)
                ax[po].plot(dates_plot, values_plot[:, i], 
                           color=colors[color_idx], linewidth=multi_line_width, 
                           alpha=multi_alpha)
        elif multi_line:
            # Multi-line mode but only one line - use lighter style
            ax[po].plot(dates_plot, values_plot, 
                       color=colors[0], linewidth=line_width * 0.5, 
                       alpha=0.5)
        else:
            # Single line - plot with full weight
            ax[po].plot(dates_plot, values_plot, 
                       color=colors[0], linewidth=line_width)
        
        # Plot quantile fills if provided (DataFrame input only)
        if isinstance(data, pd.DataFrame) and quantile_cols:
            if 'q025' in quantile_cols and 'q975' in quantile_cols:
                q025_data = quantile_data['q025'][st_upper].dropna()
                q975_data = quantile_data['q975'][st_upper].dropna()
                if not q025_data.empty and not q975_data.empty:
                    ax[po].fill_between(q025_data.index, q025_data.values, q975_data.values, 
                                      color=colors[0], alpha=alpha_fill, linewidth=0)
            
            if 'q25' in quantile_cols and 'q75' in quantile_cols:
                q25_data = quantile_data['q25'][st_upper].dropna()
                q75_data = quantile_data['q75'][st_upper].dropna()
                if not q25_data.empty and not q75_data.empty:
                    ax[po].fill_between(q25_data.index, q25_data.values, q75_data.values, 
                                      color=colors[0], alpha=alpha_fill, linewidth=0)
        
        # Always show state names - use 2-letter abbreviations
        state_abbrev = st_upper
        
        title_text = state_abbrev + title_suffix
        t = ax[po].text(0.04, 0.96, title_text, fontsize='xx-large', va='top', ha='left',
                       color='k', transform=ax[po].transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))
    
    # Let matplotlib handle y-ticks automatically unless specified
    auto_yticks = y_ticks is None
    
    # Configure axes
    leftmost = [0, 9, 0, 0, 0, 1, 0, 3]
    bottommost = [6, 5, 5, 7, 6, 6, 6, 7, 5, 4, 2]
    
    for tup in tups:
        if tup not in state_posx.values():
            ax[tup].set_axis_off()
            sns.despine(ax=ax[tup], trim=True, offset=10)
        else:
            lefty = (tup[1] == leftmost[tup[0]])
            bottomy = (tup[0] == bottommost[tup[1]])
            
            # Set date range
            if date_range:
                ax[tup].set_xlim(date_range[0], date_range[1])
            
            # Set up date formatting
            if isinstance(data, xr.DataArray) and 'season_week' in data.dims:
                # Season week format (1-53) - remove W prefix
                season_ticks = [1, 13, 26, 39, 53]  # Roughly quarterly
                ax[tup].xaxis.set_ticks(season_ticks)
                ax[tup].xaxis.set_ticklabels([str(w) for w in season_ticks])
            elif date_format == '%Y':
                dates = [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1), datetime.date(2022, 1, 1)]
                ax[tup].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax[tup].xaxis.set_ticks(dates)
            else:
                ax[tup].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            
            # Set y-axis ticks if provided
            if not auto_yticks:
                ax[tup].yaxis.set_ticks(y_ticks)
                ax[tup].yaxis.set_ticklabels([str(tick) for tick in y_ticks])
            
            ax[tup].tick_params(labelsize="x-large", direction="in", width=2)
            sns.despine(ax=ax[tup])
            
            # Handle y-axis labels based on sharey setting
            if sharey:
                # With sharey=True, only show y-labels on leftmost plots
                if not lefty:
                    ax[tup].yaxis.set_ticklabels([])
            # With sharey=False, show y-labels on all plots (matplotlib default)
            
            # Handle x-axis labels - only show on bottom plots
            if not bottomy:
                # Hide x-axis labels for non-bottom plots
                if not (tup == (0, 0) or tup == (4, 0)):  # Keep labels for Alaska and California
                    ax[tup].xaxis.set_ticklabels([])
    
    # Add US summary plot if requested
    if show_us_summary:
        # Create space for US summary plot (remove some empty plots)
        gs = ax[1, 8].get_gridspec()
        for a in ax[0:1, 2:8].flatten():
            a.remove()
        axbig = fig.add_subplot(gs[0:1, 3:7])
        
        # Calculate US summary data using the same approach as individual plots
        if isinstance(data, xr.DataArray) and 'season_week' in data.dims:
            # For xarray data, use the same data structure as individual plots
            real_weeks = min(53, len(data.season_week))
            
            if sample_idx is None:
                sample_indices = list(range(len(data.sample)))
            elif isinstance(sample_idx, int):
                sample_indices = [sample_idx]
            else:
                sample_indices = sample_idx
            
            x_vals = np.arange(1, real_weeks + 1)
            
            if multi_line and len(sample_indices) > 1:
                # Plot individual samples lightly - sum across all states for each sample
                # Use same color indexing as individual state plots
                for i, sample in enumerate(sample_indices):
                    sample_sum = []
                    for week in range(real_weeks):
                        week_sum = 0
                        for loc_idx in range(len(season_axis.locations)):
                            week_sum += data[sample, 0, week, loc_idx].values
                        sample_sum.append(week_sum)
                    # Match the color indexing from individual state plots (line 464)
                    color_idx = i % len(colors)
                    axbig.plot(x_vals, sample_sum, color=colors[color_idx], linewidth=line_width * 0.5, alpha=0.5)
            else:
                # Plot single line - sum across all states
                us_sum = []
                for week in range(real_weeks):
                    week_sum = 0
                    for loc_idx in range(len(season_axis.locations)):
                        for sample in sample_indices:
                            week_sum += data[sample, 0, week, loc_idx].values
                    us_sum.append(week_sum / len(sample_indices))  # Average across samples
                axbig.plot(x_vals, us_sum, color=colors[0], linewidth=line_width * 1.5)
            
            # Format x-axis for season weeks
            season_ticks = [1, 13, 26, 39, 53]
            axbig.xaxis.set_ticks(season_ticks)
            axbig.xaxis.set_ticklabels([str(w) for w in season_ticks])
            
        elif isinstance(data, pd.DataFrame):
            # For DataFrame data, calculate US sum (not average)
            us_summary = data.groupby(date_col)[value_col].sum()
            axbig.plot(us_summary.index, us_summary.values, color=colors[0], linewidth=line_width * 1.5)
        
        # Style the US summary plot
        axbig.text(0.04, 0.96, 'United States', fontsize='xx-large', va='top', ha='left',
                  color='black', fontweight='bold', transform=axbig.transAxes)
        
        # Set y-axis ticks for summary plot
        if not auto_yticks:
            axbig.yaxis.set_ticks(y_ticks)
            axbig.yaxis.set_ticklabels([str(tick) for tick in y_ticks])
        
        axbig.tick_params(labelsize="x-large", direction="in", width=2)
        axbig.grid(linewidth=1.5, color='w', alpha=0.9)
        sns.despine(ax=axbig)
    
    return fig, ax

def plot_few_sample(samples, dataset, 
                    indices=[1, 2, 3, 4, 5],  
                    season_labels=None,
                    save_path=None, 
                    title=False):
    """
    Figure 1 displays representative, unconditional seasons with heatmaps and curves.
    Parameters:
    - indices: list of sample indices to display (determines number of subplots)
    - season_labels: list of labels for each season (auto-generated if None)
    """
    
    n_seasons = len(indices)
    
    # Auto-generate labels if not provided
    if season_labels is None:
        default_labels = ['Canonical', 'Bimodal', 'Mild/Dispersed', 'Early-Onset', 'Spatially Heterogeneous', 
                         'Late Peak', 'Multi-Peak', 'Extended', 'Compressed', 'Regional']
        season_labels = [f'({chr(97+i)}) {default_labels[i % len(default_labels)]}' for i in range(n_seasons)]
    
    # Set paper-ready style
    plt.style.use('default')
    #plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Adjust figure size based on number of seasons
    fig_width = min(20, max(12, n_seasons * 3.6))
    fig = plt.figure(figsize=(fig_width, 10), dpi=300)
    
    # Create subplot grid: 2 rows (heatmap, curve), n_seasons columns
    gs = fig.add_gridspec(2, n_seasons, hspace=0.35, wspace=0.25, height_ratios=[2.2, 1])
    
    
    # Find global min/max for consistent heatmap scaling
    all_data = []
    for idx in indices:
        sample_data = dataset.apply_transform_inv(samples[-1][idx])
        # samples[-1] shape is (batch_size, 64, 64), we need (52, 51)
        sample_2d = sample_data[0][:52, :51]  # Take first 52 weeks, 51 locations
        all_data.append(sample_2d)
    vmin = min([data.min() for data in all_data])
    vmax = max([data.max() for data in all_data])
    
    for i, (idx, label) in enumerate(zip(indices, season_labels)):
        # Get transformed data for this sample
        sample_data = dataset.apply_transform_inv(samples[-1][idx])
        sample_2d = sample_data[0][:52, :51]  # Take first 52 weeks, 51 locations
        
        # Top row: Heatmaps
        ax_heat = fig.add_subplot(gs[0, i])
        
        # Create heatmap with consistent scaling
        im = ax_heat.imshow(sample_2d, aspect='auto', cmap='Reds', origin='upper', 
                           vmin=vmin, vmax=vmax, interpolation='nearest')
        
        ax_heat.set_title(label, fontsize=12, fontweight='bold', pad=12)
        ax_heat.set_xlabel('Location', fontsize=11, labelpad=8)
        if i == 0:
            ax_heat.set_ylabel('Epiweek', fontsize=11, labelpad=8)
        
        # Set ticks with better spacing
        ax_heat.set_xticks([0, 12, 25, 38, 50])
        ax_heat.set_xticklabels(['1', '13', '26', '39', '51'])
        ax_heat.set_yticks([0, 13, 26, 39, 51])
        ax_heat.set_yticklabels(['1', '14', '27', '40', '52'])
        
        # Style heatmap
        ax_heat.tick_params(axis='both', which='major', labelsize=9, length=3)
        
        # Add shared colorbar at the right
        if i == n_seasons - 1:  # Last subplot
            cbar_ax = fig.add_axes((0.92, 0.55, 0.015, 0.35))
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label('Incidence', fontsize=11, labelpad=15)
            cbar.ax.tick_params(labelsize=9)
        
        # Bottom row: National curves (sum across all locations)
        ax_curve = fig.add_subplot(gs[1, i])
        
        national_curve = sample_2d.sum(axis=1)  # Sum across locations for each week
        weeks = np.arange(1, 53)
        
        # Enhanced curve styling
        ax_curve.plot(weeks, national_curve, color='#C41E3A', linewidth=2.2, 
                     marker='o', markersize=2.5, markerfacecolor='#C41E3A', 
                     markeredgecolor='white', markeredgewidth=0.3, alpha=0.9)
        ax_curve.fill_between(weeks, 0, national_curve, alpha=0.25, color='#C41E3A')
        
        ax_curve.set_xlabel('Epiweek', fontsize=11, labelpad=8)
        if i == 0:
            ax_curve.set_ylabel('National Incidence', fontsize=11, labelpad=8)
        
        ax_curve.set_xlim(1, 52)
        ax_curve.set_ylim(bottom=0)
        
        # Enhanced grid
        ax_curve.grid(True, alpha=0.3, linewidth=0.5, linestyle='-')
        ax_curve.set_axisbelow(True)
        
        # Professional axis styling
        ax_curve.spines['top'].set_visible(False)
        ax_curve.spines['right'].set_visible(False)
        ax_curve.spines['left'].set_color('#666666')
        ax_curve.spines['bottom'].set_color('#666666')
        ax_curve.spines['left'].set_linewidth(0.8)
        ax_curve.spines['bottom'].set_linewidth(0.8)
        
        # Set major ticks
        ax_curve.set_xticks([1, 13, 26, 39, 52])
        ax_curve.tick_params(axis='both', which='major', labelsize=9, length=3, 
                           colors='#333333')
        
        # Remove y-tick labels for all but first subplot
        if i > 0:
            ax_curve.set_yticklabels([])
            ax_curve.tick_params(axis='y', which='major', left=False)
    
    # Main title with better positioning
    if title:
        title = f'Figure 1: Representative Unconditional Influenza Seasons ({n_seasons} samples)'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
    
    # Final layout adjustment
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.90)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png')
        # Also save as PDF for publication
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='pdf')
    
    plt.show()
    
    return fig


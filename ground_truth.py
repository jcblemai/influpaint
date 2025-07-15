import math
from inspect import isfunction
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from season_axis import SeasonAxis


import numpy as np
import pandas as pd
import xarray as xr

import datetime

import myutils, read_datasources


def pad_dataframe(df, season_setup):
    # Make sure gt_df and gt_df_final have values (even if NaN) for all dates in the season
    date_range = pd.date_range(start=df.week_enddate.min(), periods=52, freq='W-SAT')
    locations = df.location_code.unique()
    # Create expanded dataframe with all combinations of dates and locations
    expanded_df = pd.DataFrame([(d, l) for d in date_range for l in locations],
                            columns=['week_enddate', 'location_code'])

    # Calculate the season columns
    expanded_df['fluseason'] = expanded_df.week_enddate.apply(season_setup.get_fluseason_year)
    expanded_df['fluseason_fraction'] = expanded_df.week_enddate.apply(season_setup.get_fluseason_fraction)
    expanded_df['season_week'] = expanded_df.week_enddate.apply(season_setup.get_season_week)

    # Merge with original data to get values where they exist
    padded_df = expanded_df.merge(
        df[['week_enddate', 'location_code', 'value']], 
        on=['week_enddate', 'location_code'], 
        how='left'
    )
    return padded_df


class GroundTruth():
    def __init__(self, season_first_year: str, 
                data_date: datetime.datetime, 
                mask_date: datetime.datetime, 
                from_final_data:bool=False, 
                channels=1, 
                image_size=64, 
                nogit=False, 
                payload=None, 
                payload_season_first_year=None,
                dataset_coords: xr.core.coordinates.DataArrayCoordinates=None): 
        self.season_first_year = season_first_year
        self.data_date = data_date
        self.mask_date = mask_date
        self.channels = channels
        self.image_size=image_size


        if not nogit: self.git_checkout_data_rev(target_date=None)

        self.season_setup = SeasonAxis.from_flusight(season_first_year=self.season_first_year, remove_territories=True, remove_us=True)

        flusight = read_datasources.get_from_epidata(dataset=f"flusight{self.season_first_year}", season_setup=self.season_setup, write=False)
        gt_df_final = flusight[flusight["fluseason"] == int(self.season_first_year)]

        if from_final_data:
            gt_df = gt_df_final.copy()
        else:
            if not nogit: self.git_checkout_data_rev(target_date=data_date)
            flusight = read_datasources.get_from_epidata(dataset=f"flusight{self.season_first_year}", season_setup=self.season_setup, write=False)
            gt_df = flusight[flusight["fluseason"] == int(self.season_first_year)]   
            if not nogit: self.git_checkout_data_rev(target_date=None)

        
        self.gt_df = gt_df[gt_df["location_code"].isin(self.season_setup.locations)]
        self.gt_df_final = gt_df_final[gt_df_final["location_code"].isin(self.season_setup.locations)]
        
        # generates past data
        self.previous_data = [read_datasources.get_from_epidata(dataset=f"flusight2024", season_setup=self.season_setup, write=False),
                            read_datasources.get_from_epidata(dataset=f"flusight2024", season_setup=self.season_setup, write=False)]
        


        self.gt_df = pad_dataframe(self.gt_df, self.season_setup)
        self.gt_df_final = pad_dataframe(self.gt_df_final, self.season_setup)

        last_non_nan_datadate = self.gt_df.week_enddate[self.gt_df.value.notna()].max().to_pydatetime()
        # If the last data_point is not in the last week, we need to update the mask to be in the week after the last data point
        if self.mask_date > last_non_nan_datadate + datetime.timedelta(days=7):
            self.mask_date = last_non_nan_datadate + datetime.timedelta(days=2)
            print(f" WARNING: mask_date is after last non-NaN data date, setting mask_date to {self.mask_date}")


        if payload is not None:
            if payload_season_first_year is None:
                payload_season_first_year = season_first_year
            import dataset_mixer
            payload = season_setup.add_season_columns(payload, self.season_setup)
            this_payload = payload[payload["fluseason"] == int(payload_season_first_year)]
            self.gt_df = pd.concat([self.gt_df, this_payload], ignore_index=True)
            self.gt_df_final = pd.concat([self.gt_df_final, this_payload], ignore_index=True)
            self.previous_data.append(payload)
            location_codes = self.gt_df.location_code.unique()
            new_locations = pd.DataFrame({"location_code": sorted(location_codes)})
            # Ensure location_code is of type string
            new_locations['location_code'] = new_locations['location_code'].astype(str)
            # Merge with season_setup.locations_df to get the location names
            new_locations = new_locations.merge(self.season_setup.locations_df, 
                                                on='location_code',
                                                how='left')

            # Fill missing location names with the location code
            new_locations['location_name'] = new_locations['location_name'].fillna(new_locations['location_code'])
            new_locations = new_locations[['location_code', 'location_name']]
            self.season_setup.update_locations(new_locations)

        if dataset_coords is not None:
            # change the flusetup locations to be in the same order as flu_payload_array.coords["place"]
            self.season_setup.reorder_locations(list(dataset_coords["place"].values))

        self.previous_data = pd.concat(self.previous_data, ignore_index=True).drop_duplicates()

        self.gt_xarr = read_datasources.dataframe_to_xarray(self.gt_df, season_setup=self.season_setup, 
            xarray_name = "gt_flusight_incidHosp", 
            xarrax_features = "incidHosp")
        
        self.gt_final_xarr = read_datasources.dataframe_to_xarray(self.gt_df_final, season_setup=self.season_setup, 
            xarray_name = "gt_flusight_incidHos_final", 
            xarrax_features = "incidHosp")

        # Find the largest index of the data dates that are before the mask date
        dates = pd.to_datetime(self.gt_xarr.coords['date'].values)
        self.inpaintfrom_idx = sum(dates < self.mask_date)

        self.gt_keep_mask = np.ones((channels,image_size,image_size))
        self.gt_keep_mask[:,self.inpaintfrom_idx:,:] = 0
        
        print(f"Masking, >> {self.inpaintfrom_idx} weeks already in data, inpainting the next ones")

    
    def git_checkout_data_rev(self, target_date=None):
        import pygit2
        if self.season_first_year == "2023":
            repo_path = "Flusight/2023-2024/FluSight-forecast-hub-official/"
            main_branch = "main"
        elif self.season_first_year == "2022":
            repo_path = "Flusight/2022-2023/FluSight-forecast-hub-official/"
            main_branch = "master"
        elif self.season_first_year == "2024":
            repo_path = "Flusight/2024-2025/FluSight-forecast-hub-official/"
            main_branch = "main"
        print(repo_path)

        # Open the existing repository
        repo = pygit2.Repository(repo_path)

        if target_date is not None:
            # Find the commit closest to the target date
            closest_commit = None
            for commit in repo.walk(repo.head.target, pygit2.GIT_SORT_TIME):
                if commit.commit_time <= target_date.timestamp():
                    closest_commit = commit
                    break

            # Check out the commit
            if closest_commit:
                repo.checkout_tree(closest_commit.tree)
                repo.set_head(closest_commit.id)
                print(f"Checked out commit on {target_date} (SHA: {closest_commit.id}, {commit.commit_time}) for repo {repo_path}")
            else:
                print("ERROR: No commit found for the specified date on repo {repo_path}.")
        else:
            repo.checkout("refs/heads/" + main_branch)      
            print(f"Restored git repo {repo_path}")

    def plot(self):
        fig, axes = plt.subplots(8, 8, sharex=True, figsize=(14,16))
        gt_piv  = self.gt_df.pivot(index = "week_enddate", columns='location_code', values='value')
        gt_piv_final = self.gt_df_final.pivot(index = "week_enddate", columns='location_code', values='value')
        ax = axes.flat[0]
        ax.plot(gt_piv[self.season_setup.locations].sum(axis=1), color="black", linewidth=2,label="datadate")
        ax.plot(gt_piv_final[self.season_setup.locations].sum(axis=1), lw=1, color='r', ls='-.', label="final")
        ax.legend()
        ax.set_ylim(0)
        ax.set_title("US")
        for idx, pl in enumerate(gt_piv.columns):
            ax = axes.flat[idx+1]
            ax.plot(gt_piv[pl], lw=2, color='k')
            ax.plot(gt_piv_final[pl], lw=1, color='r', ls='-.')
            na_mask = gt_piv.isna()
            ax.plot(gt_piv[na_mask].index,
                    gt_piv[na_mask],
                    marker='o', 
                    color="pink",
                    fillstyle='full', 
                    markeredgecolor='red', 
                    markersize=5,
                    markeredgewidth=1)
            ax.set_title(self.season_setup.get_location_name(pl))
            #ax.grid()
            ax.set_ylim(0)
            ax.set_xlim(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=365))
            #ax.set_xticks(season_setup.get_dates(52).resample("M"))
            #ax.plot(pd.date_range(season_setup.fluseason_startdate, season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT"), data.flu_dyn[-50:,0,:,idx].T, c='r', lw=.5, alpha=.2)
        fig.tight_layout()
        fig.autofmt_xdate()

    def plot_mask(self):
        # check that it stitch
        fig, axes = plt.subplots(1, 4, figsize=(8,8), dpi=200, sharex=True, sharey=True)
        import matplotlib as mpl
        cmap_greys = mpl.colormaps.get_cmap('Greys')
        cmap_rainbow = mpl.colormaps.get_cmap("rainbow")
        cmap_greys.set_bad(color='red')
        cmap_rainbow.set_bad(color='red')
        axes[0].imshow(self.gt_xarr.data[0], cmap=cmap_greys)
        axes[0].set_title("Current data rev", fontsize=8)

        axes[1].imshow(self.gt_keep_mask[0], alpha=.3, cmap = cmap_rainbow)
        axes[1].set_title("Inpainting mask", fontsize=8)
        


        axes[2].imshow(self.gt_xarr.data[0], cmap=cmap_greys)
        axes[2].imshow(self.gt_keep_mask[0], alpha=.3, cmap = cmap_rainbow)
        axes[3].set_title("Current data rev", fontsize=8)

        axes[3].imshow(self.gt_final_xarr.data[0], cmap=cmap_greys)
        axes[3].imshow(self.gt_keep_mask[0], alpha=.3, cmap = cmap_rainbow)
        axes[3].set_title("Final data", fontsize=8)

    def export_forecasts(self, fluforecasts_ti, forecasts_national, directory=".", prefix="", forecast_date=None, save_plot=True, nochecks=False):
        forecast_date_str=str(forecast_date)
        if forecast_date == None:
            forecast_date = self.mask_date

        target_dates = pd.date_range(forecast_date, forecast_date + datetime.timedelta(days=4*7), freq="W-SAT")

        target_dict= dict(zip(
            target_dates, 
            [f"{n} wk ahead inc flu hosp" for n in range(1,5)]))

        print(target_dates)
        #pd.DataFrame(colums=["forecast_date","target_end_date","location","type","quantile","value","target"])
        df_list=[]
        for qt in myutils.flusight_quantiles:
            a =  pd.DataFrame(np.quantile(fluforecasts_ti[:,:,:,:len(self.season_setup.locations)], qt, axis=0)[0], 
                    columns= self.season_setup.locations, index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]
            #a["US"] = a.sum(axis=1)
            a["US"] = pd.DataFrame(np.quantile(forecasts_national, qt, axis=0)[0],
                    columns= ["US"], index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]

            a = a.reset_index().rename(columns={'index': 'target_end_date'})
            a = pd.melt(a,id_vars="target_end_date",var_name="location")
            a["quantile"] = '{:<.3f}'.format(qt)
            
            df_list.append(a)

        df = pd.concat(df_list)
        df["forecast_date"] = forecast_date_str
        df["type"] = "quantile"
        df["target"] = df["target_end_date"].map(target_dict)
        df = df[["forecast_date","target_end_date","location","type","quantile","value","target"]]
        df

        for col in df.columns:
            print(col)
            print(df[col].unique())

        if not nochecks:
            assert sum(df["value"]<0) == 0
            assert sum(df["value"].isna()) == 0

        # check for Error when validating format: Entries in `value` must be non-decreasing as quantiles increase:
        for tg in target_dates:
            old_vals = np.zeros(len(self.season_setup.locations)+1)
            for dfd in df_list:  # very important to not call this df: it overwrites in namesapce the exported df
                new_vals = dfd[dfd["target_end_date"]==tg]["value"].to_numpy()
                if not (new_vals-old_vals >= 0).all():
                    print(f""" !!!! failed for {dfd["quantile"].unique()} on date {tg}""")
                    print((new_vals-old_vals).max())
                    for n, o, p in zip(new_vals, old_vals, dfd.location.unique()):
                        if "US" not in p:
                            p=p+self.season_setup.get_location_name(p)
                        print((n-o>0),p, n, o)
                else:
                    pass
                    #print(f"""ok for {dfd["quantile"].unique()}, {tg}""")
                old_vals = new_vals

        df.to_csv(f"{directory}/{prefix}-{forecast_date_str}.csv", index=False)

        if save_plot:
            self.plot_forecasts(fluforecasts_ti, forecasts_national, directory=directory, prefix=prefix, forecast_date=forecast_date)
        
    def plot_forecasts(self, fluforecasts_ti, forecasts_national, directory=".", prefix="", forecast_date=None):
        forecast_date_str=str(forecast_date)
        if forecast_date == None:
            forecast_date = self.mask_date
        idx_now = self.inpaintfrom_idx-1
        idx_horizon = idx_now+4

        plot_specs = {"all" : {
                                "quantiles_idx":range(11),
                                "color":"lightcoral",
                                },
                        "50-95" : {
                                "quantiles_idx":[1, 6],
                                "color":"darkblue"
                                }
                    }

        color_gt = "black"
        color_past='grey'

        nplace_toplot = len(self.season_setup.locations)
        #nplace_toplot = 3 # less plots for faster iteration
        plot_past_median = False
        if plot_past_median:
            plotrange=slice(None)
        else:
            plotrange=slice(self.inpaintfrom_idx,-1)


        #if self.season_first_year == "2023" or self.season_first_year == "2024":
        #    gt2022 = GroundTruth(season_first_year="2022", 
        #                    data_date=datetime.datetime.combine(datetime.date(2023,7,15), datetime.datetime.min.time()),
        #                    mask_date=datetime.datetime.today(),
        #                    channels=self.channels,
        #                    image_size=self.image_size,
        #                    payload=pd.read_csv("custom_datasets/nc_payload_gt.csv", parse_dates=["week_enddate"]))
        #if self.season_first_year == "2024":
        #    gt2023 = GroundTruth(season_first_year="2023", 
        #        data_date=datetime.datetime.combine(datetime.date(2023,7,15), datetime.datetime.min.time()),
        #        mask_date=datetime.datetime.today(),
        #        channels=self.channels,
        #        image_size=self.image_size,
        #        payload=pd.read_csv("custom_datasets/nc_payload_gt.csv", parse_dates=["week_enddate"]))

        for plot_title, plot_spec in plot_specs.items():
            #print(f"doing {plot_title}...")
            fig, axes = plt.subplots(nplace_toplot+1, 2, figsize=(10,nplace_toplot*3.5), dpi=200)
            for iax in range(2):
                ax = axes[0][iax]
    
                x = np.arange(64)
                if iax == 0:
                    x_lims_idx = (0, 51)
                    x_lims = (pd.to_datetime(self.gt_xarr["date"][x_lims_idx[0]].values), 
                            pd.to_datetime(self.gt_xarr["date"][x_lims_idx[1]].values))
                elif iax == 1:
                    x_lims_idx = (idx_now-3, idx_horizon)
                    x_lims = (pd.to_datetime(self.gt_xarr["date"][x_lims_idx[0]].values), 
                            pd.to_datetime(self.gt_xarr["date"][x_lims_idx[1]].values))
                # US WIDE: quantiles and median, US-wide
                for iqt in plot_spec["quantiles_idx"]:
                    #print(f"up: {flusight_quantile_pairs[iqt,0]} - lo: {flusight_quantile_pairs[iqt,1]}")
                    # TODO: not exactly true that it is the sum of quantiles (sum of quantile is not quantile of sum)
                    ylo = np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0]
                    yup = np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0]
                    ax.fill_between(self.gt_xarr["date"][plotrange], 
                                    ylo[plotrange], 
                                    yup[plotrange], 
                                    alpha=.1, 
                                    color=plot_spec["color"])
    
                    # widest quantile pair is the first one. We take the up quantile of it + a few % as x_lim
                    if iqt == plot_spec["quantiles_idx"][0]:
                        if plot_past_median:
                            max_y_value = max(yup[x_lims_idx[0]:x_lims_idx[1]])
                        else:
                            max_y_value = max(yup[self.inpaintfrom_idx:x_lims_idx[1]])
                        max_y_value = max(max_y_value, self.gt_xarr.data[0,:self.inpaintfrom_idx].sum(axis=1)[x_lims_idx[0]:x_lims_idx[1]].max())
                        max_y_value = max_y_value + max_y_value*.05 # 10% more
    
                # median
                ax.plot(self.gt_xarr["date"][plotrange],
                        np.quantile(forecasts_national, myutils.flusight_quantiles[12], axis=0)[0][plotrange], color=plot_spec["color"], marker='.', label='forecast median')
    
                # ground truth
                ax.plot(self.gt_xarr["date"][:self.inpaintfrom_idx],
                        self.gt_xarr.data[0,:self.inpaintfrom_idx].sum(axis=1), color=color_gt, marker = '.', lw=.5, label='ground-truth')
                ax.plot(self.gt_xarr["date"][self.inpaintfrom_idx:],
                        self.gt_xarr.data[0,self.inpaintfrom_idx:].sum(axis=1), 
                        color='red', 
                        marker = '.', 
                        lw=.1, 
                        label='ground-truth',
                        markersize=.4)

                #if self.season_first_year == "2023" or self.season_first_year == "2024":
                #    ax.plot(gt2022.gt_xarr.data[0,:].sum(axis=1), color=color_past, ls='dashed', lw=.5, label='2022 ground-truth')
                #if self.season_first_year == "2024":
                #    ax.plot(gt2022.gt_xarr.data[0,:].sum(axis=1), color=color_past, ls='dashdot', lw=.5, label='2023 ground-truth')

                if iax==0:
                    ax.legend(fontsize=8)
    
                #ax.set_xticks(np.arange(0,53,13))


                ax.set_xlim(x_lims)
                ax.set_ylim(bottom=0, top=max_y_value)
                ax.axvline(self.gt_xarr["date"][idx_now].values, c='k', lw=1, ls='-.')
                if iax == 0:
                    ax.axvline(self.gt_xarr["date"][idx_horizon].values, c='k', lw=1, ls='-.')
                ax.set_title("National")

                sns.despine(ax = ax, trim = True, offset=4)

                # INDIVDIDUAL STATES: quantiles, median and ground-truth
                max_y_value = np.zeros(nplace_toplot)
                for iqt in plot_spec["quantiles_idx"]:
                    yup = np.quantile(fluforecasts_ti, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0]
                    ylo = np.quantile(fluforecasts_ti, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0]

                    # widest quantile pair is the first one. We take the up quantile of it + a few % as x_lim
                    if iqt == plot_spec["quantiles_idx"][0]:
                        for ipl in range(nplace_toplot):
                            if plot_past_median:
                                max_y_value[ipl] = max(ylo[x_lims_idx[0]:x_lims_idx[1], ipl])
                            else:
                                max_y_value[ipl] = max(ylo[self.inpaintfrom_idx:x_lims_idx[1], ipl])
                            #max_y_value[ipl] =  max(ylo[x_lims[:x_lims[1], ipl])
                            max_y_value[ipl] = max(max_y_value[ipl], self.gt_xarr.data[0,:self.inpaintfrom_idx, ipl][x_lims_idx[0]:x_lims_idx[1]].max())
                            max_y_value[ipl] = max_y_value[ipl] + max_y_value[ipl]*.05 # 10% more for the y_max value

                    for ipl in range(nplace_toplot):
                        ax = axes[ipl+1][iax]
                        ax.fill_between(self.gt_xarr["date"][plotrange],  (yup[:,ipl])[plotrange], (ylo[:,ipl])[plotrange], alpha=.1, color=plot_spec["color"])

                # median line and ground truth for states
                for ipl in range(nplace_toplot):
                    location_name=self.season_setup.get_location_name(self.season_setup.locations[ipl])
                    ax = axes[ipl+1][iax]
                    # median
                    ax.plot(self.gt_xarr["date"][plotrange],
                            np.quantile(fluforecasts_ti, myutils.flusight_quantiles[12], axis=0)[0,:,ipl][plotrange], color=plot_spec["color"], marker = '.', lw=.5)
                    # ground truth
                    ax.plot(self.gt_xarr["date"][:self.inpaintfrom_idx],
                            self.gt_xarr.data[0,:self.inpaintfrom_idx, ipl], color=color_gt, marker = '.', lw=.5)
                    ax.plot(self.gt_xarr["date"][self.inpaintfrom_idx:],
                            self.gt_xarr.data[0,self.inpaintfrom_idx:, ipl], color='red', marker = '.', lw=.1, markersize=.4)

                    # TODO I'm here
                    
                    this_hist_data = self.previous_data[self.previous_data["location_code"]==self.season_setup.locations[ipl]]
                    for hist_season in this_hist_data["fluseason"].unique():
                        if int(hist_season) != int(self.season_first_year):
                            hist_data = this_hist_data[this_hist_data["fluseason"]==hist_season]
                            hist_data = hist_data.pivot(index = "season_week", columns='location_code', values='value').sort_index()
                            thisthing = hist_data[self.season_setup.locations[ipl]]
                            # TODO MATCH HERE !!!!
                            ax.plot(self.gt_xarr["date"][0:len(thisthing)], thisthing, color=color_past, ls='dashed', lw=.5, label=f"{hist_season}")

                    ax.axvline(self.gt_xarr["date"][idx_now].values, c='k', lw=1, ls='-.')
                    if iax == 0:
                        ax.axvline(self.gt_xarr["date"][idx_horizon].values, c='k', lw=1, ls='-.')
                    ax.set_xlim(x_lims)
                    ax.set_ylim(bottom=0, top=max_y_value[ipl])
                    if iax==0: ax.set_ylabel("New Hosp. Admissions")
                    ax.set_title(location_name)
                    # rotate the x axis labels
                    ax.tick_params(axis='x', rotation=45)
                    #print the tick label as 12 J-22
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b-%y'))

                    sns.despine(ax = ax, trim = True, offset=4)
            fig.tight_layout()
            plt.savefig(f"{directory}/{prefix}-{forecast_date_str}-plot{plot_title}.pdf")

    def export_forecasts_2023(self, fluforecasts_ti, forecasts_national, directory=".", prefix="", forecast_date=None, save_plot=True, nochecks=False, rate_trend=True):
        forecast_date_str=str(forecast_date)
        if forecast_date == None:
            forecast_date = self.mask_date

        target_dates = pd.date_range(forecast_date, forecast_date + datetime.timedelta(days=3*7), freq="W-SAT")

        target_dict= dict(zip(
            target_dates, 
            [f"{n}" for n in range(0,4)]))

        df_list=[]
        for qt in myutils.flusight_quantiles:
            a =  pd.DataFrame(np.quantile(fluforecasts_ti[:,:,:,:len(self.season_setup.locations)], qt, axis=0)[0], 
                    columns= self.season_setup.locations, index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]
            #a["US"] = a.sum(axis=1)
            a["US"] = pd.DataFrame(np.quantile(forecasts_national, qt, axis=0)[0],
                    columns= ["US"], index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]

            a = a.reset_index().rename(columns={'index': 'target_end_date'})
            a = pd.melt(a,id_vars="target_end_date",var_name="location")
            a["output_type_id"] = "{:.3f}".format(qt).rstrip('0').rstrip('.')# " #'{:<.3f}'.format(qt)
            
            df_list.append(a)

        df = pd.concat(df_list)
        df["reference_date"] = forecast_date_str
        df["target"] = "wk inc flu hosp"
        df["horizon"] = df["target_end_date"].map(target_dict)
        df["output_type"] = "quantile"
        df = df[["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]]
        df

        for col in df.columns:
            print(col)
            print(df[col].unique())

        if not nochecks:
            assert sum(df["value"]<0) == 0
            assert sum(df["value"].isna()) == 0

        # check for Error when validating format: Entries in `value` must be non-decreasing as quantiles increase:
        for tg in target_dates:
            old_vals = np.zeros(len(self.season_setup.locations)+1)
            for dfd in df_list:  # very important to not call this df: it overwrites in namesapce the exported df
                new_vals = dfd[dfd["target_end_date"]==tg]["value"].to_numpy()
                if not (new_vals-old_vals >= 0).all():
                    print(f""" !!!! failed for {dfd["quantile"].unique()} on date {tg}""")
                    print((new_vals-old_vals).max())
                    for n, o, p in zip(new_vals, old_vals, dfd.location.unique()):
                        if "US" not in p:
                            p=p+self.season_setup.get_location_name(p)
                        print((n-o>0),p, n, o)
                else:
                    pass
                    #print(f"""ok for {dfd["quantile"].unique()}, {tg}""")
                old_vals = new_vals

#        if rate_trend:
#            df_list=[]
#            for sim_id in np.arange(fluforecasts_ti.shape[0]):
#            #for qt in myutils.flusight_quantiles:
#                a =  pd.DataFrame(fluforecasts_ti[:,:,:,:len(self.season_setup.locations)], 
#                        columns= self.season_setup.locations, index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]
#                a["US"] = pd.DataFrame(forecasts_national[sim_id],
#                        columns= ["US"], index=pd.date_range(self.season_setup.fluseason_startdate, self.season_setup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT")).loc[target_dates]
#
#                a = a.reset_index().rename(columns={'index': 'target_end_date'})
#                a = pd.melt(a,id_vars="target_end_date",var_name="location")
#
#                
#                df_list.append(a)
#
#            df2 = pd.concat(df_list)
#            df2["reference_date"] = forecast_date_str
#            df2["target"] = "wk flu hosp rate change"
#            df2["horizon"] = df["target_end_date"].map(target_dict)
#            df2["output_type"] = "pmf"
#            df2 = df2[["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]]


        df.to_csv(f"{directory}/{forecast_date_str}-{prefix}.csv", index=False)

        if save_plot:
            self.plot_forecasts(fluforecasts_ti, forecasts_national, directory=directory, prefix=prefix, forecast_date=forecast_date)
        

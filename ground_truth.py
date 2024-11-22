import math
from inspect import isfunction
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from season_setup import SeasonSetup


import numpy as np
import pandas as pd
import xarray as xr

import datetime

import myutils, build_dataset


class GroundTruth():
    def __init__(self, season_first_year: str, data_date: datetime.datetime, mask_date: datetime.datetime, from_final_data:bool=False, channels=1, image_size=64, nogit=False):
        self.season_first_year = season_first_year
        self.data_date = data_date
        self.mask_date = mask_date
        self.channels = channels
        self.image_size=image_size


        if not nogit: self.git_checkout_data_rev(target_date=None)

        self.season_setup = SeasonSetup.from_flusight(season_first_year=self.season_first_year, remove_territories=True, remove_us=True)

        flusight = build_dataset.get_from_epidata(dataset=f"flusight{self.season_first_year}", season_setup=self.season_setup, write=False)
        gt_df_final = flusight[flusight["fluseason"] == int(self.season_first_year)]

        if from_final_data:
            gt_df = gt_df_final.copy()
        else:
            if not nogit: self.git_checkout_data_rev(target_date=data_date)
            flusight = build_dataset.get_from_epidata(dataset=f"flusight{self.season_first_year}", season_setup=self.season_setup, write=False)
            gt_df = flusight[flusight["fluseason"] == int(self.season_first_year)]   
            if not nogit: self.git_checkout_data_rev(target_date=None)

        
        self.gt_df = gt_df[gt_df["location_code"].isin(self.season_setup.locations)]
        self.gt_df_final = gt_df_final[gt_df_final["location_code"].isin(self.season_setup.locations)]

        self.gt_xarr = build_dataset.dataframe_to_xarray(self.gt_df, season_setup=self.season_setup, 
            xarray_name = "gt_flusight_incidHosp", 
            xarrax_features = "incidHosp")
        
        self.gt_final_xarr = build_dataset.dataframe_to_xarray(self.gt_df_final, season_setup=self.season_setup, 
            xarray_name = "gt_flusight_incidHos_final", 
            xarrax_features = "incidHosp")

        # perturb the masking:
        l = [d.astype('datetime64[D]').view('int64').astype('datetime64[D]').tolist() for d in self.gt_xarr.coords['date'].to_numpy()]
        for i, d in enumerate(l):
            if d is not None and d < self.mask_date.date():
                self.inpaintfrom_idx = i + 1

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
        fig, axes = plt.subplots(11, 5, sharex=True, figsize=(12,24))
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

        nplace_toplot = 51
        #nplace_toplot = 3 # less plots for faster iteration
        plot_past_median = False
        if plot_past_median:
            plotrange=slice(None)
        else:
            plotrange=slice(self.inpaintfrom_idx,-1)


        if self.season_first_year == "2023" or self.season_first_year == "2024":
            gt2022 = GroundTruth(season_first_year="2022", 
                            data_date=datetime.datetime.combine(datetime.date(2023,7,15), datetime.datetime.min.time()),
                            mask_date=datetime.datetime.today(),
                            channels=self.channels,
                            image_size=self.image_size
                            )
        if self.season_first_year == "2024":
            gt2023 = GroundTruth(season_first_year="2023", 
                data_date=datetime.datetime.combine(datetime.date(2023,7,15), datetime.datetime.min.time()),
                mask_date=datetime.datetime.today(),
                channels=self.channels,
                image_size=self.image_size
                )

        for plot_title, plot_spec in plot_specs.items():
            #print(f"doing {plot_title}...")
            fig, axes = plt.subplots(nplace_toplot+1, 2, figsize=(10,nplace_toplot*3.5), dpi=200)
            for iax in range(2):
                ax = axes[0][iax]
    
                x = np.arange(64)
                if iax == 0:
                    x_lims = (0, 52)
                elif iax == 1:
                    x_lims = (idx_now-3, idx_horizon)
    
                # US WIDE: quantiles and median, US-wide
                for iqt in plot_spec["quantiles_idx"]:
                    #print(f"up: {flusight_quantile_pairs[iqt,0]} - lo: {flusight_quantile_pairs[iqt,1]}")
                    # TODO: not exactly true that it is the sum of quantiles (sum of quantile is not quantile of sum)
                    ylo = np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0]
                    yup = np.quantile(forecasts_national, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0]
                    ax.fill_between(x[plotrange], 
                                    ylo[plotrange], 
                                    yup[plotrange], 
                                    alpha=.1, 
                                    color=plot_spec["color"])
    
                    # widest quantile pair is the first one. We take the up quantile of it + a few % as x_lim
                    if iqt == plot_spec["quantiles_idx"][0]:
                        if plot_past_median:
                            max_y_value = max(yup[x_lims[0]:x_lims[1]])
                        else:
                            max_y_value = max(yup[self.inpaintfrom_idx:x_lims[1]])
                        max_y_value = max(max_y_value, self.gt_xarr.data[0,:self.inpaintfrom_idx].sum(axis=1)[x_lims[0]:x_lims[1]].max())
                        max_y_value = max_y_value + max_y_value*.05 # 10% more
    
                # median
                ax.plot(x[plotrange], np.quantile(forecasts_national, myutils.flusight_quantiles[12], axis=0)[0][plotrange], color=plot_spec["color"], marker='.', label='forecast median')
    
                # ground truth
                ax.plot(self.gt_xarr.data[0,:self.inpaintfrom_idx].sum(axis=1), color=color_gt, marker = '.', lw=.5, label='ground-truth')
                if self.season_first_year == "2023" or self.season_first_year == "2024":
                    ax.plot(gt2022.gt_xarr.data[0,:].sum(axis=1), color=color_past, ls='dashed', lw=.5, label='2022 ground-truth')
                if self.season_first_year == "2024":
                    ax.plot(gt2022.gt_xarr.data[0,:].sum(axis=1), color=color_past, ls='dashdot', lw=.5, label='2023 ground-truth')

                if iax==0:
                    ax.legend(fontsize=8)
    
                #ax.set_xticks(np.arange(0,53,13))


                ax.set_xlim(x_lims)
                ax.set_ylim(bottom=0, top=max_y_value)
                ax.axvline(idx_now, c='k', lw=1, ls='-.')
                if iax == 0:
                    ax.axvline(idx_horizon, c='k', lw=1, ls='-.')
                ax.set_title("National")

                sns.despine(ax = ax, trim = True, offset=4)

                # INDIVDIDUAL STATES: quantiles, median and ground-truth
                max_y_value = np.zeros(52)
                for iqt in plot_spec["quantiles_idx"]:
                    yup = np.quantile(fluforecasts_ti, myutils.flusight_quantile_pairs[iqt,0], axis=0)[0]
                    ylo = np.quantile(fluforecasts_ti, myutils.flusight_quantile_pairs[iqt,1], axis=0)[0]

                    # widest quantile pair is the first one. We take the up quantile of it + a few % as x_lim
                    if iqt == plot_spec["quantiles_idx"][0]:
                        for ipl in range(nplace_toplot):
                            if plot_past_median:
                                max_y_value[ipl] = max(ylo[x_lims[0]:x_lims[1], ipl])
                            else:
                                max_y_value[ipl] = max(ylo[self.inpaintfrom_idx:x_lims[1], ipl])
                            #max_y_value[ipl] =  max(ylo[x_lims[:x_lims[1], ipl])
                            max_y_value[ipl] = max(max_y_value[ipl], self.gt_xarr.data[0,:self.inpaintfrom_idx, ipl][x_lims[0]:x_lims[1]].max())
                            max_y_value[ipl] = max_y_value[ipl] + max_y_value[ipl]*.05 # 10% more for the y_max value

                    for ipl in range(nplace_toplot):
                        ax = axes[ipl+1][iax]
                        ax.fill_between((x)[plotrange],  (yup[:,ipl])[plotrange], (ylo[:,ipl])[plotrange], alpha=.1, color=plot_spec["color"])

                # median line and ground truth for states
                for ipl in range(nplace_toplot):   
                    ax = axes[ipl+1][iax]
                    # median
                    ax.plot(np.arange(64)[plotrange],
                            np.quantile(fluforecasts_ti, myutils.flusight_quantiles[12], axis=0)[0,:,ipl][plotrange], color=plot_spec["color"], marker = '.', lw=.5)
                    # ground truth
                    ax.plot(self.gt_xarr.data[0,:self.inpaintfrom_idx, ipl], color=color_gt, marker = '.', lw=.5)
                    if self.season_first_year == "2023" or self.season_first_year == "2024":
                        ax.plot(gt2022.gt_xarr.data[0,:, ipl], color=color_past, ls='dashed', lw=.5)
                    if self.season_first_year == "2024":
                        ax.plot(gt2023.gt_xarr.data[0,:, ipl], color=color_past, ls='dashdot', lw=.5)

                    ax.axvline(idx_now, c='k', lw=1, ls='-.')
                    if iax == 0:
                        ax.axvline(idx_horizon, c='k', lw=1, ls='-.')
                    ax.set_xlim(x_lims)
                    ax.set_ylim(bottom=0, top=max_y_value[ipl])
                    if iax==0: ax.set_ylabel("New Hosp. Admissions")
                    ax.set_title(self.season_setup.get_location_name(self.season_setup.locations[ipl]))
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

        print(target_dates)

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
        

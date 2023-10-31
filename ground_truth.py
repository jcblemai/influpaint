import math
from inspect import isfunction
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


import numpy as np
import pandas as pd
import xarray as xr

import datetime

import utils, data_utils


class GroundTruth():
    def __init__(self, season_first_year: str, data_date: datetime.datetime, mask_date: datetime.datetime, from_final_data:bool=False):
        self.season_first_year = season_first_year
        self.data_date = data_date
        self.mask_date = mask_date

        self.git_checkout_data_rev(target_date=None)

        if self.season_first_year == "2023":
            self.flusetup = data_utils.FluSetup.from_flusight2023_24(fluseason_startdate=pd.to_datetime("2023-07-24"), remove_territories=True)
            flusight = data_utils.get_from_epidata(dataset="flusight2023_24", flusetup=self.flusetup, write=False)
            gt_df_final = flusight[flusight["fluseason"] == 2023]
            if from_final_data:
                gt_df = gt_df_final.copy()
            else:
                self.git_checkout_data_rev(target_date=data_date)
                flusight = data_utils.get_from_epidata(dataset="flusight2023_24", flusetup=self.flusetup, write=False)
                gt_df = flusight[flusight["fluseason"] == 2023]   
                self.git_checkout_data_rev(target_date=None)
        elif self.season_first_year == "2022":
            self.flusetup = data_utils.FluSetup.from_flusight2023_24(fluseason_startdate=pd.to_datetime("2022-07-24"), remove_territories=True)
            flusight = data_utils.get_from_epidata(dataset="flusight2022_23", flusetup=self.flusetup, write=False)
            gt_df_final = flusight[flusight["fluseason"] == 2022]
            if from_final_data:
                gt_df = gt_df_final.copy()
            else:
                self.git_checkout_data_rev(target_date=data_date)
                flusight = data_utils.get_from_epidata(dataset="flusight2022_23", flusetup=self.flusetup, write=False)
                gt_df = flusight[flusight["fluseason"] == 2022]
                self.git_checkout_data_rev(target_date=None)  
        else:
            raise ValueError("not supported")
        
        self.gt_df = gt_df[gt_df["location_code"].isin(self.flusetup.locations)]
        self.gt_df_final = gt_df_final[gt_df_final["location_code"].isin(self.flusetup.locations)]


        self.gt_xarr = data_utils.dataframe_to_xarray(self.gt_df, flusetup=self.flusetup, 
            xarray_name = "gt_flusight_incidHosp", 
            xarrax_features = "incidHosp")
        
        self.gt_final_xarr = data_utils.dataframe_to_xarray(self.gt_df_final, flusetup=self.flusetup, 
            xarray_name = "gt_flusight_incidHos_final", 
            xarrax_features = "incidHosp")
        
    
    def git_checkout_data_rev(self, target_date=None):
        import pygit2
        if self.season_first_year == "2023":
            repo_path = "Flusight/FluSight-forecast-hub/"
            main_branch = "main"
        elif  self.season_first_year == "2022":
            repo_path = "Flusight/Flusight-forecast-data/"
            main_branch = "master"


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
        fig, axes = plt.subplots(13, 4, sharex=True, figsize=(12,24))
        gt_piv  = self.gt_df.pivot(index = "week_enddate", columns='location_code', values='value')
        gt_piv_final = self.gt_df_final.pivot(index = "week_enddate", columns='location_code', values='value')
        ax = axes.flat[0]
        ax.plot(gt_piv[self.flusetup.locations].sum(axis=1), color="black", linewidth=2,label="datadate")
        ax.plot(gt_piv_final[self.flusetup.locations].sum(axis=1), lw=1, color='r', ls='-.', label="final")
        ax.legend()
        ax.set_ylim(0)
        ax.set_title("US")
        for idx, pl in enumerate(gt_piv.columns):
            ax = axes.flat[idx+1]
            ax.plot(gt_piv[pl], lw=2, color='k')
            ax.plot(gt_piv_final[pl], lw=1, color='r', ls='-.')
            ax.set_title(self.flusetup.get_location_name(pl))
            #ax.grid()
            ax.set_ylim(0)
            ax.set_xlim(self.flusetup.fluseason_startdate, self.flusetup.fluseason_startdate + datetime.timedelta(days=365))
            #ax.set_xticks(flusetup.get_dates(52).resample("M"))
            #ax.plot(pd.date_range(flusetup.fluseason_startdate, flusetup.fluseason_startdate + datetime.timedelta(days=64*7), freq="W-SAT"), data.flu_dyn[-50:,0,:,idx].T, c='r', lw=.5, alpha=.2)
        fig.tight_layout()
        fig.autofmt_xdate()

    def plot_mask(self):
        # check that it stitch
        fig, axes = plt.subplots(1, 3, figsize=(6,6), dpi=200, sharex=True, sharey=True)
        axes[1].imshow(gt_keep_mask[0], alpha=.3, cmap = "rainbow")
        axes[0].imshow(gt[0], cmap='Greys')


        axes[2].imshow(gt[0], cmap='Greys')
        axes[2].imshow(gt_keep_mask[0], alpha=.3, cmap = "rainbow")

    def mask(self, channels ,image_size, idx=None, date=None):
        if idx is None and date is None:
            self.inpaintfrom_idx = len(self.gt_df.week_enddate.unique())
            how += "(from futur)"
        elif idx is not None:
            self.inpaintfrom_idx = idx
            how += "(from provided idx)"
        elif date is not None:
            raise ValueError("TODO")
        

        self.gt_keep_mask = np.ones((channels,image_size,image_size))
        self.gt_keep_mask[:,self.inpaintfrom_idx:,:] = 0
        
        print(f"Masking, >> {self.inpaintfrom_idx} ({how}) weeks already in data, inpainting the next ones")

        

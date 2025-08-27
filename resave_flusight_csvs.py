#!/usr/bin/env python3
"""
Resave FluSight CSV files from inpainting results.

This script processes inpainting results from the batch runs and regenerates
FluSight-format CSV files using the stored fluforecasts_ti.npy files.

Usage:
    python resave_flusight_csvs.py
"""

import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import contextlib
from io import StringIO

# Add the project root to the path
sys.path.append('/Users/chadi/Research/influpaint')

from influpaint.utils.ground_truth import GroundTruth
from influpaint.utils.season_axis import SeasonAxis


def extract_info_from_folder_name(folder_name):
    """Extract scenario info and date from folder name like:
    i806::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_PoisPadScaleSmall::inpaint_CoPaint::conf_celebahq::2023-12-09
    """
    parts = folder_name.split("::")
    
    # Extract scenario ID (first part after 'i')
    scenario_id = int(parts[0][1:])  # Remove 'i' prefix
    
    # Extract model info for prefix
    model_part = parts[1]  # e.g., m_U500cRx124
    
    # Extract date (last part)
    date_str = parts[-1]
    
    # Extract config name (second to last part after 'conf_')
    config_part = parts[-2]
    config_name = config_part.replace('conf_', '') if config_part.startswith('conf_') else config_part
    
    return {
        'scenario_id': scenario_id,
        'model_part': model_part,
        'date_str': date_str,
        'config_name': config_name,
        'folder_name': folder_name
    }


def create_model_prefix(folder_name):
    """Create a model prefix from the full folder name (excluding date).
    
    Use the complete model specification to avoid conflicts.
    """
    # Remove the date part (last component) from folder name
    # i806::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_PoisPadScaleSmall::inpaint_CoPaint::conf_celebahq::2023-12-09
    # becomes: i806::m_U500cRx124::ds_30S70M::tr_Sqrt::ri_PoisPadScaleSmall::inpaint_CoPaint::conf_celebahq
    parts = folder_name.split("::")
    model_spec = "::".join(parts[:-1])  # Everything except the date
    
    # Replace :: with underscores for filename compatibility
    return model_spec.replace("::", "_")


def process_folder(folder_path, gt_cache, season_setup, verbose=False):
    """Process a single result folder and generate CSV using cached GroundTruth objects."""
    
    folder_name = os.path.basename(folder_path)
    
    # Extract information from folder name
    try:
        info = extract_info_from_folder_name(folder_name)
    except Exception as e:
        if verbose:
            print(f"Processing: {folder_name}")
            print(f"  ERROR: Could not parse folder name: {e}")
        return False
    
    # Check if fluforecasts_ti.npy exists
    fluforecasts_ti_path = os.path.join(folder_path, 'fluforecasts_ti.npy')
    if not os.path.exists(fluforecasts_ti_path):
        if verbose:
            print(f"Processing: {folder_name}")
            print(f"  WARNING: fluforecasts_ti.npy not found, skipping")
        return False
    
    # Load the forecasts
    try:
        fluforecasts_ti = np.load(fluforecasts_ti_path)
    except Exception as e:
        if verbose:
            print(f"Processing: {folder_name}")
            print(f"  ERROR: Could not load fluforecasts_ti.npy: {e}")
        return False
    
    # Parse the forecast date
    try:
        forecast_date = pd.to_datetime(info['date_str']).date()
    except Exception as e:
        if verbose:
            print(f"Processing: {folder_name}")
            print(f"  ERROR: Could not parse date {info['date_str']}: {e}")
        return False
    
    # Determine season year from forecast date using the provided SeasonAxis
    season_first_year = str(season_setup.get_fluseason_year(pd.to_datetime(forecast_date)))
    
    # Get or create cached GroundTruth object
    if season_first_year not in gt_cache:
        if verbose:
            print(f"Creating GroundTruth for season {season_first_year}")
        try:
            gt_cache[season_first_year] = GroundTruth(
                season_first_year=season_first_year,
                data_date=datetime.datetime.today(),
                mask_date=pd.to_datetime(forecast_date),  # Use first date for mask
                channels=1,
                image_size=64,
                nogit=True
            )
        except Exception as e:
            if verbose:
                print(f"Processing: {folder_name}")
                print(f"  ERROR: Could not create GroundTruth: {e}")
            return False
    
    gt = gt_cache[season_first_year]
    
    # Calculate national forecasts (sum across locations)
    try:
        forecasts_national = fluforecasts_ti.sum(axis=-1)
    except Exception as e:
        if verbose:
            print(f"Processing: {folder_name}")
            print(f"  ERROR: Could not calculate national forecasts: {e}")
        return False
    
    # Create model prefix from full folder name
    prefix = create_model_prefix(folder_name)
    
    # Export CSV using the same function as inpainting
    try:
        gt.export_forecasts_2023(
            fluforecasts_ti=fluforecasts_ti,
            forecasts_national=forecasts_national,
            directory=folder_path,
            prefix=prefix,
            forecast_date=forecast_date,
            save_plot=False,  # Don't regenerate plots
            nochecks=True      # Skip validation checks for speed
        )
        print(f"Processing: {folder_name} - SUCCESS")
        return True
    except Exception as e:
        print(f"Processing: {folder_name}")
        print(f"  ERROR: Could not export CSV: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        return False


def read_jobs_file(jobs_file="inpaint_jobs_paper-2025-07-22.txt"):
    """Read the original jobs file to map failed results back to job IDs."""
    try:
        df = pd.read_csv(jobs_file, dtype={'location': str})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df['location'] = df['location'].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"ERROR: Could not read jobs file {jobs_file}: {e}")
        return pd.DataFrame()


def create_failed_jobs_report(failed_folders, jobs_df):
    """Create comprehensive failed jobs report from failed folder names."""
    if jobs_df.empty:
        return []
    
    failed_jobs = []
    
    for folder_name in failed_folders:
        try:
            # Extract info from folder name
            info = extract_info_from_folder_name(folder_name)
            scenario_id = info['scenario_id']
            config_name = info['config_name']
            date_str = info['date_str']
            forecast_date = pd.to_datetime(date_str).date()
            
            # Find matching job in original jobs file
            matching_jobs = jobs_df[
                (jobs_df["scenario_id"] == scenario_id) & 
                (jobs_df["config"] == config_name) & 
                (jobs_df["date"] == forecast_date)
            ]
            
            if not matching_jobs.empty:
                job_row = matching_jobs.iloc[0]
                failed_jobs.append({
                    'job_id': job_row["job_id"],
                    'scenario_id': scenario_id,
                    'run_id': job_row["run_id"],
                    'season': job_row["season"],
                    'date': forecast_date,
                    'config': config_name,
                    'folder_name': folder_name
                })
        except Exception:
            # If we can't parse the folder name, skip it
            continue
    
    return failed_jobs


def main():
    """Main function to process all results folders."""
    
    results_base_dir = "/Users/chadi/Research/influpaint/from_longleaf/influpaint_res/07b44fa_paper-2025-07-22_inpainting_2025-07-27"
    
    if not os.path.exists(results_base_dir):
        print(f"ERROR: Results directory not found: {results_base_dir}")
        return
    
    print(f"Processing results from: {results_base_dir}")
    
    # Get all subdirectories
    all_folders = [d for d in os.listdir(results_base_dir) 
                   if os.path.isdir(os.path.join(results_base_dir, d))]
    
    print(f"Found {len(all_folders)} folders to process")
    
    # Cache for GroundTruth objects by season
    gt_cache = {}
    
    # Create SeasonAxis setup once for all folders
    season_setup = SeasonAxis.for_flusight(remove_us=True, remove_territories=True)
    
    success_count = 0
    total_count = 0
    failed_folders = []
    
    print("Processing folders (only showing failures)...")
    
    for folder_name in sorted(all_folders):
        folder_path = os.path.join(results_base_dir, folder_name)
        total_count += 1
        
        if process_folder(folder_path, gt_cache, season_setup, verbose=False):
            success_count += 1
        else:
            failed_folders.append(folder_name)
    
    print(f"\nCompleted: {success_count}/{total_count} folders processed successfully")
    print(f"Failed: {len(failed_folders)}")
    
    if failed_folders:
        print(f"\nFailed folders ({len(failed_folders)}):")
        for failed in failed_folders:
            print(f"  - {failed}")
        
        # Create failed jobs report
        jobs_df = read_jobs_file()
        if not jobs_df.empty:
            failed_jobs = create_failed_jobs_report(failed_folders, jobs_df)
            
            if failed_jobs:
                # Write single combined failed jobs file
                os.makedirs("results", exist_ok=True)
                failed_jobs_df = pd.DataFrame(failed_jobs)
                failed_jobs_file = "results/failed_inpaint_jobs_all.txt"
                failed_jobs_df[['job_id', 'scenario_id', 'run_id', 'season', 'date', 'config']].to_csv(
                    failed_jobs_file, index=False
                )
                print(f"\n✓ Written {len(failed_jobs)} failed jobs to: {failed_jobs_file}")
                
                # Output summary to stdout
                print(f"✓ Failed jobs summary: {len(failed_jobs)} total")
                by_season = failed_jobs_df.groupby('season').size()
                for season, count in by_season.items():
                    print(f"  - {season}: {count} failed jobs")
    
    print(f"\nCreated GroundTruth objects for seasons: {list(gt_cache.keys())}")


if __name__ == '__main__':
    main()
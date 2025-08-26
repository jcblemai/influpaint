#!/usr/bin/env python3
"""
Full Evaluation using scoringutils R package via subprocess calls
Replicates 4-evaluate-candidates.py functionality using R scoringutils for WIS scoring
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import datetime as dt
import time
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    """Configuration settings for evaluation (copied from 4-evaluate-candidates.py)."""
    
    # File paths
    JOBS_FILE = "inpaint_jobs_paper-2025-07-22.txt"
    INPAINT_RES_BASE = "from_longleaf/influpaint_res"
    
    FLUSIGHT_BASES = {
        "2023-2024": "Flusight/2023-2024/FluSight-forecast-hub-official",
        "2024-2025": "Flusight/2024-2025/FluSight-forecast-hub-official",
    }
    
    # Model filtering
    IGNORED_FLUSIGHT_MODELS = {
        "LosAlamos_NAU-CModel_Flu",
        "SigSci-CREG", 
        "SigSci-TSENS",
        "LUcompUncertLab-experthuman",
        "VTSanghani-ExogModel",
        "CADPH-FluCAT_Ensemble",
    }
    
    # Evaluation parameters
    TARGET_NAME = "wk inc flu hosp"
    HORIZONS = [0, 1, 2, 3]
    ALLOW_MISSING_DATES_PER_MODEL = 5
    
    # Quantile requirements for forecasts
    REQUIRED_QUANTILES = [0.01, 0.025] + list(np.arange(0.05, 0.95 + 0.05, 0.05)) + [0.975, 0.99]


class ScoringutilsFullEvaluator:
    """Full evaluator using R scoringutils for real influpaint vs flusight data."""
    
    def __init__(self):
        """Initialize evaluator and find working R."""
        self.rscript_path = self.find_working_rscript()
        if not self.rscript_path:
            raise RuntimeError("Could not find R installation with scoringutils")
    
    def find_working_rscript(self) -> str:
        """Find Rscript that has scoringutils installed."""
        candidates = ['/usr/local/bin/Rscript', '/usr/bin/Rscript']
        
        for rscript in candidates:
            if os.path.exists(rscript):
                try:
                    subprocess.check_output([rscript, '-e', 'library(scoringutils); cat("OK")'], 
                                          stderr=subprocess.DEVNULL)
                    print(f"✓ Found working Rscript: {rscript}")
                    return rscript
                except:
                    continue
        
        print("❌ No working Rscript found with scoringutils")
        return None
    
    # Copy helper functions from 4-evaluate-candidates.py
    def read_jobs(self, jobs_file: Optional[str] = None) -> pd.DataFrame:
        """Read job dates from inpaint jobs file."""
        if jobs_file is None:
            jobs_file = Config.JOBS_FILE
        df = pd.read_csv(jobs_file)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    
    def season_dates_from_jobs(self, df: pd.DataFrame) -> Dict[str, List[dt.date]]:
        """Extract unique dates per season from jobs dataframe."""
        out = {}
        for season, g in df.groupby("season"):
            out[str(season)] = sorted(g["date"].unique().tolist())
        return out
    
    def load_ground_truth(self, season: str) -> pd.DataFrame:
        """Load ground truth hospital admissions data for a season."""
        if season not in Config.FLUSIGHT_BASES:
            raise KeyError(f"Season '{season}' not found in configuration")
        
        base = Config.FLUSIGHT_BASES[season]
        p = os.path.join(base, "target-data", "target-hospital-admissions.csv")
        
        if not os.path.exists(p):
            raise FileNotFoundError(f"Ground truth file not found: {p}")
        
        gt = pd.read_csv(p)
        gt["date"] = pd.to_datetime(gt["date"]).dt.date
        required_cols = ["date", "location", "value"]
        return gt[required_cols]
    
    def list_flusight_models(self, season: str) -> List[str]:
        """List available FluSight models for a season."""
        base = Config.FLUSIGHT_BASES[season]
        model_output = os.path.join(base, "model-output")
        if not os.path.isdir(model_output):
            return []
        models = [d for d in os.listdir(model_output) if os.path.isdir(os.path.join(model_output, d))]
        models = [m for m in models if m not in Config.IGNORED_FLUSIGHT_MODELS]
        return sorted(models)
    
    def load_flusight_forecast(self, season: str, model: str, ref_date: dt.date) -> pd.DataFrame:
        """Load FluSight forecast for specific model and date."""
        base = Config.FLUSIGHT_BASES[season]
        p = os.path.join(base, "model-output", model, f"{str(ref_date)}-{model}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        
        df = pd.read_csv(p)
        df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
        df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["location"] = df["location"].astype(str).str.strip()
        return df
    
    def find_influpaint_csvs(self, base_dir: str, config: str, ref_date: dt.date) -> List[Tuple[str, str]]:
        """Search for InfluPaint CSVs matching config and date."""
        res = []
        date_str = str(ref_date)
        batch_prefix = "07b44fa_paper-2025-07-22_inpainting_"
        
        for batch in os.listdir(base_dir):
            if not batch.startswith(batch_prefix):
                continue
            batch_dir = os.path.join(base_dir, batch)
            if not os.path.isdir(batch_dir):
                continue
            
            for root, _, files in os.walk(batch_dir):
                if f"::conf_{config}::{date_str}" in root:
                    for f in files:
                        if f.endswith(".csv") and f.startswith(date_str) and not f.endswith("-copaint.csv"):
                            path = os.path.join(root, f)
                            parts = root.split("::conf_")
                            left = parts[0].rstrip(":")
                            model_id = os.path.basename(left)
                            full_name = f"{model_id}::{config}"
                            res.append((full_name, path))
        return res
    
    def filter_forecast_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter forecast dataframe to quantile forecasts for target of interest."""
        keep = (df["output_type"] == "quantile") & (df["target"] == Config.TARGET_NAME)
        df = df.loc[keep].copy()
        df = df[df["horizon"].isin(Config.HORIZONS)]
        df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
        df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
        df["location"] = df["location"].astype(str).str.strip()
        return df
    
    def collect_all_forecasts(self, season: str, dates: List[dt.date]) -> pd.DataFrame:
        """Collect all forecasts (FluSight + InfluPaint) for a season."""
        all_forecasts = []
        
        # Collect FluSight forecasts
        print(f"Collecting FluSight forecasts for {season}...")
        flusight_models = self.list_flusight_models(season)
        
        for model in flusight_models:
            present = 0
            for d in dates:
                try:
                    df = self.load_flusight_forecast(season, model, d)
                    df = self.filter_forecast_df(df)
                    
                    # Check quantile requirements
                    have = sorted(df["output_type_id"].unique().tolist())
                    missing_q = sorted(set(np.round(Config.REQUIRED_QUANTILES, 6)) - set(np.round(have, 6)))
                    if missing_q:
                        continue
                    
                    # Add metadata
                    df['model'] = model
                    df['group'] = 'flusight'
                    df['forecast_date'] = d
                    all_forecasts.append(df)
                    present += 1
                except Exception:
                    continue
            
            missing = len(dates) - present
            if missing <= Config.ALLOW_MISSING_DATES_PER_MODEL:
                print(f"  {model}: {present}/{len(dates)} dates")
        
        # Collect InfluPaint forecasts
        print(f"Collecting InfluPaint forecasts for {season}...")
        jobs = self.read_jobs()
        season_jobs = jobs[jobs["season"] == season]
        configs = sorted(season_jobs["config"].unique().tolist())
        
        model_paths: Dict[str, Dict[dt.date, str]] = {}
        for config in configs:
            for d in dates:
                matches = self.find_influpaint_csvs(Config.INPAINT_RES_BASE, config, d)
                for run_name, path in matches:
                    model_paths.setdefault(run_name, {})[d] = path
        
        for run_name, by_date in model_paths.items():
            present = 0
            for d in dates:
                if d not in by_date:
                    continue
                    
                try:
                    df = pd.read_csv(by_date[d])
                    
                    # Normalize column names
                    if "output_type" not in df.columns and "type" in df.columns:
                        df = df.rename(columns={"type": "output_type"})
                    if "output_type_id" not in df.columns and "quantile" in df.columns:
                        df = df.rename(columns={"quantile": "output_type_id"})
                    
                    df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.date
                    df["output_type_id"] = pd.to_numeric(df["output_type_id"], errors="coerce")
                    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
                    df = self.filter_forecast_df(df)
                    
                    # Check quantile requirements
                    have = sorted(df["output_type_id"].unique().tolist())
                    missing_q = sorted(set(np.round(Config.REQUIRED_QUANTILES, 6)) - set(np.round(have, 6)))
                    if missing_q:
                        continue
                    
                    # Add metadata
                    df['model'] = run_name
                    df['group'] = 'influpaint'
                    df['forecast_date'] = d
                    all_forecasts.append(df)
                    present += 1
                except Exception:
                    continue
            
            missing = len(dates) - present
            if missing <= Config.ALLOW_MISSING_DATES_PER_MODEL:
                print(f"  {run_name}: {present}/{len(dates)} dates")
        
        if not all_forecasts:
            raise RuntimeError(f"No valid forecasts found for {season}")
        
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        print(f"✓ Collected {len(combined_forecasts)} total forecast records")
        return combined_forecasts
    
    def prepare_combined_data(self, forecast_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare combined forecast+truth data for scoringutils in Python."""
        print("Preparing combined data for scoringutils...")
        
        # Rename truth columns to match merge
        truth_renamed = truth_df.rename(columns={'date': 'target_end_date', 'value': 'observed'})
        
        # Merge forecasts with truth
        print(f"Merging {len(forecast_df):,} forecasts with {len(truth_renamed):,} truth records...")
        combined = forecast_df.merge(
            truth_renamed[['target_end_date', 'location', 'observed']], 
            on=['target_end_date', 'location'], 
            how='inner'
        )
        
        # Rename forecast columns for scoringutils
        combined = combined.rename(columns={
            'output_type_id': 'quantile',
            'value': 'predicted'
        })
        
        # Filter to quantile forecasts only
        combined = combined[combined['output_type'] == 'quantile'].copy()
        
        # Remove rows without truth data
        combined = combined.dropna(subset=['observed'])
        
        print(f"✓ Combined data: {len(combined):,} records")
        print(f"✓ Models: {len(combined['model'].unique())}")
        print(f"✓ Seasons: {sorted(combined['season'].unique())}")
        print(f"✓ Date range: {combined['target_end_date'].min()} to {combined['target_end_date'].max()}")
        
        return combined
    
    def evaluate_season(self, season: str, dates: List[dt.date], save_dir: str) -> Dict:
        """Evaluate a single season using scoringutils."""
        print(f"\n🚀 Evaluating {season} with {len(dates)} dates")
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Collect all forecasts
            all_forecasts = self.collect_all_forecasts(season, dates)
            
            # Load ground truth
            truth_df = self.load_ground_truth(season)
            
            # Score with scoringutils
            scores_df = self.score_with_scoringutils(all_forecasts, truth_df)
            
            # Print summary
            print(f"\n📊 Results Summary for {season}:")
            print("="*50)
            
            # Overall summary
            total_models = len(scores_df['model'].unique())
            print(f"Total models evaluated: {total_models}")
            
            # Group summary
            group_summary = scores_df.groupby('group')['wis'].agg(['count', 'mean', 'std']).round(3)
            print(f"\nWIS by group:")
            print(group_summary)
            
            # Top models
            model_wis = scores_df.groupby(['model', 'group'])['wis'].mean().sort_values()
            print(f"\nTop 10 models by WIS:")
            print(model_wis.head(10))
            
            # Save results
            results_file = os.path.join(save_dir, f'{season}_scoringutils_results.csv')
            scores_df.to_csv(results_file, index=False)
            print(f"✓ Saved detailed results: {results_file}")
            
            return {
                'season': season,
                'scores_df': scores_df,
                'forecasts_df': all_forecasts,
                'truth_df': truth_df,
                'summary': group_summary,
                'save_dir': save_dir
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {season}: {e}")
            raise
    
    def run_full_evaluation(self):
        """Run full evaluation for all seasons and save combined results."""
        total_start = time.time()
        print("🚀 Starting full InfluPaint vs FluSight evaluation using scoringutils")
        
        # Read jobs and get seasons
        jobs = self.read_jobs()
        by_season = self.season_dates_from_jobs(jobs)
        
        all_forecasts = []
        all_truth = []
        
        # Collect data from all seasons
        data_start = time.time()
        for season, dates in by_season.items():
            if season not in Config.FLUSIGHT_BASES:
                print(f"Skipping {season}: no local FluSight data")
                continue
            
            print(f"\n📊 Processing {season}...")
            
            # Collect forecasts for this season
            season_forecasts = self.collect_all_forecasts(season, dates)
            season_forecasts['season'] = season
            all_forecasts.append(season_forecasts)
            
            # Collect truth for this season
            season_truth = self.load_ground_truth(season)
            season_truth['season'] = season
            all_truth.append(season_truth)
        
        if not all_forecasts:
            raise RuntimeError("No forecast data found for any season")
        
        # Combine all seasons
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        combined_truth = pd.concat(all_truth, ignore_index=True)
        
        data_elapsed = time.time() - data_start
        print(f"⏱️  Data collection completed in {data_elapsed:.2f} seconds")
        
        print(f"\n🔄 Combined data:")
        print(f"  Total forecasts: {len(combined_forecasts):,}")
        print(f"  Total models: {len(combined_forecasts['model'].unique())}")
        print(f"  Seasons: {sorted(combined_forecasts['season'].unique())}")
        
        # Prepare combined data in Python (all data wrangling done here)
        print("\n🔧 Preparing combined dataset...")
        prep_start = time.time()
        combined_data = self.prepare_combined_data(combined_forecasts, combined_truth)
        prep_elapsed = time.time() - prep_start
        print(f"⏱️  Data preparation completed in {prep_elapsed:.2f} seconds")
        
        # Save combined data to disk for R to use
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        
        combined_file = os.path.join(save_dir, 'combined_forecast_truth_data.csv')
        combined_data.to_csv(combined_file, index=False)
        print(f"✓ Saved combined data: {combined_file}")
        
        # Print summary for user
        print(f"\n📊 Data Summary:")
        print("="*60)
        print(f"Total records ready for scoring: {len(combined_data):,}")
        print(f"Models: {len(combined_data['model'].unique())}")
        print(f"Seasons: {sorted(combined_data['season'].unique())}")
        print(f"Locations: {len(combined_data['location'].unique())}")
        print(f"Date range: {combined_data['target_end_date'].min()} to {combined_data['target_end_date'].max()}")
        
        print(f"\nColumns available for scoring:")
        print(list(combined_data.columns))
        
        print(f"\nSample of prepared data:")
        display_cols = ['model', 'group', 'season', 'location', 'horizon', 'target_end_date', 'quantile', 'predicted', 'observed']
        available_display_cols = [col for col in display_cols if col in combined_data.columns]
        print(combined_data[available_display_cols].head(10))
        
        # Final timing
        total_elapsed = time.time() - total_start
        print(f"\n⏱️  TOTAL DATA PREPARATION TIME: {total_elapsed:.2f} seconds")
        print(f"💾 Combined data saved to: {combined_file}")
        
        print(f"\n🎯 Ready for R scoring!")
        print(f"Run: Rscript score_with_scoringutils.R {combined_file} results/scoringutils_scores.csv")
        
        return {
            'combined_data': combined_data,
            'combined_file': combined_file,
            'timing': {
                'data_collection': data_elapsed,
                'data_preparation': prep_elapsed,
                'total': total_elapsed
            }
        }


def main():
    """Main function to run full evaluation."""
    try:
        evaluator = ScoringutilsFullEvaluator()
        results = evaluator.run_full_evaluation()
        return results
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    results = main()
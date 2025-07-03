import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
import click
import epiframework
import nn_blocks, idplots, ddpm, myutils, inpaint_module, ground_truth
import mlflow
import mlflow.pytorch


import seaborn as sns
from tqdm.auto import tqdm

import torch
import pandas as pd

import build_dataset, training_datasets
from torch.utils.data import DataLoader
from torchvision import transforms


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



@click.command()
@click.option("-s", "--spec_id", "spec_ids", default=-1, help="ID of the model to run")
@click.option("-t", "--train", "do_training", type=bool, default=False, show_default=True,
            help="Whether to run the inpainting of just train models")
@click.option("-i", "--inpaint", "do_inpainting", type=bool, default=False, show_default=True,
            help="Whether to run the inpainting of just train models")
@click.option("-f", "--experiment_name", "experiment_name", envvar="EXPERIMENT", type=str, default='test',
            show_default=True, help="file prefix to add to identify the current set of runs.")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, default='/work/users/c/h/chadi/influpaint_res/',
            show_default=True, help="Where to write runs")
def cli(spec_ids, do_training, do_inpainting, experiment_name, outdir):
    if spec_ids == -1:
            spec_ids = list(np.arange(100))
    if not isinstance(spec_ids, list):
        spec_ids = [int(spec_ids)]
    return spec_ids, do_training, do_inpainting, experiment_name, outdir

if __name__ == '__main__':
    # standalone_mode: so click doesn't exit, see
    # https://stackoverflow.com/questions/60319832/how-to-continue-execution-of-python-script-after-evaluating-a-click-cli-function
    spec_ids, do_training, do_inpainting, experiment_name, outdir = cli(standalone_mode=False)
    season_first_year="2022"
    
    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
    

    gt1 = ground_truth.GroundTruth(season_first_year=season_first_year, 
                                    data_date=datetime.datetime(2022,10,25), 
                                    mask_date=datetime.datetime(2022,10,25),
                                    channels=channels,
                                    image_size=image_size,
                                    nogit=True #so git is not damaged.
                                )
    # gather all unet and dataset specifications
    unet_spec = epiframework.model_libary(image_size=image_size, channels=channels, epoch=epoch, device=device, batch_size=batch_size)
    dataset_spec = epiframework.dataset_library(gt1=gt1, channels=channels)

    this_spec_id = 0

    for unet_name, unet in unet_spec.items():
        for dataset_name, dataset in dataset_spec.items():
            scaling_per_channel = np.array(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])))
            transforms_spec, transform_enrich = epiframework.transform_library(scaling_per_channel=scaling_per_channel)
            for transform_name, transform in transforms_spec.items():
                for enrich_name, enrich in transform_enrich.items():
                    if this_spec_id in spec_ids:
                        model_id = f"{experiment_name}::model_{unet_name}::dataset_{dataset_name}::trans_{transform_name}::enrich_{enrich_name}"
                        dataset.add_transform(transform=transform["reg"], transform_inv=transform["inv"], transform_enrich=enrich, bypass_test=False)

                        print(f"id: {this_spec_id} >> doing {model_id}")

                        # *************** TRAINING ***************
                        if do_training:
                            # Set experiment for training
                            mlflow.set_experiment(f"{experiment_name}_training")
                            with mlflow.start_run(run_name=f"training_{model_id}") as run:
                                # Log parameters
                                mlflow.log_params({
                                    "spec_id": this_spec_id,
                                    "model_type": unet_name,
                                    "dataset": dataset_name,
                                    "transform": transform_name,
                                    "enrich": enrich_name,
                                    "epoch": epoch,
                                    "batch_size": batch_size,
                                    "image_size": image_size,
                                    "channels": channels,
                                    "season_first_year": season_first_year,
                                    "experiment_name": experiment_name,
                                    "phase": "training"
                                })
                                
                                # Log dataset info
                                mlflow.log_param("dataset_max_per_feature", str(dataset.max_per_feature))
                                mlflow.log_param("scaling_per_channel", str(scaling_per_channel.tolist()))
                                
                                model_folder = f"{outdir}{epiframework.get_git_revision_short_hash()}_{datetime.date.today()}"
                                epiframework.create_folders(model_folder)
                                
                                print(f">>> training {model_id}")
                                print(f">>> saving to {model_folder}")
                                
                                # Log model folder
                                mlflow.log_param("model_folder", model_folder)

                                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
                                unet.train(dataloader=dataloader)
                                
                                # Save checkpoint
                                checkpoint_path = f"{model_folder}/{model_id}::{epoch}.pth"
                                unet.write_train_checkpoint(save_path=checkpoint_path)
                                
                                # Log model to MLflow
                                mlflow.pytorch.log_model(unet.model, "model", registered_model_name=f"influpaint_{unet_name}")
                                
                                # Log checkpoint as artifact
                                mlflow.log_artifact(checkpoint_path, "checkpoints")

                                # Generate and log samples
                                samples = unet.sample()
                                fig, axes = plt.subplots(8, 7, figsize=(16,16), dpi=100)
                                for ipl in range(51):
                                    ax = axes.flat[ipl]
                                    for i in range(batch_size):
                                        idplots.show_tensor_image(dataset.apply_transform_inv(samples[-1][i]), ax = ax, place=ipl, multi=True)
                                
                                samples_path = f"{model_folder}/{model_id}-{epoch}::samples.pdf"
                                plt.savefig(samples_path)
                                plt.close(fig)
                                
                                # Log samples plot
                                mlflow.log_artifact(samples_path, "samples")
                                
                                print(f">>> MLflow run ID: {run.info.run_id}")
                        
                        # *************** INPAINTING ***************
                        if do_inpainting:
                            # Set experiment for inpainting
                            mlflow.set_experiment(f"{experiment_name}_inpainting")
                            with mlflow.start_run(run_name=f"inpainting_{model_id}") as run:
                                # Log parameters
                                mlflow.log_params({
                                    "spec_id": this_spec_id,
                                    "model_type": unet_name,
                                    "dataset": dataset_name,
                                    "transform": transform_name,
                                    "enrich": enrich_name,
                                    "phase": "inpainting",
                                    "experiment_name": experiment_name
                                })
                                
                                model_folder = f"/work/users/c/h/chadi/influpaint_res/3d47f4a_2023-11-07"
                                checkpoint_fn = f"{model_folder}/{model_id}::{epoch}.pth"

                                model_str = checkpoint_fn.split('/')[-1]
                                # THIS is already done
                                for part in model_str.split('::')[1:-1]:
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
                                
                                ddpm1.load_model_checkpoint(checkpoint_fn)
                                
                                # Log checkpoint info
                                mlflow.log_param("checkpoint_path", checkpoint_fn)

                                #fdates = pd.date_range("2022-11-14", "2023-05-15", freq="5W-MON")
                                #fdates = pd.DatetimeIndex(['2022-11-07','2022-11-14','2022-12-12','2023-01-09','2023-03-06'])
                                fdates = pd.date_range("2022-10-12", "2023-05-15", freq="2W-MON")
                                
                                # Log inpainting parameters
                                mlflow.log_param("n_forecast_dates", len(fdates))
                                mlflow.log_param("forecast_start_date", str(fdates[0].date()))
                                mlflow.log_param("forecast_end_date", str(fdates[-1].date()))
                                
                                inpainting_metrics = {}
                                
                                for date_idx, date in enumerate(fdates):
                                gt1 = ground_truth.GroundTruth(season_first_year="2022", 
                                                            data_date=datetime.datetime.today(), 
                                                            mask_date=date,
                                                            channels=channels,
                                                            image_size=image_size,
                                                            nogit=True
                                                            )

                                gt = dataset.apply_transform(gt1.gt_xarr.data) # data.apply_transform
                                gt_keep_mask = torch.from_numpy(gt1.gt_keep_mask).type(torch.FloatTensor).to(device)
                                gt = torch.from_numpy(gt).type(torch.FloatTensor).to(device)

                                # # ****************** REPaint ******************
                                # for resampling_steps in [1, 10]:
                                #     inpaint1 = inpaint.REpaint(ddpm=ddpm1, gt=gt, gt_keep_mask=gt_keep_mask, resampling_steps=resampling_steps)
                                # 
                                #     n_samples = batch_size
                                #     all_samples = []
                                #     for i in range(max(n_samples//batch_size,1)):
                                #         samples = inpaint1.sample_paint()
                                #         all_samples.append(samples)
                                #     fluforecasts = -1*np.ones((batch_size*max(n_samples//batch_size,1), 1, 64, 64))
                                #     for i in range(max(n_samples//batch_size,1)):
                                #         fluforecasts[i*batch_size:i*batch_size+batch_size] = all_samples[i][-1]
                                #     
                                #     fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
                                #     # compute the national quantiles, important as sum of quantiles >> quantiles of sum
                                #     forecasts_national = fluforecasts_ti.sum(axis=-1)

                                #     forecast_fn = f"{model_str.split('.')[0]}::inpaint_Repaint::resamp_{resampling_steps}"
                                #     inpaint_folder = f"{model_folder}/forecasts/{forecast_fn}"
                                #     epiframework.create_folders(inpaint_folder)

                                #     gt1.export_forecasts(fluforecasts_ti=fluforecasts_ti,
                                #                         forecasts_national=forecasts_national,
                                #                         directory=inpaint_folder,
                                #                         prefix=forecast_fn,
                                #                         forecast_date=date.date(),
                                #                         save_plot=True,
                                #                         nochecks=True)

                                    # ****************** CoPaint ******************
                                    for conf_name, conf in epiframework.copaint_config_library(ddpm1.timesteps).items():
                                        if "TT" in conf_name:
                                            print(f">>> Running CoPaint with config: {conf_name} for date: {date.date()}")
                                            
                                            sampler = O_DDIMSampler(use_timesteps=np.arange(ddpm1.timesteps), 
                                                                conf=conf,
                                                                betas=ddpm1.betas, 
                                                                model_mean_type=None,
                                                                model_var_type=None,
                                                                loss_type=None)
                                            
                                            a = sampler.p_sample_loop(model_fn=ddpm1.model, 
                                                                    shape=(batch_size, channels, image_size, image_size),
                                                                    conf=conf,
                                                                    model_kwargs={"gt": gt.repeat(batch_size, 1, 1, 1),
                                                                                    "gt_keep_mask":gt_keep_mask.repeat(batch_size, 1, 1, 1),
                                                                                    "mymodel":True, 
                                                                                }
                                                                    )
                                            fluforecasts = np.array(a['sample'].cpu())

                                            fluforecasts_ti = dataset.apply_transform_inv(fluforecasts)
                                            # compute the national quantiles, important as sum of quantiles >> quantiles of sum
                                            forecasts_national = fluforecasts_ti.sum(axis=-1)

                                            forecast_fn = f"{model_str.split('.')[0]}::inpaint_CoPaint::conf_{conf_name}"
                                            inpaint_folder = f"{model_folder}/forecasts_noTT/{forecast_fn}"
                                            epiframework.create_folders(inpaint_folder)

                                            gt1.export_forecasts(fluforecasts_ti=fluforecasts_ti,
                                                                forecasts_national=forecasts_national,
                                                                directory=inpaint_folder,
                                                                prefix=forecast_fn,
                                                                forecast_date=date.date(),
                                                                save_plot=True,
                                                                nochecks=True)
                                            
                                            # Log forecast artifacts
                                            mlflow.log_artifacts(inpaint_folder, f"forecasts/{date.date()}/{conf_name}")
                                            
                                            # Log basic forecast metrics
                                            metric_key = f"forecast_{date.date()}_{conf_name}"
                                            mlflow.log_metric(f"{metric_key}_mean_forecast", float(np.mean(fluforecasts_ti)))
                                            mlflow.log_metric(f"{metric_key}_std_forecast", float(np.std(fluforecasts_ti)))
                                            mlflow.log_metric(f"{metric_key}_national_mean", float(np.mean(forecasts_national)))
                                            
                                            inpainting_metrics[metric_key] = {
                                                "mean_forecast": float(np.mean(fluforecasts_ti)),
                                                "std_forecast": float(np.std(fluforecasts_ti)),
                                                "national_mean": float(np.mean(forecasts_national))
                                            }
                                
                                # Log summary metrics
                                mlflow.log_metric("total_forecasts_generated", len(inpainting_metrics))
                                
                                # Log overall statistics
                                if inpainting_metrics:
                                    all_means = [m["mean_forecast"] for m in inpainting_metrics.values()]
                                    all_stds = [m["std_forecast"] for m in inpainting_metrics.values()]
                                    
                                    mlflow.log_metric("overall_mean_forecast", float(np.mean(all_means)))
                                    mlflow.log_metric("overall_std_forecast", float(np.mean(all_stds)))
                                
                                print(f">>> Inpainting completed. MLflow run ID: {run.info.run_id}")

                    this_spec_id += 1

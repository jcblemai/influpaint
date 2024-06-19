import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
import click
import epiframework
import nn_blocks, idplots, ddpm, myutils, inpaint, ground_truth


import seaborn as sns
from tqdm.auto import tqdm

import torch
import pandas as pd
import nn_blocks, idplots, ddpm, myutils, inpaint, ground_truth

import data_utils, training_datasets
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
@click.option("-f", "--file_prefix", "file_prefix", envvar="FILE_PREFIX", type=str, default='test',
            show_default=True, help="file prefix to add to identify the current set of runs.")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, default='/work/users/c/h/chadi/influpaint_res/',
            show_default=True, help="Where to write runs")
def cli(spec_ids, do_training, do_inpainting, file_prefix, outdir):
    if spec_ids == -1:
            spec_ids = list(np.arange(100))
    if not isinstance(spec_ids, list):
        spec_ids = [int(spec_ids)]
    return spec_ids, do_training, do_inpainting, file_prefix, outdir

if __name__ == '__main__':
    # standalone_mode: so click doesn't exit, see
    # https://stackoverflow.com/questions/60319832/how-to-continue-execution-of-python-script-after-evaluating-a-click-cli-function
    spec_ids, do_training, do_inpainting, file_prefix, outdir = cli(standalone_mode=False)
    season_first_year="2022"
    

    gt1 = ground_truth.GroundTruth(season_first_year=season_first_year, 
                                    data_date=datetime.datetime(2022,10,25), 
                                    mask_date=datetime.datetime(2022,10,25),
                                    channels=channels,
                                    image_size=image_size,
                                    nogit=True #so git is not damaged.
                                )
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
                        model_id = f"{file_prefix}::model_{unet_name}::dataset_{dataset_name}::trans_{transform_name}::enrich_{enrich_name}"
                        dataset.add_transform(transform=transform["reg"], transform_inv=transform["inv"], transform_enrich=enrich, bypass_test=False)

                        print(f"id: {this_spec_id} >> doing {model_id}")

                        # *************** TRAINING ***************
                        if do_training:
                            model_folder = f"{outdir}{epiframework.get_git_revision_short_hash()}_{datetime.date.today()}"
                            epiframework.create_folders(model_folder)
                            
                            print(f">>> training {model_id}")
                            print(f">>> saving to {model_folder}")

                            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
                            unet.train(dataloader=dataloader)
                            unet.write_train_checkpoint(save_path=f"{model_folder}/{model_id}::{epoch}.pth")

                            samples = unet.sample()
                            fig, axes = plt.subplots(8, 7, figsize=(16,16), dpi=100)
                            for ipl in range(51):
                                ax = axes.flat[ipl]
                                for i in range(batch_size):
                                    idplots.show_tensor_image(dataset.apply_transform_inv(samples[-1][i]), ax = ax, place=ipl, multi=True)
                            plt.savefig(f"{model_folder}/{model_id}-{epoch}::samples.pdf")
                        
                        # *************** INPAINTING ***************
                        if do_inpainting:
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

                            #fdates = pd.date_range("2022-11-14", "2023-05-15", freq="5W-MON")
                            #fdates = pd.DatetimeIndex(['2022-11-07','2022-11-14','2022-12-12','2023-01-09','2023-03-06'])
                            fdates = pd.date_range("2022-10-12", "2023-05-15", freq="2W-MON")
                            for date in fdates:
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

                    this_spec_id += 1

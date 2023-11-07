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

import data_utils, data_classes
from torch.utils.data import DataLoader
from torchvision import transforms


image_size = 64
channels = 1
batch_size=512
epoch = 800

device = "cuda" if torch.cuda.is_available() else "cpu"



@click.command()
@click.option("-s", "--spec_id", "spec_ids", default=1, help="ID of the model to run")
@click.option("-i", "--inpaint", "inpaint", type=bool, default=False, show_default=True,
            help="Whether to run the inpainting of just train models")
@click.option("-f", "--file_prefix", "file_prefix", envvar="FILE_PREFIX", type=str, default='test',
            show_default=True, help="file prefix to add to identify the current set of runs.")
@click.option("-d", "--output_directory", "outdir", envvar="OCP_OUTDIR", type=str, default='/work/users/c/h/chadi/influpaint_res/',
            show_default=True, help="Where to write runs")
def cli(spec_ids, inpaint, file_prefix, outdir):
    if not isinstance(spec_ids, list):
        spec_ids = [int(spec_ids)]
    return spec_ids, inpaint, file_prefix, outdir


if __name__ == '__main__':
    # standalone_mode: so click doesn't exit, see
    # https://stackoverflow.com/questions/60319832/how-to-continue-execution-of-python-script-after-evaluating-a-click-cli-function
    spec_ids, inpaint, file_prefix, outdir = cli(standalone_mode=False)
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
                        model_folder = f"{outdir}{epiframework.get_git_revision_short_hash()}_{datetime.date.today()}"

                        print(f">>> doing {model_id}")
                        print(f">>> saving to {model_folder}")

                        epiframework.create_folders(model_folder)
                        dataset.add_transform(transform=transform["reg"], transform_inv=transform["inv"], transform_enrich=enrich, bypass_test=False)
                        
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

                    this_spec_id += 1


                    
                    
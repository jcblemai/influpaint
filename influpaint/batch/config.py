"""
Simple configuration libraries for InfluPaint research.
Easy to modify - just comment/uncomment lines or add new entries.
"""

import scipy.interpolate
import itertools
import datetime
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

from ..datasets import loaders as training_datasets
from ..models import nn_blocks, ddpm
from ..utils import plotting as idplots, helpers as myutils, ground_truth
from ..models import inpaint_module

import sys
sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler
from guided_diffusion import unet
from utils import config


def copaint_config_library(timesteps):
    """CoPaint inpainting configurations - easy to modify parameters"""
    config_lib = {
        "celebahq_try1": config.Config(default_config_dict={
            "respace_interpolate": False,
            "ddim": {
                "ddim_sigma": 0.0,
                "schedule_params": {
                    "ddpm_num_steps": timesteps,
                    "jump_length": 20,  # 10,
                    "jump_n_sample": 4,  # 2,
                    "num_inference_steps": timesteps,
                    "schedule_type": "linear",
                    "time_travel_filter_type": "none",
                    "use_timetravel": True
                }
            },
            "optimize_xt": {
                "coef_xt_reg": 0.00001,  # 0.0001,
                "coef_xt_reg_decay": 1.05,  # 1.01,
                "filter_xT": False,
                "lr_xt": 0.02,
                "lr_xt_decay": 1.012,
                "mid_interval_num": 1,
                "num_iteration_optimize_xt": 5,
                "optimize_before_time_travel": True,
                "optimize_xt": True,
                "use_adaptive_lr_xt": True,
                "use_smart_lr_xt_decay": True
            },
            "debug": False
        }, use_argparse=False),
        
        "celebahq_noTT": config.Config(default_config_dict={
            "respace_interpolate": False,
            "ddim": {
                "ddim_sigma": 0.0,
                "schedule_params": {
                    "ddpm_num_steps": timesteps,
                    "jump_length": 10,
                    "jump_n_sample": 2,
                    "num_inference_steps": timesteps,
                    "schedule_type": "linear",
                    "time_travel_filter_type": "none",
                    "use_timetravel": False,
                }
            },
            "optimize_xt": {
                "coef_xt_reg": 0.0001,
                "coef_xt_reg_decay": 1.01,
                "filter_xT": False,
                "lr_xt": 0.02,
                "lr_xt_decay": 1.012,
                "mid_interval_num": 1,
                "num_iteration_optimize_xt": 2,
                "optimize_before_time_travel": True,
                "optimize_xt": True,
                "use_adaptive_lr_xt": True,
                "use_smart_lr_xt_decay": True
            },
            "debug": False
        }, use_argparse=False),
        
        "celebahq_noTT2": config.Config(default_config_dict={
            "respace_interpolate": False,
            "ddim": {
                "ddim_sigma": 0.0,
                "schedule_params": {
                    "ddpm_num_steps": timesteps,
                    "jump_length": 10,
                    "jump_n_sample": 2,
                    "num_inference_steps": timesteps,
                    "schedule_type": "linear",
                    "time_travel_filter_type": "none",
                    "use_timetravel": False,
                }
            },
            "optimize_xt": {
                "coef_xt_reg": 0.0001,
                "coef_xt_reg_decay": 1.01,
                "filter_xT": False,
                "lr_xt": 0.02,
                "lr_xt_decay": 1.012,
                "mid_interval_num": 1,
                "num_iteration_optimize_xt": 5,
                "optimize_before_time_travel": True,
                "optimize_xt": True,
                "use_adaptive_lr_xt": True,
                "use_smart_lr_xt_decay": True
            },
            "debug": False
        }, use_argparse=False),
        
        "celebahq_try3": config.Config(default_config_dict={
            "respace_interpolate": False,
            "ddim": {
                "ddim_sigma": 0.0,
                "schedule_params": {
                    "ddpm_num_steps": timesteps,
                    "jump_length": 5,  # 10
                    "jump_n_sample": 2,
                    "num_inference_steps": timesteps,
                    "schedule_type": "linear",
                    "time_travel_filter_type": "none",
                    "use_timetravel": True
                }
            },
            "optimize_xt": {
                "coef_xt_reg": 0.0001,
                "coef_xt_reg_decay": 1.01,
                "filter_xT": False,
                "lr_xt": 0.02,
                "lr_xt_decay": 1.012,
                "mid_interval_num": 1,
                "num_iteration_optimize_xt": 5,  # 2,
                "optimize_before_time_travel": True,
                "optimize_xt": True,
                "use_adaptive_lr_xt": True,
                "use_smart_lr_xt_decay": True
            },
            "debug": False
        }, use_argparse=False),
        
        "celebahq": config.Config(default_config_dict={
            "respace_interpolate": False,
            "ddim": {
                "ddim_sigma": 0.0,
                "schedule_params": {
                    "ddpm_num_steps": timesteps,
                    "jump_length": 10,
                    "jump_n_sample": 2,
                    "num_inference_steps": timesteps,
                    "schedule_type": "linear",
                    "time_travel_filter_type": "none",
                    "use_timetravel": True
                }
            },
            "optimize_xt": {
                "coef_xt_reg": 0.0001,
                "coef_xt_reg_decay": 1.01,
                "filter_xT": False,
                "lr_xt": 0.02,
                "lr_xt_decay": 1.012,
                "mid_interval_num": 1,
                "num_iteration_optimize_xt": 2,
                "optimize_before_time_travel": True,
                "optimize_xt": True,
                "use_adaptive_lr_xt": True,
                "use_smart_lr_xt_decay": True
            },
            "debug": False
        }, use_argparse=False),
        
        # "imagenet": config.Config(default_config_dict={
        #     "respace_interpolate": False,
        #     "ddim": {
        #         "ddim_sigma": 0.0,
        #         "schedule_params": {
        #             "ddpm_num_steps": timesteps,
        #             "jump_length": 10,
        #             "jump_n_sample": 2,
        #             "num_inference_steps": 200,
        #             "schedule_type": "linear",
        #             "time_travel_filter_type": "none",
        #             "use_timetravel": True
        #         }
        #     },
        #     "optimize_xt": {
        #         "coef_xt_reg": 0.01,
        #         "coef_xt_reg_decay": 1.0,
        #         "filter_xT": False,
        #         "lr_xt": 0.02,
        #         "lr_xt_decay": 1.012,
        #         "mid_interval_num": 1,
        #         "num_iteration_optimize_xt": 2,
        #         "optimize_before_time_travel": True,
        #         "optimize_xt": True,
        #         "use_adaptive_lr_xt": True,
        #         "use_smart_lr_xt_decay": True
        #     },
        #     "debug": False
        # }, use_argparse=False),
    }
    return config_lib


def model_library(image_size, channels, epoch, device, batch_size):
    """Model configurations - easy to add new models"""
    unet_spec = {
        "MyUnet200": ddpm.DDPM(
            model=nn_blocks.Unet(
                dim=image_size,
                channels=channels,
                dim_mults=(1, 2, 4,),
                use_convnext=False
            ), 
            image_size=image_size, 
            channels=channels, 
            batch_size=batch_size, 
            epochs=epoch, 
            timesteps=200,
            device=device
        ),
        "MyUnet500": ddpm.DDPM(
            model=nn_blocks.Unet(
                dim=image_size,
                channels=channels,
                dim_mults=(1, 2, 4,),
                use_convnext=False
            ), 
            image_size=image_size, 
            channels=channels, 
            batch_size=batch_size, 
            epochs=epoch, 
            timesteps=500,
            device=device
        )
    }
    return unet_spec


def dataset_library(season_setup, channels):
    """Dataset configurations - easy to add new datasets"""
    day = "2025-07-14"
    
    dataset_spec = {
        # Legacy datasets
        # "Fv": training_datasets.FluDataset.from_fluview(season_setup=season_setup, download=False),
        #"R1Fv": training_datasets.FluDataset.from_SMHR1_fluview(season_setup=season_setup, download=False),
        #"R1": training_datasets.FluDataset.from_csp_SMHR1('Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc', channels=channels),
        
        # New DATASET_GRIDS - just comment/uncomment to enable/disable
        "SURV_ONLY": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_SURV_ONLY_{day}.nc", channels=channels),
        "HYBRID_70S_30M": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_HYBRID_70S_30M_{day}.nc", channels=channels),
        "HYBRID_30S_70M": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_HYBRID_30S_70M_{day}.nc", channels=channels),
        "MOD_ONLY": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_MOD_ONLY_{day}.nc", channels=channels),
    }
    return dataset_spec


def transform_library(scaling_per_channel):
    """Transform configurations - easy to modify parameters"""
    from torchvision import transforms
    from influpaint.datasets import transforms as epitransforms

    print(scaling_per_channel)

    transform_enrich = {
        "No": transforms.Compose([]),
        "PoisPadScale": transforms.Compose([
            transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
            transforms.Lambda(lambda t: epitransforms.transform_random_padintime(t, min_shift=-15, max_shift=15)),
            transforms.Lambda(lambda t: epitransforms.transform_randomscale(t, min=.1, max=1.9)),
        ]),
        "PoisPadScaleSmall": transforms.Compose([
            transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
            transforms.Lambda(lambda t: epitransforms.transform_random_padintime(t, min_shift=-4, max_shift=4)),
            transforms.Lambda(lambda t: epitransforms.transform_randomscale(t, min=.7, max=1.3)),
        ]),
        "Pois": transforms.Compose([
            transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
        ])
    }

    transforms_spec = {
        # No scaling (linear scale)
        "Lins": {
            "reg": transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale=1/scaling_per_channel)),
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale=2)),
            ]),
            "inv": transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale=1/scaling_per_channel)),
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale=2)),
            ][::-1])  
        },
        # sqrt scale
        "Sqrt": {
            "reg": transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale=1/scaling_per_channel)),
                epitransforms.transform_sqrt,
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale=2)),
            ]),
            "inv": transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale=1/scaling_per_channel)),
                epitransforms.transform_sqrt_inv,
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale=2)),
            ][::-1])  
        },
    }

    return transforms_spec, transform_enrich


# Available options - easy to modify by commenting/uncommenting
AVAILABLE_MODELS = ["MyUnet200", "MyUnet500"]
AVAILABLE_DATASETS = ["SURV_ONLY", "HYBRID_70S_30M", "HYBRID_30S_70M", "MOD_ONLY"] #R1Fv, R1
AVAILABLE_TRANSFORMS = ["Lins", "Sqrt"]
AVAILABLE_ENRICHMENTS = ["No", "PoisPadScale", "PoisPadScaleSmall", "Pois"]
AVAILABLE_COPAINT_CONFIGS = ["celebahq_try1", "celebahq_noTT", "celebahq_noTT2", "celebahq_try3", "celebahq"]


# Helper functions
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def create_folders(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_dataset(dataset_name, season_setup, channels):
    """Get dataset by name, handling lambda functions"""
    dataset_spec = dataset_library(season_setup, channels)
    dataset_factory = dataset_spec[dataset_name]
    
    # Handle lambda functions (for new training datasets)
    if callable(dataset_factory):
        return dataset_factory()
    else:
        return dataset_factory
"""
Factory classes for creating heavy objects from scenario specifications.
Contains all the object creation logic separated from scenario definitions.
"""

import datetime
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..datasets import loaders as training_datasets
from ..models import nn_blocks, ddpm
from ..utils import plotting as idplots, helpers as myutils, ground_truth
import inpaint_module
from .scenarios import TrainingScenarioSpec, InpaintingScenarioSpec

import sys
sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler
from guided_diffusion import unet
from utils import config


@dataclass
class TrainingRunConfig:
    """Runtime configuration for training"""
    image_size: int = 64
    channels: int = 1
    batch_size: int = 512
    epochs: int = 800
    device: str = "cuda"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": self.image_size,
            "channels": self.channels, 
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device
        }


@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    experiment_name: str
    season_setup: Any
    season_first_year: str = "2022"
    data_date: datetime.datetime = datetime.datetime(2022, 10, 25)
    mask_date: datetime.datetime = datetime.datetime(2022, 10, 25)
    output_directory: str = '/work/users/c/h/chadi/influpaint_res/'
    
    @property
    def training_experiment_name(self) -> str:
        return f"{self.experiment_name}_training"
    
    @property
    def inpainting_experiment_name(self) -> str:
        return f"{self.experiment_name}_inpainting"


def copaint_config_library(timesteps):
    """Create CoPaint configuration library"""
    config_lib = {
        "celebahq_try1": config.Config(default_config_dict={
            "respace_interpolate": False,
            "ddim": {
                "ddim_sigma": 0.0,
                "schedule_params": {
                    "ddpm_num_steps": timesteps,
                    "jump_length": 20,
                    "jump_n_sample": 4,
                    "num_inference_steps": timesteps,
                    "schedule_type": "linear",
                    "time_travel_filter_type": "none",
                    "use_timetravel": True
                }
            },
            "optimize_xt": {
                "coef_xt_reg": 0.00001,
                "coef_xt_reg_decay": 1.05,
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
                    "jump_length": 5,
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
                "num_iteration_optimize_xt": 5,
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
    }
    return config_lib


def model_library(image_size, channels, epoch, device, batch_size):
    """Create model library"""
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
    """Create dataset library"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    dataset_spec = {
        "R1Fv": training_datasets.FluDataset.from_SMHR1_fluview(season_setup=season_setup, download=False),
        "R1": training_datasets.FluDataset.from_csp_SMHR1('Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc', channels=channels),
        
        # Training datasets from DATASET_GRIDS
        "SURV_ONLY": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_SURV_ONLY_{today}.nc", channels=channels),
        "HYBRID_70S_30M": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_HYBRID_70S_30M_{today}.nc", channels=channels),
        "HYBRID_30S_70M": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_HYBRID_30S_70M_{today}.nc", channels=channels),
        "MOD_ONLY": lambda: training_datasets.FluDataset.from_xarray(f"training_datasets/TS_MOD_ONLY_{today}.nc", channels=channels),
    }
    return dataset_spec


def transform_library(scaling_per_channel):
    """Create transform library"""
    from torchvision import transforms
    import epitransforms

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


class ObjectFactory:
    """Factory for creating heavy objects from scenario specs"""
    
    @staticmethod
    def create_unet(spec: TrainingScenarioSpec, run_config: TrainingRunConfig):
        """Create unet from scenario spec"""
        unet_spec = model_library(
            image_size=run_config.image_size,
            channels=run_config.channels, 
            epoch=run_config.epochs,
            device=run_config.device,
            batch_size=run_config.batch_size
        )
        return unet_spec[spec.unet_name]
    
    @staticmethod
    def create_dataset(spec: TrainingScenarioSpec, season_setup):
        """Create dataset from scenario spec"""
        dataset_spec = dataset_library(season_setup=season_setup, channels=1)  # assume channels=1
        dataset_factory = dataset_spec[spec.dataset_name]
        
        # Handle lambda functions (for training datasets)
        if callable(dataset_factory):
            return dataset_factory()
        else:
            return dataset_factory
    
    @staticmethod
    def create_transforms(spec: TrainingScenarioSpec, dataset):
        """Create transforms from scenario spec"""
        try:
            scaling_per_channel = np.array(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])))
        except NameError:
            scaling_per_channel = np.array(dataset.max_per_feature)
        
        transforms_spec, transform_enrich = transform_library(scaling_per_channel=scaling_per_channel)
        transform = transforms_spec[spec.transform_name]
        enrich = transform_enrich[spec.enrich_name]
        return transform, enrich, scaling_per_channel
    
    @staticmethod
    def create_copaint_config(spec: InpaintingScenarioSpec, timesteps: int):
        """Create CoPaint config from scenario spec"""
        config_lib = copaint_config_library(timesteps)
        return config_lib[spec.config_name]
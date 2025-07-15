import scipy.interpolate
import itertools
import datetime
import numpy as np
import pandas as pd
import read_datasources
from ..datasets import loaders as training_datasets
from ..models import nn_blocks, ddpm
from ..utils import plotting as idplots, helpers as myutils, ground_truth
import inpaint_module
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import subprocess
from pathlib import Path

import sys
sys.path.append('CoPaint4influpaint')
from guided_diffusion import O_DDIMSampler
from guided_diffusion import unet
from utils import config


def copaint_config_library(timesteps):
    config_lib = {
        "celebahq_try1":config.Config(default_config_dict={
                            "respace_interpolate": False,
                                "ddim": {
                                    "ddim_sigma": 0.0,
                                    "schedule_params": {
                                        "ddpm_num_steps": timesteps,
                                        "jump_length": 20, #10,
                                        "jump_n_sample": 4, #2,
                                        "num_inference_steps": timesteps,
                                        "schedule_type": "linear",
                                        "time_travel_filter_type": "none",
                                        "use_timetravel": True
                                    }
                                },
                                "optimize_xt": {
                                    "coef_xt_reg": 0.00001,#0.0001,
                                    "coef_xt_reg_decay": 1.05,#1.01,
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
                            "debug":False
                        },  use_argparse=False),
        "celebahq_noTT":config.Config(default_config_dict={
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
                            "debug":False
                        },  use_argparse=False),
        "celebahq_noTT2":config.Config(default_config_dict={
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
                            "debug":False
                        },  use_argparse=False),
        "celebahq_try3":config.Config(default_config_dict={
                            "respace_interpolate": False,
                                "ddim": {
                                    "ddim_sigma": 0.0,
                                    "schedule_params": {
                                        "ddpm_num_steps": timesteps,
                                        "jump_length": 5, #10
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
                                    "num_iteration_optimize_xt": 5,#2,
                                    "optimize_before_time_travel": True,
                                    "optimize_xt": True,
                                    "use_adaptive_lr_xt": True,
                                    "use_smart_lr_xt_decay": True
                                },
                            "debug":False
                        },  use_argparse=False),
        "celebahq":config.Config(default_config_dict={
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
                            "debug":False
                        },  use_argparse=False),
        #"imagenet":config.Config(default_config_dict={
        #                    "respace_interpolate": False,
        #                        "ddim": {
        #                            "ddim_sigma": 0.0,
        #                            "schedule_params": {
        #                                "ddpm_num_steps": timesteps,
        #                                "jump_length": 10,
        #                                "jump_n_sample": 2,
        #                                "num_inference_steps": 200,
        #                                "schedule_type": "linear",
        #                                "time_travel_filter_type": "none",
        #                                "use_timetravel": True
        #                            }
        #                        },
        #                        "optimize_xt": {
        #                            "coef_xt_reg": 0.01,
        #                            "coef_xt_reg_decay": 1.0,
        #                            "filter_xT": False,
        #                            "lr_xt": 0.02,
        #                            "lr_xt_decay": 1.012,
        #                            "mid_interval_num": 1,
        #                            "num_iteration_optimize_xt": 2,
        #                            "optimize_before_time_travel": True,
        #                            "optimize_xt": True,
        #                            "use_adaptive_lr_xt": True,
        #                            "use_smart_lr_xt_decay": True
        #                        
        #                        },
        #                    "debug":False
        #                },  use_argparse=False),
    }
    return config_lib


def model_libary(image_size, channels, epoch, device, batch_size):
    unet_spec = {
        "MyUnet200": ddpm.DDPM(model=nn_blocks.Unet(
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
                    device=device),
        "MyUnet500": ddpm.DDPM(model=nn_blocks.Unet(
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
                    device=device)
    }
    return unet_spec
def dataset_library(season_setup, channels):
    dataset_spec = {
            #"Fv":training_datasets.FluDataset.from_fluview(season_setup=season_setup, download=False),
            "R1Fv": training_datasets.FluDataset.from_SMHR1_fluview(season_setup=season_setup, download=False),
            "R1": training_datasets.FluDataset.from_csp_SMHR1('Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc', channels=channels)
    }
    return dataset_spec


@dataclass(frozen=True)
class TrainingScenarioSpec:
    """Specification for a training scenario - no heavy objects, just config"""
    scenario_id: int
    unet_name: str
    dataset_name: str
    transform_name: str
    enrich_name: str
    
    @property
    def scenario_string(self) -> str:
        return f"i{self.scenario_id}::model_{self.unet_name}::dataset_{self.dataset_name}::trans_{self.transform_name}::enrich_{self.enrich_name}"
    
    @property
    def timesteps(self) -> int:
        return 200 if self.unet_name == "MyUnet200" else 500
    
    @property
    def model_key(self) -> str:
        return f"{self.unet_name}"
    
    @property 
    def dataset_key(self) -> str:
        return f"{self.dataset_name}"

@dataclass(frozen=True)
class InpaintingScenarioSpec:
    """Specification for an inpainting scenario"""
    scenario_id: int
    config_name: str
    
    @property
    def scenario_string(self) -> str:
        return f"i{self.scenario_id}::conf_{self.config_name}"

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

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def create_folders(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def transform_library(scaling_per_channel):
    from torchvision import transforms
    import epitransforms

    print(scaling_per_channel)

    transform_enrich = {
        "No":transforms.Compose([]),
        "PoisPadScale":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
                transforms.Lambda(lambda t: epitransforms.transform_random_padintime(t, min_shift = -15, max_shift = 15)),
                transforms.Lambda(lambda t: epitransforms.transform_randomscale(t, min=.1, max=1.9)),
        ]),
        "PoisPadScaleSmall":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
                transforms.Lambda(lambda t: epitransforms.transform_random_padintime(t, min_shift = -4, max_shift = 4)),
                transforms.Lambda(lambda t: epitransforms.transform_randomscale(t, min=.7, max=1.3)),
        ]),
        "Pois":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_poisson(t)),
        ])
    }

#                         transforms.Lambda(lambda t: epitransforms.transform_skewednoise(t, scale=.4, a=-1.8))

    transforms_spec = {
        # No scaling (linear scale)
        "Lins":{
            "reg":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale = 1/scaling_per_channel)),
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale = 2))  ,
        ]),
            "inv":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale = 1/scaling_per_channel)),
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale = 2)),
        ][::-1])  
        },
        # sqrt scale
        "Sqrt":{
            "reg":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale = 1/scaling_per_channel)),
                epitransforms.transform_sqrt,
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale(t, scale = 2))  ,
        ]),
            "inv":transforms.Compose([
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale = 1/scaling_per_channel)),
                epitransforms.transform_sqrt_inv,
                transforms.Lambda(lambda t: epitransforms.transform_channelwisescale_inv(t, scale = 2)),
        ][::-1])  
            },
        }

    return transforms_spec, transform_enrich


class ObjectFactory:
    """Factory for creating heavy objects from scenario specs"""
    
    @staticmethod
    def create_unet(spec: TrainingScenarioSpec, run_config: TrainingRunConfig):
        """Create unet from scenario spec"""
        unet_spec = model_libary(
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
        return dataset_spec[spec.dataset_name]
    
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

class ScenarioLibrary:
    """Library that creates scenario specifications"""
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model names"""
        return ["MyUnet200", "MyUnet500"]
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available dataset names"""
        return ["R1Fv", "R1"]
    
    @staticmethod
    def get_available_transforms() -> List[str]:
        """Get list of available transform names"""
        return ["Lins", "Sqrt"]
    
    @staticmethod
    def get_available_enrichments() -> List[str]:
        """Get list of available enrichment names"""
        return ["No", "PoisPadScale", "PoisPadScaleSmall", "Pois"]
    
    @staticmethod
    def get_available_copaint_configs() -> List[str]:
        """Get list of available CoPaint config names"""
        return ["celebahq_try1", "celebahq_noTT", "celebahq_noTT2", "celebahq_try3", "celebahq"]
    
    @staticmethod
    def get_training_scenarios() -> List[TrainingScenarioSpec]:
        """Get all training scenario specifications"""
        scenarios = []
        scn_id = 0
        
        for unet_name in ScenarioLibrary.get_available_models():
            for dataset_name in ScenarioLibrary.get_available_datasets():
                for transform_name in ScenarioLibrary.get_available_transforms():
                    for enrich_name in ScenarioLibrary.get_available_enrichments():
                        scenario = TrainingScenarioSpec(
                            scenario_id=scn_id,
                            unet_name=unet_name,
                            dataset_name=dataset_name,
                            transform_name=transform_name,
                            enrich_name=enrich_name
                        )
                        scenarios.append(scenario)
                        scn_id += 1
        
        return scenarios
    
    @staticmethod
    def get_inpainting_scenarios() -> List[InpaintingScenarioSpec]:
        """Get all inpainting scenario specifications"""
        scenarios = []
        scn_id = 0
        
        for config_name in ScenarioLibrary.get_available_copaint_configs():
            scenario = InpaintingScenarioSpec(
                scenario_id=scn_id,
                config_name=config_name
            )
            scenarios.append(scenario)
            scn_id += 1
        
        return scenarios
    
    @staticmethod
    def get_training_scenario(scenario_id: int) -> TrainingScenarioSpec:
        """Get specific training scenario by ID"""
        scenarios = ScenarioLibrary.get_training_scenarios()
        if scenario_id >= len(scenarios):
            raise ValueError(f"Scenario ID {scenario_id} out of range. Max: {len(scenarios)-1}")
        return scenarios[scenario_id]
    
    @staticmethod
    def get_inpainting_scenario(scenario_id: int) -> InpaintingScenarioSpec:
        """Get specific inpainting scenario by ID"""
        scenarios = ScenarioLibrary.get_inpainting_scenarios()
        if scenario_id >= len(scenarios):
            raise ValueError(f"Scenario ID {scenario_id} out of range. Max: {len(scenarios)-1}")
        return scenarios[scenario_id]

def create_all_ddpm_scenarios(experiment_name, image_size, channels, epoch, device, batch_size, season_setup):
    """Legacy function - creates scenarios with heavy objects for backward compatibility"""
    scenarios = []
    all_specs = ScenarioLibrary.get_training_scenarios()
    
    unet_spec = model_libary(image_size=image_size, channels=channels, epoch=epoch, device=device, batch_size=batch_size)
    dataset_spec = dataset_library(season_setup=season_setup, channels=channels)
    
    for spec in all_specs:
        # Get heavy objects
        unet = unet_spec[spec.unet_name]
        dataset = dataset_spec[spec.dataset_name]
        
        # This assumes gt1 is available globally - may need to be passed as parameter
        try:
            scaling_per_channel = np.array(max(dataset.max_per_feature, gt1.gt_xarr.max(dim=["date", "place"])))
        except NameError:
            # Fallback if gt1 not available
            scaling_per_channel = np.array(dataset.max_per_feature)
        
        transforms_spec, transform_enrich = transform_library(scaling_per_channel=scaling_per_channel)
        transform = transforms_spec[spec.transform_name]
        enrich = transform_enrich[spec.enrich_name]
        
        scenarios_config = {
            "experiment_name": experiment_name,
            "scenarios_strid": spec.scenario_string,
            "scenarios_id": spec.scenario_id,
            "unet_name": spec.unet_name,
            "dataset_name": spec.dataset_name,
            "transform_name": spec.transform_name,
            "enrich_name": spec.enrich_name,
            "unet": unet,
            "dataset": dataset,
            "transform": transform,
            "enrich": enrich,
            "scaling_per_channel": scaling_per_channel
        }
        scenarios.append(scenarios_config)
    
    print(f"Total number of DDPM scenarios: {len(scenarios)}")
    return scenarios

def create_all_inpainting_scenarios(experiment_name, n_diffusion_steps):
    """Legacy function - creates scenarios with heavy objects for backward compatibility"""
    scenarios = []
    all_specs = ScenarioLibrary.get_inpainting_scenarios()
    config_lib = copaint_config_library(n_diffusion_steps)
    
    for spec in all_specs:
        conf = config_lib[spec.config_name]
        scenarios_config = {
            "experiment_name": experiment_name,
            "scenarios_strid": spec.scenario_string,
            "conf_name": spec.config_name,
            "config": conf
        }
        scenarios.append(scenarios_config)
    
    print(f"Total number of inpainting scenarios: {len(scenarios)}")
    return scenarios

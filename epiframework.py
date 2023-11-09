import scipy.interpolate
import itertools
import datetime
import numpy as np
import pandas as pd
import data_utils, data_classes
import nn_blocks, idplots, ddpm, myutils, inpaint, ground_truth

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
def dataset_library(gt1, channels):
    dataset_spec = {
            #"Fv":data_classes.FluDataset.from_fluview(flusetup=gt1.flusetup, download=False),
            "R1Fv": data_classes.FluDataset.from_SMHR1_fluview(flusetup=gt1.flusetup, download=False),
            "R1": data_classes.FluDataset.from_csp_SMHR1('Flusight/flu-datasets/synthetic/CSP_FluSMHR1_weekly_padded_4scn.nc', channels=channels)
    }
    return dataset_spec


def get_git_revision_short_hash() -> str:
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def create_folders(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def transform_library(scaling_per_channel):
    from torchvision import transforms

    print(scaling_per_channel)

    transform_enrich = {
        "No":transforms.Compose([]),
        "PoisPadScale":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_poisson(t)),
                transforms.Lambda(lambda t: data_classes.transform_random_padintime(t, min_shift = -15, max_shift = 15)),
                transforms.Lambda(lambda t: data_classes.transform_randomscale(t, min=.1, max=1.9)),
        ]),
        "PoisPadScaleSmall":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_poisson(t)),
                transforms.Lambda(lambda t: data_classes.transform_random_padintime(t, min_shift = -4, max_shift = 4)),
                transforms.Lambda(lambda t: data_classes.transform_randomscale(t, min=.7, max=1.3)),
        ]),
        "Pois":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_poisson(t)),
        ])
    }

#                         transforms.Lambda(lambda t: data_classes.transform_skewednoise(t, scale=.4, a=-1.8))

    transforms_spec = {
        # No scaling (linear scale)
        "Lins":{
            "reg":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale(t, scale = 1/scaling_per_channel)),
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale(t, scale = 2))  ,
        ]),
            "inv":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale_inv(t, scale = 1/scaling_per_channel)),
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale_inv(t, scale = 2)),
        ][::-1])  
        },
        # sqrt scale
        "Sqrt":{
            "reg":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale(t, scale = 1/scaling_per_channel)),
                data_classes.transform_sqrt,
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale(t, scale = 2))  ,
        ]),
            "inv":transforms.Compose([
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale_inv(t, scale = 1/scaling_per_channel)),
                data_classes.transform_sqrt_inv,
                transforms.Lambda(lambda t: data_classes.transform_channelwisescale_inv(t, scale = 2)),
        ][::-1])  
            },
        }

    return transforms_spec, transform_enrich


def create_run_config(run_id, specifications):

    if setup.scale == 'Regions':
        scenarios_specs = {
            'dataset': [3, 15, 150],  # ax.set_ylim(0.05, 0.4)
            # 'vacctotalM': [2, 5, 10, 15, 20],
            'newdoseperweek': [125000, 250000, 479700, 1e6, 1.5e6, 2e6],
            'epicourse': ['U', 'L']  # 'U'
        }
    elif setup.scale == 'Provinces':
        if setup.nnodes == 107:
            scenarios_specs = {
                'vaccpermonthM': [3, 15, 150],  # ax.set_ylim(0.05, 0.4)
                # 'vacctotalM': [2, 5, 10, 15, 20],
                'newdoseperweek': [125000, 250000, 479700, 1e6, 1.5e6, 2e6],
                'epicourse': ['U', 'L']  # 'U'
            }
        elif setup.nnodes == 10:
            scenarios_specs = {
                'newdoseperweek': [125000*10, 250000*10],
                'vaccpermonthM': [14*10, 15*10],
                'epicourse': ['U']  # 'U', 'L'
            }

    # Compute all permutatios
    keys, values = zip(*scenarios_specs.items())
    permuted_specs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    specs_df = pd.DataFrame.from_dict(permuted_specs)

    if setup.nnodes == 107:
        specs_df = specs_df[((specs_df['vaccpermonthM'] == 15.0) | (specs_df['newdoseperweek'] == 479700.0))].reset_index(drop=True) # Filter out useless scenarios

    # scn_spec = permuted_specs[scn_id]
    scn_spec = specs_df.loc[scn_id]


    tot_pop = setup.pop_node.sum()
    scenario = {'name': f"{scn_spec['epicourse']}-r{int(scn_spec['vaccpermonthM'])}-t{int(scn_spec['newdoseperweek'])}-id{scn_id}",
                'newdoseperweek': scn_spec['newdoseperweek'],
                'rate_fomula': f"({scn_spec['vaccpermonthM'] * 1e6 / tot_pop / 30}*pop_nd)"
                }

 
    return scenario
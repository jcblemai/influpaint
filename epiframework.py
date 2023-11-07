import scipy.interpolate
import itertools
import datetime
import numpy as np
import pandas as pd
import data_utils, data_classes


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
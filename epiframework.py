import scipy.interpolate
import itertools
import datetime
import numpy as np
import pandas as pd


def get_git_revision_short_hash() -> str:
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def create_folders(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)


def create_run_config(run_id, specifications):

    if setup.scale == 'Regions':
        scenarios_specs = {
            'vaccpermonthM': [3, 15, 150],  # ax.set_ylim(0.05, 0.4)
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

    # Build beta scenarios:
    if scn_spec['epicourse'] == 'C':
        scenario['beta_mult'] = np.ones((setup.nnodes, setup.ndays))
    elif scn_spec['epicourse'] == 'U':
        course = scipy.interpolate.interp1d([0, 10, 40, 80, 100, 1000], [1.4, 1.2, .9,.8, 1.2, .75], kind='linear')
        course = scipy.interpolate.interp1d([0, 10, 40, 80, 100, 1000],
                                            [1.8 * .97, 1.5 * .97, .7 * .97, 1.2 * .97, 1.2 * .97, .75 * .97],
                                            kind='cubic')
        course = course(np.arange(0, setup.ndays))
        scenario['beta_mult'] = np.ones((setup.nnodes, setup.ndays)) * course
    elif scn_spec['epicourse'] == 'L':
        course = scipy.interpolate.interp1d([0, 10, 40, 80, 100, 1000], [.8, .4, 1.2, .7, .75, .75], kind='linear')
        course = scipy.interpolate.interp1d([0, 10, 40, 80, 100, 1000], [1.8, .9, .8, .68, .75, .75], kind='cubic')
        course = course(np.arange(0, setup.ndays))
        scenario['beta_mult'] = np.ones((setup.nnodes, setup.ndays)) * course
    return scenario
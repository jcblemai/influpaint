import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import pandas as pd
import xarray as xr

import build_dataset, training_datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import datetime


flusight_quantiles = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.050)), [0.975,0.99])
flusight_quantile_pairs = np.array([flusight_quantiles[:11],flusight_quantiles[12:][::-1]]).T

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def get_folders_in_directory(directory_path):
    import os
    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    return folders


def extract(a, t, x_shape):
    """
    define an `extract` function, which will allow us to extract the appropriate \\(t\\) index for a batch of indices.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cuda_mem_info():
    # print(torch.cuda.memory_summary(device=None, abbreviated=False)) is the long form
    convert_to_gb = 1024 ** 3
    return f"{torch.cuda.get_device_name(0)} -- Allocated: {torch.cuda.memory_allocated(0)/convert_to_gb:.1f}GB, Cached: {torch.cuda.memory_reserved(0)/convert_to_gb:.1f}GB -- {torch.cuda.mem_get_info()[0]/convert_to_gb:.1f}/{torch.cuda.mem_get_info()[1]/convert_to_gb:.1f} (free/total)"

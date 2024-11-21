import math
from inspect import isfunction
from functools import partial
from pathlib import Path
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
from torchvision.utils import save_image
from torch.optim import Adam
import datetime

import myutils


def plot_to_ax(array, ax=None, place=None, multi=False, channels=1):
    c = ["firebrick", "slateblue", "orange"]
    # (3, 48, 51)
    if len(array.shape) == 4:  # it's a batch
        array = array[0, :, :, :]
    if ax is None:
        ax = plt.gca()  # get current ax

    if place is None:
        array_to_plot = array.sum(axis=2)
    else:
        array_to_plot = array[:, :, place]
    for k in range(channels):
        if multi:  # several lines will be plotted:
            ax.plot(array_to_plot[k], c=c[k], lw=0.5, alpha=0.5)
        else:
            ax.plot(array_to_plot[k], c=c[k], lw=2, alpha=1)


def show_tensor_image(inv_transd_image, ax=None, place=None, multi=False):
    plot_to_ax(inv_transd_image, ax=ax, place=place, multi=multi)

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

import data_utils, data_classes
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import Adam
import datetime

import utils


class REpaint:
    def __init__(self, ddpm, gt, gt_keep_mask, resampling_steps=1):
        self.ddpm=ddpm
        self.gt = gt
        self.gt_keep_mask = gt_keep_mask
        self.resampling_steps = resampling_steps

    @torch.no_grad()
    def p_sample_paint(self, x, t, t_index):
        # timestep parameters
        betas_t = utils.extract(self.ddpm.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = utils.extract(
            self.ddpm.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = utils.extract(self.ddpm.sqrt_recip_alphas, t, x.shape)

        posterior_variance_t = utils.extract(self.ddpm.posterior_variance, t, x.shape)

        for u in range(self.resampling_steps):
            # RePaint algorithm, line 4 and 6
            if t_index == 0:
                epsilon = 0
                z = 0
            else:
                epsilon = torch.randn_like(x)
                z = torch.randn_like(x)

            # RePaint algorithm, line 5
            x_tminus1_known = (
                1 / sqrt_recip_alphas_t * self.gt
                + sqrt_one_minus_alphas_cumprod_t ** 2 * epsilon
            )

            # RePaint algorithm, line 4 and 7
            x_tminus1_unknow = (
                sqrt_recip_alphas_t
                * (x - betas_t * self.ddpm.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
                + torch.sqrt(posterior_variance_t) * z
            )

            x_tminus1 = x_tminus1_known * self.gt_keep_mask + x_tminus1_unknow * (
                1 - self.gt_keep_mask
            ) * 1 / torch.sqrt(1 - posterior_variance_t)

            # TODO This is used for debug: return the full infered dynamics.
            if t_index == 0:
                x_tminus1 = x_tminus1_unknow * 1 / torch.sqrt(1 - posterior_variance_t)

            if u < self.resampling_steps - 1 and (t > 1).all():
                # taken from q_sample:
                noise = torch.randn_like(x)
                sqrt_alphas_cumprod_t = utils.extract(
                    self.ddpm.sqrt_alphas_cumprod, t - 1, x.shape
                )
                sqrt_one_minus_alphas_cumprod_t = utils.extract(
                    self.ddpm.sqrt_one_minus_alphas_cumprod, t - 1, x.shape
                )
                x = (
                    sqrt_alphas_cumprod_t * x_tminus1
                    + sqrt_one_minus_alphas_cumprod_t * noise
                )

        return x_tminus1

    @torch.no_grad()
    def p_sample_loop_paint(self, shape):
        device = next(self.ddpm.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)  # this is x_T
        imgs = []

        # t_T = self.timesteps
        # jump_len = 3
        # jump_n_sample = 3
        # jumps = {}
        # for j in range(0, t_T - jump_len, jump_len):
        #   jumps[j] = jump_n_sample - 1
        # t = t_T
        # ts = []
        # while t >= 1:
        #   t = t-1
        #   ts.append(t)
        #   if jumps.get(t, 0) > 0:
        #    jumps[t] = jumps[t] - 1
        #    for _ in range(jump_len):
        #       t=t+1
        #       ts.append(t)
        # ts.append(-1)
        # for i in tqdm(ts, desc='sampling loop time step', total=self.timesteps):

        for i in tqdm(
            reversed(range(0, self.ddpm.timesteps)),
            desc="sampling loop time step",
            total=self.ddpm.timesteps,
        ):
            img = self.p_sample_paint(
                x=img,
                t=torch.full(
                    (b,), i, device=device, dtype=torch.long
                ),  # an array of size "self.batch_size" containting i
                t_index=i,
            )
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample_paint(self):
        return self.p_sample_loop_paint(
            shape=(self.ddpm.batch_size, self.ddpm.channels, self.ddpm.image_size, self.ddpm.image_size),
        )


# schedule with J
# t_T = ddpm1.timesteps
# jump_len = 5 
# jump_n_sample = 5 
# jumps = {} 
# for j in range(0, t_T - jump_len, jump_len): 
#   jumps[j] = jump_n_sample - 1 
# t = t_T 
# ts = []
# while t >= 1: 
#   t = t-1
#   ts.append(t) 
#   if jumps.get(t, 0) > 0: 
#    jumps[t] = jumps[t] - 1 
#    for _ in range(jump_len): 
#      t=t+1 
#      ts.append(t)
# ts.append(-1)
# plt.plot(ts) 
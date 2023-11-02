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

import myutils


class DDPM:
    def __init__(self, model, image_size=64, channels=1, batch_size=512, epochs=500,  timesteps=200, loss_type="huber", device=None) -> None:
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size # 256 * max(1, torch.cuda.device_count())

        self.epochs = epochs

        self.timesteps = timesteps
        self.loss_type=loss_type

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=self.timesteps)
        # self.betas = quadratic_beta_schedule(self.timesteps=self.timesteps)
        # self.betas = sigmoid_beta_schedule(self.timesteps=self.timesteps)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.results_folder = Path("./results")
        self.results_folder.mkdir(exist_ok=True)
        self.save_and_sample_every = 1000



        
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def q_sample(self, x_start, t, noise=None):
        """ Forward diffusion """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = myutils.extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = myutils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = myutils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = myutils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = myutils.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = myutils.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), i
            )
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self):
        return self.p_sample_loop(
            shape=(self.batch_size, self.channels, self.image_size, self.image_size),
        )

    def train(self, dataloader):
        print(f"/!\ training on {self.device}")
        if torch.cuda.device_count() > 1:
            print(" -- using dataparallel")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        if self.device == "cuda":
            print(myutils.cuda_mem_info())

        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        losses = []

        for epoch in range(self.epochs):
            for step, batch in enumerate(dataloader):
                self.optimizer.zero_grad()

                # self.batch_size = batch["pixel_values"].shape[0]
                # batch = batch["pixel_values"].to(self.device)
                self.batch_size = batch.shape[0]
                batch = batch.to(self.device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                # Important to have a number of epoch sufficiently large to see all the self.timesteps
                t = torch.randint(
                    0, self.timesteps, (self.batch_size,), device=self.device
                ).long()

                loss = self.p_losses(
                    denoise_model=self.model, x_start=batch, t=t, loss_type=self.loss_type
                )  # loss_type="l2")#

                if step % 100 == 0:
                    print(f"Epoch: {epoch:<4} -- Loss: {loss.item()}")
                # if self.device == "cuda":
                #    print(f"   -- {helpers.cuda_mem_info()}")
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                if epoch % 50 == 0 and epoch > 0 and step == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(6, 2), dpi=100)
                    axes.flat[0].plot(np.arange(len(losses)), np.array(losses))
                    axes.flat[1].plot(
                        np.arange(len(losses[-100:])), np.array(losses[-100:])
                    )
                    axes.flat[2].plot(
                        np.arange(len(losses[-50:])), np.array(losses[-50:])
                    )
                    plt.show()

                # save generated images
                if step != 0 and step % self.save_and_sample_every == 0:
                    milestone = step // self.save_and_sample_every
                    batches = myutils.num_to_groups(4, self.batch_size)
                    all_images_list = list(
                        map(
                            lambda n: self.sample(
                                self.model, batch_size=n, channels=self.channels
                            ),
                            batches,
                        )
                    )
                    all_images = torch.cat(all_images_list, dim=0)
                    all_images = (all_images + 1) * 0.5
                    save_image(
                        all_images,
                        str(self.results_folder / f"sample-{milestone}.png"),
                        nrow=6,
                    )

            # scheduler1.step()

    def write_train_checkpoint(self):
        save_path = f"checkpoint-{self.epoch}.pth"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            save_path,
        )
        return save_path

    def load_model_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        self.model.eval()
        # necessary ????
        self.model.train()

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(
            x_start=x_start, t=t, noise=noise
        )  # forward diffusion of the dataset image
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


# ## Defining the forward diffusion process
# The forward diffusion process gradually adds noise to an image from the real distribution, in a number of time steps $T$. This happens according to a **variance schedule**. The original DDPM authors employed a linear schedule:
#
# > We set the forward process variances to constants
# increasing linearly from $\beta_1 = 10^{−4}$
# to $\beta_T = 0.02$.
#
# However, it was shown in ([Nichol et al., 2021](https://arxiv.org/abs/2102.09672)) that better results can be achieved when employing a cosine schedule.
#
# Below, we define various schedules for the $T$ self.timesteps, as well as corresponding variables which we'll need, such as cumulative variances.
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    # beta_start = 0.00001
    # beta_end = 0.001
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

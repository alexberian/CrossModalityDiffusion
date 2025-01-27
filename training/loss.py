# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from genvs.utils import utils
from torchvision import transforms

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.add_noise = transforms.RandomApply([utils.AddGaussianNoise(0.0)], p=0.5) 

    def __call__(self, net, data, device, labels=None, augment_pipe=None):
        # 0. randomly choose number of input views to use
        num_input_views = torch.randint(1, 3+1, (1,)).item()  # 1 to 3

        # 1. format data
        focal = data["focal"][0].item()
        z_near = data["z_near"][0]
        z_far = data["z_far"][0]
        images = data["images"]  # (batch_size, imgs_per_scene, C, H, W)
        assert len(images.shape) == 5
        poses = data["poses"]  # (batch_size, imgs_per_scene, 4, 4)
        input_imgs, input_poses = utils.pick_random_views(num_input_views, images, poses)
        target_imgs, target_poses = utils.pick_random_views(1, images, poses)
        input_imgs, input_poses = input_imgs.to(device), input_poses.to(device)
        target_imgs, target_poses = target_imgs.to(device), target_poses.to(device)

        # 1.5. with probability 0.5, add noise to input_imgs
        for i in range(input_imgs.shape[0]):
            for j in range(input_imgs.shape[1]):
                input_imgs[i, j] = self.add_noise(input_imgs[i, j])

        # 2. GeNVS forward pass
        assert augment_pipe is not None
        D_yn, feature_images, yn, weight, target_imgs = net(  # need to return target_imgs due to augmentation inside net
                input_imgs, input_poses, target_imgs, target_poses,
                focal, z_near, z_far,
                P_mean=-1.0, P_std=1.4, sigma_data=0.5, augment_pipe=augment_pipe)

        # 3. calculate loss
        loss = weight * ((D_yn - target_imgs) ** 2)

        return loss, feature_images, D_yn, yn, input_imgs, target_imgs

#----------------------------------------------------------------------------

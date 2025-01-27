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
import numpy as np

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
        # 1. format data
        focal = data["focal"][0].item()
        z_near = data["z_near"][0]
        z_far = data["z_far"][0]
        input_imgs  = data["input_images"]  # (batch_size, imgs_per_scene, C, H, W)
        target_imgs  = data["target_images"].unsqueeze(1)  # (batch_size, 1, C, H, W)
        input_domain_info = data["input_domain_info"] # a (batch_size, imgs_per_scene) size list of dictionaries of domain information
        target_domain_info = data["target_domain_info"] # a (batch_size,) size list of dictionaries of domain information
        input_poses = data["input_poses"]  # (batch_size, imgs_per_scene, 4, 4)
        target_poses = data["target_poses"].unsqueeze(1)  # (batch_size, 1, 4, 4)

        # 1.1. place data on device
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
                P_mean=-1.0, P_std=1.4, sigma_data=0.5, augment_pipe=augment_pipe,
                input_domain_info=input_domain_info, target_domain_info=target_domain_info)

        # 3. calculate loss
        loss = weight * ((D_yn - target_imgs) ** 2)

        return loss, feature_images, D_yn, yn, input_imgs, target_imgs

#----------------------------------------------------------------------------
# Cross-domain version of EDM loss.

@persistence.persistent_class
class CrossDomainEDMLoss:
    """
    Written by Alex Berian 2024

    This loss function is a cross-domain version of the EDM loss function.

    This version does not decide which images are input and which are target, 
    it assumes the data loader has already done that.  The primary different 
    is that for each source/target there is a domain info dictionary that is 
    passed to the model. The model will decide what to do with that dictionary.
    """
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.add_noise = transforms.RandomApply([utils.AddGaussianNoise(0.0)], p=0.5) 

    def __call__(self, net, data, device, labels=None, augment_pipe=None):
        # 1. format data
        focal = data["focal"][0].item()
        z_near = data["z_near"][0]
        z_far = data["z_far"][0]
        input_imgs  = data["input_images"]  # (batch_size, imgs_per_scene, C, H, W)
        target_imgs  = data["target_images"].unsqueeze(1)  # (batch_size, 1, C, H, W)
        input_domain_info = data["input_domain_info"] # a (batch_size, imgs_per_scene) size list of dictionaries of domain information
        target_domain_info = data["target_domain_info"] # a (batch_size,) size list of dictionaries of domain information
        input_poses = data["input_poses"]  # (batch_size, imgs_per_scene, 4, 4)
        target_poses = data["target_poses"].unsqueeze(1)  # (batch_size, 1, 4, 4)

        # 1.1. place data on device
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
                P_mean=-1.0, P_std=1.4, sigma_data=0.5, augment_pipe=augment_pipe,
                input_domain_info=input_domain_info, target_domain_info=target_domain_info)

        # 3. calculate loss
        loss = weight * ((D_yn - target_imgs) ** 2)

        return loss, feature_images, D_yn, yn, input_imgs, target_imgs


    @staticmethod
    def _test():
        # create the loss object
        loss_obj = CrossDomainEDMLoss()

        # create dummy inputs for the loss
        class DummyNet():
            # D_yn, feature_images, yn, weight, target_imgs = net(  # need to return target_imgs due to augmentation inside net
            #         input_imgs, input_poses, target_imgs, target_poses,
            def __call__(self, input_imgs, input_poses, target_imgs, target_poses, focal, z_near, z_far, **kwargs):
                return target_imgs, None, None, 1.0, target_imgs

        dummy_net = DummyNet()

        device = torch.device("cuda")
        
        data = {
            "focal": torch.tensor([1.0]),
            "z_near": torch.tensor([0.1]),
            "z_far": torch.tensor([100.0]),
            "input_images": torch.randn(2, 3, 3, 64, 64).to(device),
            "target_images": torch.randn(2, 3, 64, 64).to(device),
            "input_domain_info": [[{"domain": "A"}]*3]*2,
            "target_domain_info": [{"domain": "B"}]*2,
            "input_poses": torch.randn(2, 3, 4, 4).to(device),
            "target_poses": torch.randn(2, 4, 4).to(device),
            # input_imgs  = data["input_images"]  # (batch_size, imgs_per_scene, C, H, W)
            # target_imgs  = data["target_images"]  # (batch_size, C, H, W)
            # input_domain_info = data["input_domain_info"] # a (batch_size, imgs_per_scene) size list of dictionaries of domain information
            # target_domain_info = data["target_domain_info"] # a (batch_size,) size list of dictionaries of domain information
            # input_poses = data["input_poses"]  # (batch_size, imgs_per_scene, 4, 4)
            # target_poses = data["target_poses"]  # (batch_size, 4, 4)
        }
        augment_pipe = lambda x: x
        loss = loss_obj(dummy_net, data, device, augment_pipe=augment_pipe)
        assert len(loss) == 6   

        print("CrossDomainEDMLoss tests passed.")


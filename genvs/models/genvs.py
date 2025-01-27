# written by JhihYang Wu <jhihyangwu@arizona.edu>
# model class for GeNVS's pipeline

import torch
import torch.nn as nn
from genvs.models.pixel_nerf_net import PixelNeRFNet

class GeNVS(nn.Module):
    """
    Model class for GeNVS.
    """

    def __init__(self, denoiser, cam_weighting_method="baseline_mean"):
        """
        Constructor for GeNVS class.

        Args:
            denoiser (nn.Module): diffusion model constructed by edm
        """
        super(GeNVS, self).__init__()
        # define model parameters (pixelNeRF network, denoiser)
        self.pixel_nerf_net = PixelNeRFNet(cam_weighting_method=cam_weighting_method)
        self.denoiser = denoiser

    def forward(self, input_imgs, input_poses, target_imgs, target_poses,
                focal, z_near, z_far,
                P_mean, P_std, sigma_data, augment_pipe,
                ):
        """
        Forward pass of GeNVS pipeline.

        Args:
            input_imgs (tensor): .shape=(num_scenes, num_input_views, 3, H, W)
            input_poses (tensor): .shape=(num_scenes, num_input_views, 4, 4)
            target_imgs (tensor): needed for training diffusion model .shape=(num_scenes, num_target_views, 3, H, W)
            target_poses (tensor): .shape=(num_scenes, num_target_views, 4, 4)
            focal (float): normalized focal length of camera
            z_near (float): z near to use when generating sample rays for vol rendering
            z_far (float): z far to use when generating sample rays for vol rendering
            P_mean (float): edm hyperparameter
            P_std (float): edm hyperparameter
            sigma_data (float): edm hyperparameter
            augment_pipe (obj): augmentation pipeline
        
        Returns:
            D_yn (tensor): denoised predicted novel views .shape=(num_scenes, num_target_views, 3, H, W)
            feature_images (tensor): internal feature img used to cond diffusion model .shape=(num_scenes, num_target_views, C, H, W)
            yn (tensor): noisy novel view passed into diffusion model .shape=(num_scenes, num_target_views, 3, H, W)
            weight (tensor): used for scaling loss later
        """
        # 1. encode input views
        inputs_encoded = self.pixel_nerf_net.encode_input_views(input_imgs, input_poses)

        # 2. render feature images at target poses
        feature_images = self.pixel_nerf_net(inputs_encoded, target_poses,
                                        focal, z_near, z_far)  # (num_scenes, num_target_views, C, H, W)

        # 2.1. reshapes
        num_scenes, num_target_views, feat_dim, H, W = feature_images.shape
        feature_images = feature_images.reshape(-1, feat_dim, H, W)
        target_imgs = target_imgs.reshape(-1, 3, H, W)

        # 3. prepare diffusion model input https://github.com/NVlabs/edm/blob/main/training/loss.py#L73
        rnd_normal = torch.randn([target_imgs.shape[0], 1, 1, 1], device=target_imgs.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        # generate noise to add to the first 3 channels.
        noise = torch.randn_like(target_imgs) * sigma
        # get noisy target image y
        yn = target_imgs + noise
        # augment the images
        yn, feature_images, target_imgs, _ = augment_pipe(yn, feature_images, target_imgs)
        if torch.rand(1) <= 0.1:  # feature image dropout
            feature_images = torch.randn_like(feature_images)
        # concatenate noisy target image y with feature image
        # F_concat_y: (num_scenes * num_target_views, 3 + C, H, W)
        F_concat_yn = torch.cat([yn, feature_images], dim=1)

        # 4. denoise F_concat_yn
        D_yn = self.denoiser(F_concat_yn, sigma, None)

        # reshapes
        D_yn = D_yn.reshape(num_scenes, num_target_views, 3, H, W)
        feature_images = feature_images.reshape(num_scenes, num_target_views, feat_dim, H, W)
        yn = yn.reshape(num_scenes, num_target_views, 3, H, W)
        weight = weight.reshape(num_scenes, num_target_views, 1, 1, 1)
        target_imgs = target_imgs.reshape(num_scenes, num_target_views, 3, H, W)
        return D_yn, feature_images, yn, weight, target_imgs

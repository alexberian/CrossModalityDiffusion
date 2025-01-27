# written by JhihYang Wu <jhihyangwu@arizona.edu>
# model class for training any domain in and any domain out by jointing training multiple GeNVS models

import torch
import torch.nn as nn
from genvs.models.genvs import GeNVS
import copy

class JointGeNVS(nn.Module):
    """
    Model class for jointing training multiple GeNVS models
    """

    def __init__(self,
                 denoiser,
                 domains=["rgb", "sonar", "lidar_depth", "raysar"],
                 cam_weighting_method="baseline_mean"):
        """
        Constructor for JointGeNVS class.

        Args:
            denoiser (nn.Module): diffusion model constructed by edm
            domains (list): list of domain folders
            cam_weighting_method (str): method for weighting camera features
        """
        super(JointGeNVS, self).__init__()
        self.domains = domains        

        # create GeNVS models for each domain
        self.models = {}
        for domain in domains:
            self.models[domain] = GeNVS(copy.deepcopy(denoiser), cam_weighting_method)
            self.__setattr__("genvs_" + domain, self.models[domain])  # needed for torch to search for submodules
    
    def forward(self, input_imgs, input_poses, target_imgs, target_poses,
                focal, z_near, z_far,
                P_mean, P_std, sigma_data, augment_pipe,
                input_domain_info=None, target_domain_info=None,
                **kwargs
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
            input_domain_info (list): a (num_scenes, num_input_views) size list of dictionaries of domain information
            target_domain_info (list): a (num_scenes,) size list of dictionaries of domain information
        
        Returns:
            D_yn (tensor): denoised predicted novel views .shape=(num_scenes, num_target_views, 3, H, W)
            feature_images (tensor): internal feature img used to cond diffusion model .shape=(num_scenes, num_target_views, C, H, W)
            yn (tensor): noisy novel view passed into diffusion model .shape=(num_scenes, num_target_views, 3, H, W)
            weight (tensor): used for scaling loss later
        """

        # 1. encode input views
        num_scenes, num_input_views = len(input_domain_info), len(input_domain_info[0])
        all_inputs_encoded = []
        for i in range(num_scenes):
            tmp_encodings = []
            for j in range(num_input_views):
                domain = input_domain_info[i][j]["domain"]
                enc, _ = self.models[domain].pixel_nerf_net.encode_input_views(input_imgs[i, j][None][None],
                                                                               None)
                tmp_encodings.append(enc)
            tmp_encodings = torch.cat(tmp_encodings, dim=1)  # (1, num_input_views, feat_dim, num_depths, H, W)
            tmp_poses = input_poses[i][None]  # (1, num_input_views, 4, 4)
            all_inputs_encoded.append((tmp_encodings, tmp_poses))
        
        # 2. render feature images at target poses
        assert target_imgs.shape[1] == 1
        range_angle_feat_img = {
            "rgb": False,
            "sonar": True,
            "lidar_depth": False,
            "raysar": False
        }
        feature_images = []
        for i in range(num_scenes):
            inputs_encoded = all_inputs_encoded[i]
            target_domain = target_domain_info[i]["domain"]
            feat_img = self.models["rgb"].pixel_nerf_net(inputs_encoded, target_poses[i][None],
                                                focal, z_near, z_far, range_angle_feat_img[target_domain])  # (1, num_target_views, C, H, W)
            feature_images.append(feat_img)
        feature_images = torch.cat(feature_images, dim=0)  # (num_scenes, num_target_views, C, H, W)

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
        assert num_target_views == 1
        D_yn = []
        for i in range(num_scenes):
            domain = target_domain_info[i]["domain"]
            D_yn.append(self.models[domain].denoiser(F_concat_yn[i][None], sigma[i][None], None))
        D_yn = torch.cat(D_yn, dim=0)  # (num_scenes * num_target_views, 3, H, W)
                                       # num_target_views must be 1

        # reshapes
        D_yn = D_yn.reshape(num_scenes, num_target_views, 3, H, W)
        feature_images = feature_images.reshape(num_scenes, num_target_views, feat_dim, H, W)
        yn = yn.reshape(num_scenes, num_target_views, 3, H, W)
        weight = weight.reshape(num_scenes, num_target_views, 1, 1, 1)
        target_imgs = target_imgs.reshape(num_scenes, num_target_views, 3, H, W)
        return D_yn, feature_images, yn, weight, target_imgs

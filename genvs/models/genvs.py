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
        # define model parameters (pixelNeRF network, denoiser)
        super(GeNVS, self).__init__()
        self.pixel_nerf_net = PixelNeRFNet(cam_weighting_method=cam_weighting_method)
        self.denoiser = denoiser

    def forward(self, input_imgs, input_poses, target_imgs, target_poses,
                focal, z_near, z_far,
                P_mean, P_std, sigma_data, augment_pipe, **kwargs
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
        range_angle_feat_img = False  # TODO: make this change depending if we are rendering sonar or rgb
        feature_images = self.pixel_nerf_net(inputs_encoded, target_poses,
                                        focal, z_near, z_far, range_angle_feat_img)  # (num_scenes, num_target_views, C, H, W)

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





class CrossDomainGeNVS(nn.Module):
    """
    Written by Alex Berian 2024

    A cross domain version of GeNVS that takes in domain information. 
    If domain information is not provided, it will default to the normal GeNVS.
    """
    def __init__(self, denoiser, cam_weighting_method="baseline_mean", domain_dim=64):
        """
        Constructor for GeNVS class.

        Args:
            denoiser (nn.Module): diffusion model constructed by edm
        """
        super(CrossDomainGeNVS, self).__init__()
        self.domain_dim = domain_dim
        self.init_domain_embedding(domain_dim)
        self.pixel_nerf_net = PixelNeRFNet(
                cam_weighting_method = cam_weighting_method,
                use_domain_embedding = True,
                domain_dim = domain_dim,
        )
        self.denoiser = denoiser

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
        # 0. Default case
        if input_domain_info is None or target_domain_info is None:
            return super().forward(input_imgs, input_poses, target_imgs, target_poses,
                focal, z_near, z_far,
                P_mean, P_std, sigma_data, augment_pipe,
                **kwargs,
            )

        # 0.1 domain embedding
        input_domain_emb = []
        assert(len(input_domain_info) == input_imgs.shape[0]), "input_domain_info and input_imgs must have the same number of scenes"
        for i in range(len(input_domain_info)):
            current_batch = []
            assert(len(input_domain_info[i]) == input_imgs.shape[1]), "input_domain_info and input_imgs must have the same number of views"
            for j in range(len(input_domain_info[i])):
                current_batch.append(self.embed_domain_info(input_domain_info[i][j], input_imgs.device))
            input_domain_emb.append( torch.stack(current_batch, dim=0) )
        input_domain_emb = torch.stack(input_domain_emb, dim=0)
        target_domain_emb = []
        assert(len(target_domain_info) == target_imgs.shape[0]), "target_domain_info and target_imgs must have the same number of scenes"
        for i in range(len(target_domain_info)):
            target_domain_emb.append(self.embed_domain_info(target_domain_info[i], input_imgs.device))
        target_domain_emb = torch.stack(target_domain_emb, dim=0)

        # 1. encode input views
        inputs_encoded = self.pixel_nerf_net.encode_input_views(input_imgs, input_poses, domain_embeddings=input_domain_emb)

        # 2. render feature images at target poses
        range_angle_feat_img = False  # TODO: make this change depending if we are rendering sonar or rgb
        feature_images = self.pixel_nerf_net(inputs_encoded, target_poses,
                                        focal, z_near, z_far, range_angle_feat_img)  # (num_scenes, num_target_views, C, H, W)

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
        D_yn = self.denoiser(F_concat_yn, sigma, None, domain_embedding=target_domain_emb)

        # reshapes
        D_yn = D_yn.reshape(num_scenes, num_target_views, 3, H, W)
        feature_images = feature_images.reshape(num_scenes, num_target_views, feat_dim, H, W)
        yn = yn.reshape(num_scenes, num_target_views, 3, H, W)
        weight = weight.reshape(num_scenes, num_target_views, 1, 1, 1)
        target_imgs = target_imgs.reshape(num_scenes, num_target_views, 3, H, W)
        return D_yn, feature_images, yn, weight, target_imgs


    def embed_domain_info(self, domain_info, device):
        """
        Embed domain information into a tensor.

        Args:
            domain_info (dictionary): a (num_scenes, num_views) size list of dictionaries of domain information
        
        Returns:
            domain_emb (tensor): .shape=(embedding_size,)
        """

        # rgb embedding
        if domain_info["domain"] == "rgb":
            domain_enc = torch.tensor([1, 0, 0], dtype=torch.float32).to(device)
            embedding = self.domain_embedding(domain_enc)

        # sonar embedding
            # Water depth (m)
            # Salinity (ppt)
            # Temperature (°C)
            # Sonar frequency (kHz)
            # Directivity index DI (dB)
        elif domain_info["domain"] == "sonar":
            domain_enc = torch.tensor([0, 1, 0], dtype=torch.float32).to(device)

            domain_info_enc = [0,0,0,0,0]
            if "water_depth" in domain_info:
                domain_info_enc[0] = domain_info["water_depth"]
            if "salinity" in domain_info:
                domain_info_enc[1] = domain_info["salinity"]
            if "temperature" in domain_info:
                domain_info_enc[2] = domain_info["temperature"]
            if "sonar_frequency" in domain_info:
                domain_info_enc[3] = domain_info["sonar_frequency"]
            if "directivity_index" in domain_info:
                domain_info_enc[4] = domain_info["directivity_index"]
            domain_info_enc = torch.tensor(domain_info_enc, dtype=torch.float32).to(device)

            embedding = self.domain_embedding(domain_enc) + self.sonar_embedding(domain_info_enc)

        # sar embedding
            # band (k,x,c,s)
            # range resolution (m)
        elif domain_info["domain"] == "sar":
            domain_enc = torch.tensor([0, 0, 1], dtype=torch.float32).to(device)

            bands = ["k", "x", "c", "s"]
            domain_info_enc = [0,0,0,0,0]
            if "band" in domain_info:
                domain_info_enc[bands.index(domain_info["band"])] = 1
            if "range_resolution" in domain_info:
                domain_info_enc[4] = domain_info["range_resolution"]
            domain_info_enc = torch.tensor(domain_info_enc, dtype=torch.float32).to(device)

            embedding = self.domain_embedding(domain_enc) + self.sar_embedding(domain_info_enc)

        # default embedding
        else:
            embedding = torch.zeros(self.domain_dim, dtype=torch.float32).to(device)

        return embedding


    def init_domain_embedding(self,domain_dim):
        """
        Initialize the domain embedding layer.
        """
        self.domain_embedding = nn.Sequential(
            nn.Linear(3, domain_dim),
            nn.ReLU(),
        )

        # sonar domain info
        # Water depth (m)
        # Salinity (ppt)
        # Temperature (°C)
        # Sonar frequency (kHz)
        # Directivity index DI (dB)
        self.sonar_embedding = nn.Sequential(
            nn.Linear(5, domain_dim),
            nn.ReLU(),
        )

        # sar domain info
        # band (k,x,c,s)
        # range resolution (m)
        self.sar_embedding = nn.Sequential(
            nn.Linear(5, domain_dim),
            nn.ReLU(),
        )

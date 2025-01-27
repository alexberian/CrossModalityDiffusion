# written by JhihYang Wu <jhihyangwu@arizona.edu> and colleagues
# model class for something like pixelNeRF's pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from genvs.models.genvs_encoder import GeNVSEncoder
from genvs.utils import utils
from genvs.models.cam_weighting import create_cam_weighting_object

class PixelNeRFNet(nn.Module):
    """
    Our model class for something like pixelNeRF pipeline.
    """

    def __init__(self, cam_weighting_method="baseline_mean"):
        """
        Constructor for PixelNeRFNet class.
        """
        super(PixelNeRFNet, self).__init__()

        # define model parameters (GeNVS encoder, radiance field MLP)
        self.encoder = GeNVSEncoder()
        self.mlp = MLP()

        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)

        self.cam_weighter = create_cam_weighting_object(cam_weighting_method)

    def forward(self, inputs_encoded, target_poses,
                focal, z_near, z_far, samples_per_ray=64):
        """
        Forward pass for pixelNeRF.

        Args: 
            inputs_encoded (tuple of 2 tensors): output from calling encode_input_views
            target_poses (tensor): novel poses to try to predict .shape=(num_scenes, num_target_views, 4, 4)
            focal (float): normalized focal length of camera
            z_near (float): z near to use when generating sample rays for vol rendering
            z_far (float): z far to use when generating sample rays for vol rendering
            samples_per_ray (int): number of samples along ray to sample from z_near to z_far for vol rendering
        
        Returns:
            renders (tensor): predicted novel view .shape=(num_scenes, num_target_views, num_channels, H, W)
        """
        # for efficiency, only shoot (H//2, W//2) rays per novel view and then upsample to (H, W) like GeNVS
        _, num_input_views, feat_dim, num_depths, H, W = inputs_encoded[0].shape
        num_scenes, num_target_views, _, _ = target_poses.shape

        # generate points in 3D space to sample for latents later
        rays = utils.gen_rays(target_poses.reshape(num_scenes*num_target_views, 4, 4),
                              H//2, W//2, focal, device=target_poses.device)  # (num_scenes * num_target_views, H//2, W//2, 6)
        pts, times = utils.gen_pts_from_rays(rays, z_near, z_far, samples_per_ray)  # (num_scenes * num_target_views, H//2, W//2, samples_per_ray, 3)
        times = times.reshape(num_scenes, num_target_views, H//2, W//2, samples_per_ray)
        pts = pts.reshape(num_scenes, 1, num_target_views, H//2, W//2, samples_per_ray, 3, 1)
        pts = torch.cat((pts, torch.ones_like(pts)[..., :1, :]), dim=-2)  # convert to homogenous coords (..., 4, 1)

        # project the points in 3D space to local coordinates 
        encoded_imgs, input_poses = inputs_encoded
        assert input_poses.device == target_poses.device
        extrinsics = utils.invert_pose_mats(input_poses)  # (num_scenes, num_input_views, 4, 4)
        extrinsics = extrinsics.reshape(num_scenes, num_input_views, 1, 1, 1, 1, 4, 4)
        local_pts = extrinsics @ pts  # (num_scenes, num_input_views, num_target_views, H//2, W//2, samples_per_ray, 4, 1)

        # convert the pts to uv coordinates
        # most values in u v w should be between -1 and +1 but can be outside
        x = local_pts[..., 0, 0]
        y = local_pts[..., 1, 0]
        z = local_pts[..., 2, 0]
        z = -z  # flip z to make it positive and easier to think about
        u = x / z * focal  # normalized focal length away means on -1 +1 image plane
        v = y / z * focal
        min_z, max_z = focal * z_near, focal * z_far
        w = 2 * (z - min_z) / (max_z - min_z) - 1  # most values should be between -1 and +1

        uvw = torch.stack((u, -v, -w), dim=-1)  # -v because "x = -1, y = -1 is the left-top pixel of input" https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        assert uvw.shape == (num_scenes, num_input_views, num_target_views, H//2, W//2, samples_per_ray, 3)

        # sample the features from encoded_imgs
        feats = F.grid_sample(encoded_imgs.reshape(num_scenes * num_input_views, feat_dim, num_depths, H, W),
                              uvw.reshape(num_scenes * num_input_views, 1, 1, -1, 3),
                              mode="bilinear",  # will automatically become trilinear due to 5D input shape
                              padding_mode="zeros",
                              align_corners=False)
        feats = feats.reshape(num_scenes, num_input_views, feat_dim, num_target_views, H//2, W//2, samples_per_ray)
        
        # combine feats across input views
        weights = self.cam_weighter(input_poses, target_poses).unsqueeze(2)  # (num_scenes, num_input_views, 1, num_target_views)
        feats = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * feats
        feats = torch.sum(feats, dim=1)  # (num_scenes, feat_dim, num_target_views, H//2, W//2, samples_per_ray)
        feats = torch.permute(feats, (0, 2, 3, 4, 5, 1))  # (num_scenes, num_target_views, H//2, W//2, samples_per_ray, feat_dim)

        # pass through NeRF MLP
        new_feats = []
        for i in range(num_scenes):
            buffer = []
            for j in range(num_target_views):
                buffer.append(self.mlp(feats[i, j]))
            new_feats.append(torch.stack(buffer, dim=0))
        new_feats = torch.stack(new_feats, dim=0)
        assert new_feats.shape == (num_scenes, num_target_views, H//2, W//2, samples_per_ray, self.mlp.output_dim)

        # volume render samples_per_ray dimension
        density = new_feats[..., 0]
        rgb = new_feats[..., 1:]
        deltas = times[..., 1:] - times[..., :-1]
        deltas = torch.cat((deltas, torch.tensor([1e10], device=deltas.device).expand(deltas[..., :1].shape)), dim=-1)
        alpha = 1.0 - torch.exp(-density * deltas)
        transmittance = torch.exp(-torch.cumsum(density * deltas, dim=-1))
        weights = alpha * transmittance
        renders = torch.sum(weights[..., None] * rgb, dim=-2)
        assert renders.shape == (num_scenes, num_target_views, H//2, W//2, self.mlp.output_dim - 1)
        
        # upsample from (H//2, W//2) to (H, W)
        renders = torch.permute(renders, (0, 1, 4, 2, 3))  # (num_scenes, num_target_views, num_channels, H//2, W//2)
        renders = renders.reshape(num_scenes * num_target_views, -1, H//2, W//2)
        renders = self.upsampler(renders)
        renders = renders.reshape(num_scenes, num_target_views, -1, H, W)

        return renders  # (num_scenes, num_target_views, num_channels, H, W)

    def encode_input_views(self, input_imgs, input_poses):
        """
        Get the encoded features of input views.

        Args: 
            input_imgs (tensor): .shape=(num_scenes, num_input_views, 3, H, W)
            input_poses (tensor): .shape=(num_scenes, num_input_views, 4, 4)
        
        Returns:
            inputs_encoded (tuple of 2 tensors):
                index 0: encoded imgs .shape=(num_scenes, num_input_views, feat_dim=16, num_depths=64, H, W)
                index 1: what you passed in as input_poses
        """
        # combine first and second dimension before passing in encoder
        num_scenes, num_input_views, num_channels, H, W = input_imgs.shape
        input_imgs = input_imgs.reshape(num_scenes * num_input_views, num_channels, H, W)

        # pass into GeNVS encoder
        encoded_imgs = self.encoder(input_imgs)

        # un-combine first and second dimension
        _, feat_dim, num_depths, H, W = encoded_imgs.shape
        encoded_imgs = encoded_imgs.reshape(num_scenes, num_input_views, feat_dim, num_depths, H, W)

        return (encoded_imgs, input_poses)



class MLP(nn.Module):
    """
    Our implementation of GeNVS's MLP.
    Written by: Daniel Brignac <dbrignac@arizona.edu>
    """

    def __init__(self, input_dim=16, hidden_dim=64, output_dim=17):
        """
        Constructor for MLP class.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ])
    
    def forward(self, x):
        """
        Forward pass of MLP.

        Args:
            x (tensor): tensor with shape (..., input_dim)
        
        Returns:
            y (tensor): tensor with shape (..., output_dim) after passing through MLP
        """
        orig_x = x
        # pass x through MLP
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                # last layer don't relu
                x = layer(x)
            else:
                x = F.relu(layer(x))

        density = F.relu(x[..., :1])  # density for volume rendering
        feats = x[..., 1:]

        # skip connection features
        feats = orig_x + feats

        y = torch.cat((density, feats), dim=-1)
        return y

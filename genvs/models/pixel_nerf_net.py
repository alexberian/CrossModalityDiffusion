# written by JhihYang Wu <jhihyangwu@arizona.edu> and colleagues
# model class for something like pixelNeRF's pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from genvs.models.genvs_encoder import GeNVSEncoder
from genvs.models.cross_domain_genvs_encoder import CrossDomainGeNVSEncoder
from genvs.utils import utils
from genvs.models.cam_weighting import create_cam_weighting_object
import numpy as np

class PixelNeRFNet(nn.Module):
    """
    Our model class for something like pixelNeRF pipeline.
    """

    def __init__(self, cam_weighting_method="baseline_mean", use_domain_embedding=False, domain_dim=64):
        """
        Constructor for PixelNeRFNet class.
        """
        super(PixelNeRFNet, self).__init__()

        # define model parameters (GeNVS encoder, radiance field MLP)
        if use_domain_embedding:
            self.domain_dim = domain_dim
            self.encoder = CrossDomainGeNVSEncoder(embedding_dim=domain_dim)
        else:
            self.encoder = GeNVSEncoder()
        self.mlp = MLP()

        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)

        self.cam_weighter = create_cam_weighting_object(cam_weighting_method)

    def forward(self, inputs_encoded, target_poses,
                focal, z_near, z_far, range_angle_feat_img, samples_per_ray=64):
        """
        Forward pass for pixelNeRF.

        Args: 
            inputs_encoded (tuple of 2 tensors): output from calling encode_input_views
            target_poses (tensor): novel poses to try to predict .shape=(num_scenes, num_target_views, 4, 4)
            focal (float): normalized focal length of camera
            z_near (float): z near to use when generating sample rays for vol rendering
            z_far (float): z far to use when generating sample rays for vol rendering
            range_angle_feat_img (bool): whether to render feat img as range angle plot
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

        # if range_angle_feat_img=True, don't use volume rendering
        # if multiple points have the same range and angle, their features get accumulated
        if range_angle_feat_img:
            renders = []
            for s_index in range(num_scenes):
                for t_index in range(num_target_views):
                    # 1. calculate range and angle of all sampling points
                    # get unit vectors for how the sensor is orientated
                    right_vec = target_poses[s_index, t_index, :3, 0]  # (3,), u vector of cam pose
                    upward_vec = target_poses[s_index, t_index, :3, 1]  # (3,), v vector of cam pose

                    # get displacement/direction vectors (final - initial) = (points_in_3D_space - sensor_center)
                    sensor_center = target_poses[s_index, t_index, :3, 3]  # (3,)
                    points_in_3D_space = pts[s_index, 0, t_index, :, :, :, :3, 0]  # (H//2, W//2, samples_per_ray, 3)
                    points_in_3D_space = points_in_3D_space.reshape(-1, 3)  # (num_sampled_points, 3)
                    disp_vec = points_in_3D_space - sensor_center.unsqueeze(-2)  # (num_sampled_points, 3)
                    num_sampled_points = disp_vec.shape[0]

                    # calculate range
                    range_ = torch.linalg.vector_norm(disp_vec, dim=-1)  # (num_sampled_points)

                    # project displacement vectors onto upward_vec and remove that component from disp_vec
                    # (projecting disp_vec onto plane made by forward_vec and right_vec)
                    tmp_upward_vec = upward_vec.unsqueeze(-2).unsqueeze(-1)
                    # disp_vec.unsqueeze(-2).shape = (num_sampled_points, 1, 3)
                    # tmp_upward_vec.shape = (1, 3, 1)
                    # (disp_vec.unsqueeze(-2) @ tmp_upward_vec).shape = (num_sampled_points, 1, 1)
                    upward_comp = (disp_vec.unsqueeze(-2) @ tmp_upward_vec) / (tmp_upward_vec.transpose(-1, -2) @ tmp_upward_vec)
                    upward_comp = upward_comp.squeeze(-1)  # (num_sampled_points, 1)
                    upward_comp = upward_comp * upward_vec.unsqueeze(-2)  # (num_sampled_points, 3)
                    assert upward_comp.shape == (num_sampled_points, 3)
                    disp_vec -= upward_comp  # disp_vec now not does not have upward component

                    # finally calculate angle
                    norm_disp_vec = disp_vec / torch.linalg.vector_norm(disp_vec, dim=-1).unsqueeze(-1)
                    cos_theta = norm_disp_vec.unsqueeze(-2) @ right_vec.unsqueeze(-2).unsqueeze(-1)  # (num_sampled_points, 1, 1)
                    cos_theta = cos_theta.squeeze(-1).squeeze(-1)  # (num_sampled_points)
                    theta = torch.arccos(cos_theta)  # in radians
                    theta = theta * 180 / np.pi  # in degrees
                    theta -= 90  # centered at 0 degrees
                    angle = theta

                    # 2. use range_, angle, and new_feats to create a feature image
                    image_tensor = torch.zeros(((H//2) * (W//2), self.mlp.output_dim - 1), dtype=torch.float32, requires_grad=True, device=new_feats.device)

                    # normalize angle and range_ to 0 and 1 for uv_coords
                    fov = 2 * torch.arctan(torch.tensor([1 / focal]))
                    fov = fov.item() * 180 / np.pi
                    angle_min = -fov/2
                    angle_max = fov/2
                    range_min = z_near * focal  # un-scale by focal, make it absolute range
                    range_max = z_far * focal  # un-scale by focal, make it absolute range
                    angle = (angle - angle_min) / (angle_max - angle_min)  # (num_sampled_points)
                    range_ = (range_ - range_min) / (range_max - range_min)  # (num_sampled_points)
                    # x-axis of image is angle +fov/2 to -fov/2 from left to right
                    # y-axis of image is range z_far to z_near from top to bottom
                    angle = 1 - angle
                    range_ = 1 - range_

                    # get a list of uv coordinates and colors used to fill the image
                    uv_coords = torch.stack((angle, range_), dim=-1)  # (num_sampled_points, 2)
                    density = new_feats[s_index, t_index, ..., 0]
                    colors = new_feats[s_index, t_index, ..., 1:] * density.unsqueeze(-1)
                    colors = colors.reshape(num_sampled_points, self.mlp.output_dim - 1)

                    # below is code from ChatGPT
                    # map UV coordinates to pixel indices
                    u = uv_coords[:, 0] * (W//2 - 1)
                    v = uv_coords[:, 1] * (H//2 - 1)

                    # integer and fractional parts
                    u_int = u.floor().long()
                    v_int = v.floor().long()
                    u_frac = u - u_int.float()
                    v_frac = v - v_int.float()

                    # get neighboring pixel indices
                    u0, u1 = u_int, u_int + 1
                    v0, v1 = v_int, v_int + 1

                    # clamp to image boundaries
                    u0 = u0.clamp(0, W//2 - 1)
                    u1 = u1.clamp(0, W//2 - 1)
                    v0 = v0.clamp(0, H//2 - 1)
                    v1 = v1.clamp(0, H//2 - 1)

                    # bilinear interpolation weights
                    w00 = (1 - u_frac) * (1 - v_frac)
                    w01 = (1 - u_frac) * v_frac
                    w10 = u_frac * (1 - v_frac)
                    w11 = u_frac * v_frac

                    # scatter colors into the image tensor
                    image_tensor = image_tensor.scatter_add(0, (v0 * W//2 + u0).unsqueeze(-1).expand_as(colors), (w00.unsqueeze(-1) * colors))
                    image_tensor = image_tensor.scatter_add(0, (v0 * W//2 + u1).unsqueeze(-1).expand_as(colors), (w10.unsqueeze(-1) * colors))
                    image_tensor = image_tensor.scatter_add(0, (v1 * W//2 + u0).unsqueeze(-1).expand_as(colors), (w01.unsqueeze(-1) * colors))
                    image_tensor = image_tensor.scatter_add(0, (v1 * W//2 + u1).unsqueeze(-1).expand_as(colors), (w11.unsqueeze(-1) * colors))

                    # save image_tensor
                    renders.append(image_tensor)
            renders = torch.stack(renders)
            renders = renders.reshape(num_scenes, num_target_views, H//2, W//2, self.mlp.output_dim - 1)

        # volume render samples_per_ray dimension
        else:
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

    def encode_input_views(self, input_imgs, input_poses, domain_embeddings=None):
        """
        Get the encoded features of input views.

        Args: 
            input_imgs (tensor): .shape=(num_scenes, num_input_views, 3, H, W)
            input_poses (tensor): .shape=(num_scenes, num_input_views, 4, 4)
            domain_embeddings (tensor): .shape=(num_scenes, num_input_views, domain_embedding_dim)
        
        Returns:
            inputs_encoded (tuple of 2 tensors):
                index 0: encoded imgs .shape=(num_scenes, num_input_views, feat_dim=16, num_depths=64, H, W)
                index 1: what you passed in as input_poses
        """
        # combine first and second dimension before passing in encoder
        num_scenes, num_input_views, num_channels, H, W = input_imgs.shape
        input_imgs = input_imgs.reshape(num_scenes * num_input_views, num_channels, H, W)
        domain_embeddings = domain_embeddings.reshape(num_scenes * num_input_views, self.domain_dim) if domain_embeddings is not None else None

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

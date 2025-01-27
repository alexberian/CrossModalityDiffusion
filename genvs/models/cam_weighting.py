# file containing the code for all the camera weighting algorithms
# Written by JhihYang Wu and Alex Berian 2024

import torch
from torch import nn
import numpy as np
import torch.autograd.profiler as profiler
import math
import torch.nn.functional as F


def cam_weighting(method_name, input_poses, target_poses):
    """
    Dispatch method for all the deterministic camera weighting algorithms.

    Args:
        method_name (str): which camera weighting algorithm to use
        input_poses (tensor): .shape=(num_scenes, num_input_views, 4, 4)
        target_poses (tensor): .shape=(num_scenes, num_target_views, 4, 4)

    Returns:
        weights (tensor): used to weight features from different cameras .shape=(num_scenes, num_input_views, num_target_views)
    """
    num_scenes = input_poses.shape[0]
    all_weights = []
    for i in range(num_scenes):  # one scene at a time
        weights = []
        for j in range(target_poses.shape[1]):  # one target view at a time
            if method_name == "baseline_mean" or input_poses.shape[1] < 2:
                weights.append(baseline_mean(input_poses[i], target_poses[i, j]))
            elif method_name == "distance":
                weights.append(distance(input_poses[i], target_poses[i, j]))
            elif method_name.startswith("error_weighing_alpha="):
                alpha = float(method_name.split("=")[-1])
                weights.append(error_weighting(input_poses[i], target_poses[i, j], alpha))
            elif method_name.startswith("distance_gaussian_b="):
                b = float(method_name.split("=")[-1])
                weights.append(distance_gaussian(input_poses[i], target_poses[i, j], b))
            elif method_name == "l1_weighing":
                weights.append(l1_weighting(input_poses[i], target_poses[i, j]))
            elif method_name == "f_norm_weighting":
                weights.append(f_norm_weighting(input_poses[i], target_poses[i, j]))
            elif method_name == "rel_cam_poses_l2":
                weights.append(rel_cam_poses_l2(input_poses[i], target_poses[i, j]))
            elif method_name == "rel_cam_poses_f_norm":
                weights.append(rel_cam_poses_f_norm(input_poses[i], target_poses[i, j]))
            elif method_name == "distance_squared":
                weights.append(distance_squared(input_poses[i], target_poses[i, j]))
            elif method_name == "distance_cubed":
                weights.append(distance_cubed(input_poses[i], target_poses[i, j]))
            else:
                assert False, f"{method_name} cam weighting algo not implemented"
        weights = torch.squeeze(torch.cat(weights, dim=2), dim=-1)  # (1, num_input_views, num_target_views)
        all_weights.append(weights)
    return torch.cat(all_weights, dim=0)  # (num_scenes, num_input_views, num_target_views)

def baseline_mean(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    return torch.tensor([1 / num_input_views], device=input_poses.device).expand(1, num_input_views, 1, 1)

def distance(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    input_locs = input_poses[:, :3, 3]  # (num_input_views, 3)
    target_loc = target_pose[:3, 3]  # (3)
    distances = torch.sqrt(((target_loc[None, :] - input_locs) ** 2).sum(dim=-1))  # (num_input_views)
    weights = 1 / (distances + 1e-6)
    # normalize
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()

def error_weighting(input_poses, target_pose, alpha):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    R_target = target_pose[:3, :3].unsqueeze(0).repeat(num_input_views, 1, 1)  # (num_input_views, 3, 3)
    R_input = input_poses[:, :3, :3]  # (num_input_views, 3, 3)
    R_ti = torch.matmul(R_target.transpose(-1, -2), R_input)  # (num_input_views, 3, 3)
    trace_R_ti = R_ti[:, 0, 0] + R_ti[:, 1, 1] + R_ti[:, 2, 2]  # (num_input_views)
    theta = torch.acos((trace_R_ti - 1) / 2)
    rot_err = theta / 3.1415926535  # (num_input_views) with values from 0.0 to 1.0
    # calculate translation error
    target_loc = target_pose[:3, 3]  # (3)
    input_locs = input_poses[:, :3, 3]  # (num_input_views, 3)
    distances = torch.sqrt(((target_loc[None, :] - input_locs) ** 2).sum(dim=-1))  # (num_input_views)
    trans_err = distances / torch.max(distances)  # (num_input_views) with values from 0.0 to 1.0
    # combine errors
    total_err = alpha * rot_err + (1 - alpha) * trans_err  # (num_input_views)
    # calculate weights
    weights = 1 / (total_err + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)  # larger err should have less weight
    weights = torch.nan_to_num(weights)
    return weights / weights.sum()

def distance_gaussian(input_poses, target_pose, b):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    input_locs = input_poses[:, :3, 3]  # (num_input_views, 3)
    target_loc = target_pose[:3, 3]  # (3)
    distances = torch.sqrt(((target_loc[None, :] - input_locs) ** 2).sum(dim=-1))  # (num_input_views)
    weights = torch.exp(-b * (distances ** 2))
    # normalize
    norm_fac = weights.sum(dim=-1)  # (1)
    weights = weights / norm_fac
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights

def l1_weighting(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    diffs = torch.abs(target_pose.unsqueeze(0) - input_poses).sum(-1).sum(-1)  # (num_input_views)
    # normalize
    weights = 1 / (diffs + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()

def f_norm_weighting(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    diffs = torch.sqrt(((target_pose.unsqueeze(0) - input_poses) ** 2).sum(-1).sum(-1))  # (num_input_views)
    # normalize
    weights = 1 / (diffs + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()

def rel_cam_poses_l2(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    target_pose = target_pose[None, :, :].repeat(num_input_views, 1, 1)  # expand target_pose shape to match input_poses
    # calculate relative camera poses from target to each input
    rel_cam_poses = torch.zeros(num_input_views, 4, 4, device=input_poses.device)
    rel_cam_poses[:, :3, :3] = torch.matmul(input_poses[:, :3, :3], target_pose[:, :3, :3].transpose(-1, -2))
    rel_cam_poses[:, :3, 3:] = -torch.matmul(rel_cam_poses[:, :3, :3], target_pose[:, :3, 3:]) + input_poses[:, :3, 3:]
    rel_cam_poses[:, 3, 3] = 1
    # calculate 2-norm of rel_cam_poses
    weights = torch.norm(rel_cam_poses, dim=(1, 2), p=2)  # (num_input_views)
    # normalize
    weights = 1 / (weights + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()

def rel_cam_poses_f_norm(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    target_pose = target_pose[None, :, :].repeat(num_input_views, 1, 1)  # expand target_pose shape to match input_poses
    # calculate relative camera poses from target to each input
    rel_cam_poses = torch.zeros(num_input_views, 4, 4, device=input_poses.device)
    rel_cam_poses[:, :3, :3] = torch.matmul(input_poses[:, :3, :3], target_pose[:, :3, :3].transpose(-1, -2))
    rel_cam_poses[:, :3, 3:] = -torch.matmul(rel_cam_poses[:, :3, :3], target_pose[:, :3, 3:]) + input_poses[:, :3, 3:]
    rel_cam_poses[:, 3, 3] = 1
    # calculate f-norm of rel_cam_poses
    weights = torch.sqrt((rel_cam_poses ** 2).sum(dim=-1).sum(dim=-1))  # (num_input_views)
    # normalize
    weights = 1 / (weights + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()

def distance_squared(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    input_locs = input_poses[:, :3, 3]  # (num_input_views, 3)
    target_loc = target_pose[:3, 3]  # (3)
    distances = torch.sqrt(((target_loc[None, :] - input_locs) ** 2).sum(dim=-1))  # (num_input_views)
    weights = distances ** 2
    # normalize
    weights = 1 / (weights + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()

def distance_cubed(input_poses, target_pose):
    num_input_views, _, _ = input_poses.shape
    # given input_poses of shape (num_input_views, 4, 4)
    # given target_pose of shape (4, 4)
    #
    # return a tensor of shape (1, num_input_views, 1, 1) to weight mlp_input
    # tensor should sum to 1
    input_locs = input_poses[:, :3, 3]  # (num_input_views, 3)
    target_loc = target_pose[:3, 3]  # (3)
    distances = torch.sqrt(((target_loc[None, :] - input_locs) ** 2).sum(dim=-1))  # (num_input_views)
    weights = distances ** 3
    # normalize
    weights = 1 / (weights + 1e-6)
    weights = weights.reshape(1, num_input_views, 1, 1)
    return weights / weights.sum()


###############################################
# Attention based camera weighting algorithms #
###############################################
#         Written by Alex Berian 2024         #
###############################################
# Attention based camera weighting algorithms #
###############################################
#         Written by Alex Berian 2024         #
###############################################
# Attention based camera weighting algorithms #
###############################################
#         Written by Alex Berian 2024         #
###############################################
# Attention based camera weighting algorithms #
###############################################

def create_cam_weighting_object(method_name):
    """
    returns the camera weighting object based on the method name
    """

    if "attention" in method_name:
        if method_name == "cross_attention":
            return CrossAttentionCamWeighter(learned_attention = False)

        elif method_name == "learned_cross_attention":
            return CrossAttentionCamWeighter(learned_attention = True)

        elif method_name == "relative_pose_self_attention":
            return RelativePoseSelfAttentionCamWeighter()

        else:
            raise ValueError(f"attention based camera weighting algorithm {method_name} not implemented")
    else:
        return DeterministicCamWeighter(method_name)

    

class DeterministicCamWeighter(nn.Module):
    """
    wrapper class for deterministic camera weighting algorithms
    """
    def __init__(self, method_name):
        super().__init__()
        self.method_name = method_name
    
    def forward(self, input_poses, target_poses):
        with profiler.record_function("deterministic_cam_weighting"):
            return cam_weighting(self.method_name, input_poses, target_poses)



class PositionalEncoding(torch.nn.Module):
    """
    Copied from Alex Yu's PixelNeRF repository on GitHub
    Modified by Alex Berian 2024

    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )



class PrincipalRayCameraEmbedder(nn.Module):
    """
    Uses the pincipal axis direction and camera center concatenated to create 
    the embedding of a camera pose matrix. Then a single dense layer is used to
    create the final embedding.
    """
    def __init__(
            self,
            num_freqs = 6,
            freq_factor = 1.5,
            num_linear_layers = 0,
            embed_dim = 128,
            activation = nn.ReLU(),
            **kwargs,
        ):
        super().__init__(**kwargs)

        # define stuff
        self.positional_encoder = PositionalEncoding(num_freqs=num_freqs, d_in=3, freq_factor=freq_factor, include_input=True)
        self.activation = activation

        # create the linear layers
        first_layer_input_dim = self.positional_encoder.d_out + 3
        self.linear_layers = nn.ModuleList()
        if num_linear_layers > 0:
            for i in range(num_linear_layers):
                in_dim = embed_dim
                if i == 0:
                    in_dim = first_layer_input_dim
                new_layer = nn.Linear(in_dim, embed_dim)
                self.linear_layers.append(new_layer)
            
        # set define embed dim of this embedder
            self.embed_dim = embed_dim
        else:
            self.embed_dim = first_layer_input_dim


    def forward(self, poses):
        """
        :param poses (..., 4, 4) poses
        :return embeddings (..., embed_dim) embeddings of the poses
        """
        with profiler.record_function("principal_axis_camra_embedder"):

            assert(poses.shape[-2:] == (4,4)), "poses must have shape (..., 4, 4)"

            # get camera centers and directions of view
            shape_prefix = poses.shape[:-2] # the ... part of the shape
            poses = poses.reshape(-1, 4, 4) # (B, 4, 4)
            direction  = -poses[:, :3, 2] # (B, 3)
            cam_center =  poses[:, :3, 3] # (B, 3)

            # calculate the embeddings
            encoded_cam_center = self.positional_encoder(cam_center) # (B, d_out)
            x = torch.cat([encoded_cam_center, direction], dim=-1) # (B, d_out + 3)
            for layer in self.linear_layers:
                x = layer(x)
                x = self.activation(x)

            return x.reshape(*shape_prefix, self.embed_dim)




class CrossAttentionCamWeighter(nn.Module):
    """
    TODO: use embedder object instead of doing the embedding in this object

    Uses cross attention between target pose and source poses to calculate weights for combining source views.
    Has the option of using or not using learned weights.

    Query: encoded TARGET camera center concatenated with unencoded direction of view
    Key: encoded SOURCE camera center concatenated with unencoded direction of view
    Value: hidden state of the MLP for source views
    """
    def __init__(self, learned_attention = True, **kwargs):
        super().__init__(**kwargs)

        self.positional_encoder = PositionalEncoding(num_freqs=6, d_in=3, freq_factor=1.5, include_input=True)
        self.learned_attention = learned_attention

        # initialize learned attention layers
        if self.learned_attention:
            self.attention_dim = self.positional_encoder.d_out + 3
            self.hidden_dim = 64
            self.head_dim = 128
            self.query_layer = nn.Sequential(
                nn.Linear(self.attention_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.head_dim)
            )
            self.key_layer = self.query_layer

    def forward(self, input_poses=None, target_poses=None):
        """
        SB = number of scenes
        NS = number of source views per scene
        B' = number of target views per scene
        
        :param input_poses (SB, NS, 4, 4) source poses
        :param target_poses (SB, B', 4, 4) target poses
        :return weights (SB, NS, B') weights for combining source views for each target view
        """
        with profiler.record_function("cross_attention_cam_weighting"):
            # input_poses.shape = (num_scenes, num_input_views, 4, 4)
            # target_poses.shape = (num_scenes, num_target_views, 4, 4)
            num_scenes = input_poses.shape[0]
            all_weights = []
            for i in range(num_scenes):  # one scene at a time
                weights = []
                for j in range(target_poses.shape[1]):  # one target view at a time
                    if input_poses.shape[1] < 2:
                        weights.append(baseline_mean(input_poses[i], target_poses[i, j]))
                    else:
                        # === learned weighting ===
                        i_poses = input_poses[i]  # (num_input_views, 4, 4)
                        t_pose = target_poses[i, j]  # (4, 4)
                        num_input_views = i_poses.shape[0]
                        # append to weights a (1, num_input_views, 1, 1) tensor
                        input_dirs = i_poses[:, :3, 2]  # (num_input_views, 3)
                        target_dir = t_pose[:3, 2]  # (3)
                        input_locs = i_poses[:, :3, 3]  # (num_input_views, 3)
                        target_loc = t_pose[:3, 3]  # (3)

                        # encode locs and concat with dirs
                        encoded_input_locs = self.positional_encoder(input_locs)  # (num_input_views, d_out)
                        encoded_target_loc = self.positional_encoder(target_loc.reshape(1, 3))  # (1, d_out)
                        q = torch.cat((encoded_target_loc, target_dir[None]), dim=1)  # (1, d_out + 3)
                        k = torch.cat((encoded_input_locs, input_dirs), dim=-1)  # (num_input_views, d_out + 3)

                        xq = self.query_layer(q)  # (1, head_dim)
                        xk = self.key_layer(k)  # (num_input_views, head_dim)
                        head_dim = xq.shape[-1]

                        scores = torch.matmul(xq, xk.transpose(-1, -2)) / math.sqrt(head_dim)  # (1, num_input_views)
                        assert scores.shape[0] == 1 and scores.shape[1] == num_input_views and len(scores.shape) == 2

                        scores = F.softmax(scores, dim=-1)
                        scores = scores.reshape(1, num_input_views, 1, 1)
                        weights.append(scores)
                        #if i == 0 and j == 0:
                        #    print(weights[-1])
                        # =========================
                weights = torch.squeeze(torch.cat(weights, dim=2), dim=-1)  # (1, num_input_views, num_target_views)
                all_weights.append(weights)
            return torch.cat(all_weights, dim=0)  # (num_scenes, num_input_views, num_target_views)



class RelativePoseSelfAttentionCamWeighter(nn.Module):
    """
    Uses self attention on the relative poses using pytorch's multihead attention.
    """
    def __init__(self, num_heads = 4, embedder=PrincipalRayCameraEmbedder(num_linear_layers=2), **kwargs):
        super().__init__(**kwargs)

        self.embedder = embedder
        self.attention_dim = self.embedder.embed_dim
        
        # initialize learned attention layers
        self.multihead_attention = nn.MultiheadAttention(self.attention_dim, num_heads, batch_first=True)
        
        # initialize the weight calculation layers
        self.weight_calculation_subnet = nn.Sequential(
            nn.Linear(self.attention_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        

    def forward(self, src_poses=None, target_poses=None, **kwargs):
        """
        SB = number of scenes
        NS = number of source views per scene
        B' = number of target views per scene
        
        :param src_poses (SB, NS, 4, 4) source poses
        :param target_poses (SB, B', 4, 4) target poses
        :return weights (SB, NS, B') weights for combining source views for each target view
        """

        with profiler.record_function("relative_pose_self_attention_cam_weighting"):

            # get shape information
            SB, NS, _, _ = src_poses.shape
            _, Bp, _, _ = target_poses.shape
            if NS == 1: # if only one source view, don't need to combine
                weights = torch.ones(SB, NS, Bp, device=src_poses.device)
                return weights

            # calculate relative poses
            relative_source_poses = torch.linalg.inv(src_poses).reshape(SB,NS,1,4,4) \
                                    @ target_poses.reshape(SB,1,Bp,4,4)
            # (SB, NS, B', 4, 4)
        
            # create the vectors for attention
            vectors = self.embedder(relative_source_poses) # (SB, NS, B', A)

            # apply attention
            vectors = vectors.permute(0, 2, 1, 3) # (SB, B', NS, A)
            vectors = vectors.reshape(-1, NS, self.attention_dim) # (SB*B', NS, A)
            vectors_with_attention = self.multihead_attention(vectors, vectors, vectors, need_weights = False)[0] # (SB*B', NS, A)

            # calculate weights
            weight  = self.weight_calculation_subnet(vectors_with_attention) # (SB*B', NS, 1)
            weight = torch.softmax(weight, dim=1) # (SB*B', NS, 1)

            # reshape and return weights
            weights = weight.reshape(SB, Bp, NS)
            weights = weights.permute(0, 2, 1) # (SB, NS, B')
            return weights

def create_cam_weighting_object(method_name):
    """
    returns the camera weighting object based on the method name
    """

    if "attention" in method_name:
        if method_name == "cross_attention":
            return CrossAttentionCamWeighter(learned_attention = False)

        elif method_name == "learned_cross_attention":
            return CrossAttentionCamWeighter(learned_attention = True)

        elif method_name == "relative_pose_self_attention":
            return RelativePoseSelfAttentionCamWeighter( num_heads = 3)

        else:
            raise ValueError(f"attention based camera weighting algorithm {method_name} not implemented")
    else:
        return DeterministicCamWeighter(method_name)

# file containing the code for all the camera weighting algorithms
# Written by JhihYang Wu and Alex Berian 2024

import torch
from torch import nn
import numpy as np
import torch.autograd.profiler as profiler

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


class CrossAttentionCamWeighter(nn.Module):
    """
    Written by Alex Berian 2024

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
            self.query_layer = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
            self.key_layer = nn.Linear(self.attention_dim, self.attention_dim, bias=False)
        

    def forward(self, src_poses=None, target_poses=None, **kwargs):
        """
        SB = number of scenes
        NS = number of source views per scene
        B' = number of target views per scene
        
        :param src_poses (SB, NS, 4, 4) source poses
        :param target_poses (SB, B', 4, 4) target poses
        :return weights (SB, NS, B') weights for combining source views for each target view
        """
        with profiler.record_function("cross_attention_cam_weighting"):

            # get shape information
            SB, NS, _, _ = src_poses.shape
            _, Bp, _, _ = target_poses.shape
            if NS == 1: # if only one source view, don't need to combine
                weights = torch.ones(SB, NS, Bp, device=src_poses.device)
                return weights

            # get camera centers and directions of view
            d_s = src_poses[..., :3, 2]     # (SB, NS, 3)
            d_t = target_poses[..., :3, 2]  # (SB, B', 3)
            c_s = src_poses[..., :3, 3]     # (SB, NS, 3)
            c_t = target_poses[..., :3, 3]  # (SB, B', 3)

            # calculate queries and keys
            coded_c_t = self.positional_encoder(c_t.reshape(-1,3)) # (SB*B', A-3) where A := self.attention_dim
            coded_c_s = self.positional_encoder(c_s.reshape(-1,3)) # (SB*NS, A-3)
            q = torch.cat([coded_c_t, d_t.reshape(-1,3)], dim=-1).reshape(SB,Bp,-1) # (SB, B', A)
            k = torch.cat([coded_c_s, d_s.reshape(-1,3)], dim=-1).reshape(SB,NS,-1) # (SB, NS, A)

            # use learned attention weights if needed
            if self.learned_attention:
                q = self.query_layer(q)
                k = self.key_layer(k)

            # calculate attention weights
            k_T = k.permute(0, 2, 1)      # (SB, A , NS)
            Wp = q @ k_T                  # (SB, B', NS)
            W = torch.softmax(Wp, dim=-1) # (SB, B', NS)
            weights = W.permute(0, 2, 1)        # (SB, NS, B')

            return weights
        



class RelativePoseSelfAttentionCamWeighter(nn.Module):
    """
    Written by Alex Berian 2024

    Uses self attention on the relative poses using pytorch's multihead attention.
    """
    def __init__(self, num_heads = 3, **kwargs):
        super().__init__(**kwargs)

        self.positional_encoder = PositionalEncoding(num_freqs=6, d_in=3, freq_factor=1.5, include_input=True)
        self.attention_dim = self.positional_encoder.d_out + 3
        
        # initialize learned attention layers
        hidden_attention_dim = int( num_heads * np.ceil(self.attention_dim / num_heads) )
        self.multihead_attention = nn.MultiheadAttention(hidden_attention_dim, num_heads, batch_first=True)
        self.weight_calculation_layer = nn.Linear(hidden_attention_dim, 1)
        

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
        

            # get camera centers and directions of view
            c_r = relative_source_poses[..., :3, 3] # (SB, NS, B', 3)
            d_r = relative_source_poses[..., :3, 2] # (SB, NS, B', 3)

            # create the vectors for attention
            coded_c_r = self.positional_encoder(c_r.reshape(-1,3)) # (SB*NS*B', A-3) where A := self.attention_dim
            vectors = torch.cat([coded_c_r, d_r.reshape(-1,3)], dim=-1).reshape(SB,NS,Bp,-1) # (SB, NS, B', A)

            # apply attention
            vectors = vectors.permute(0, 2, 1, 3) # (SB, B', NS, A)
            vectors = vectors.reshape(-1, NS, self.attention_dim) # (SB*B', NS, A)
            vectors_with_attention = self.multihead_attention(vectors, vectors, vectors, need_weights = False)[0] # (SB*B', NS, A)
            weight  = self.weight_calculation_layer(vectors_with_attention) # (SB*B', NS, 1)
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

# written by JhihYang Wu <jhihyangwu@arizona.edu> and colleagues
# useful utilities

import torch
import numpy as np

def gen_rays(poses, height, width, focal, device="cpu"):
    """
    Generates rays (origin, direction) from camera poses.

    Args:
        poses (tensor): .shape=(N, 4, 4)
        height (int): number of rows of rays to generate
        width (int): number of cols of rays to generate
        focal (float): normalized focal length of camera
    
    Returns:
        rays (tensor): R^3 origin and R^3 direction .shape=(N, H, W, 6)
    """
    aspect_ratio = width / height
    # generate points on canonical -1 +1 image plane
    x = torch.linspace(-1 * aspect_ratio, 1 * aspect_ratio, steps=width, device=device)
    y = torch.linspace(1, -1, steps=height, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    # place these points -focal away (negative so it is in front of camera)
    grid_z = torch.full_like(grid_x, -focal, device=device)

    # combine to create un-rotated directions tensor
    directions = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(1, height, width, 3, 1)  # (1, H, W, 3, 1)
    # rotate
    R = poses[:, :3, :3].unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 3, 3)
    directions = R @ directions  # (N, H, W, 3, 1)
    directions = torch.squeeze(directions, -1)  # (N, H, W, 3)

    # extract origins
    origins = poses[:, :3, 3].unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 3)
    origins = origins.repeat(1, height, width, 1)  # (N, H, W, 3)

    # concatenate
    rays = torch.cat((origins, directions), dim=-1)  # (N, H, W, 6)
    return rays

def gen_pts_from_rays(rays, z_near, z_far, samples_per_ray):
    """
    Generates random points in 3D space by sampling distances along rays from z_near to z_far.

    Args:
        rays (tensor): return value of gen_ray() .shape=(N, H, W, 6)
        z_near (float): z near value to sample from
        z_far (float): z far value to sample to
        samples_per_ray (int): number of samples to randomly sample along each ray

    Returns:
        pts (tensor): .shape=(N, H, W, samples_per_ray, 3)
        times (tensor): times along the ray sampled .shape=(N, H, W, sample_per_ray, 1)
    """
    origins = rays[..., :3]  # (N, H, W, 3)
    N, H, W, _ = origins.shape
    directions = rays[..., 3:]  # (N, H, W, 3)
    times = torch.linspace(z_near, z_far, steps=samples_per_ray, device=origins.device)
    times = times.reshape(1, 1, 1, samples_per_ray, 1).repeat(N, H, W, 1, 1)  # (N, H, W, sample_per_ray, 1)

    # add some randomness to the times to sample randomly along the ray
    # don't touch the z_near and z_far times
    dist_between_samples = (z_far - z_near) / (samples_per_ray - 1)  # (x - 1) deltas for a list of x numbers
    randomness = (torch.rand(N, H, W, samples_per_ray - 2, 1, device=origins.device) - 0.5) * dist_between_samples  # (N, H, W, sample_per_ray - 2, 1)
    times[..., 1:-1, :] += randomness

    # pts = o + t * d
    pts = origins.unsqueeze(3) + times * directions.unsqueeze(3)

    return pts, times

def invert_pose_mats(poses):
    """
    Invert pose matrices to get extrinsic matrices.
    Faster than torch.linalg.inv(poses) but only works for 4x4 pose matrices.

    Args:
        poses (tensor): .shape=(..., 4, 4)
    
    Returns:
        inv_poses (tensor): linear algebra inverse of poses .shape=(..., 4, 4)
    """
    inv_poses = torch.zeros_like(poses)

    R = poses[..., :3, :3]
    t = poses[..., :3, 3:]
    R_transposed = torch.transpose(R, -1, -2)

    inv_poses[..., :3, :3] = R_transposed  # new rotation
    inv_poses[..., :3, 3:] = -R_transposed @ t # new translation
    inv_poses[..., 3, 3] = 1  # 1 at bottom right

    return inv_poses

def pick_random_views(num_views, images, poses):
    """
    Slice out a few views and poses given all images and poses of a scene.

    Args:
        num_views (int): number of views to slice out
        images (tensor): .shape=(N, views_per_scene, C, H, W)
        poses (tensor): .shape=(N, views_per_scene, 4, 4)
    
    Returns:
        images_ (tensor): .shape=(N, num_views, C, H, W)
        poses_ (tensor): .shape=(N, num_views, 4, 4)
    """
    n, views_per_scene, _, _, _ = images.shape
    rand_indices = torch.randint(0, views_per_scene, (n, num_views))
    tmp = torch.arange(n).unsqueeze(1).expand(-1, num_views)
    images_ = images[tmp, rand_indices]
    poses_ = poses[tmp, rand_indices]
    return images_, poses_

def img_tensor_to_npy(img_tensor):
    """
    Converts any range float image tensor to uint8 numpy array.

    Args:
        img_tensor (tensor): float tensor any range .shape=(C, H, W)
    
    Returns:
        img_npy (numpy arr): uint8 0 255 numpy array .shape=(H, W, C)
    """
    img_tensor = img_tensor - img_tensor.min()
    img_tensor = img_tensor / img_tensor.max()
    img_npy = img_tensor * 255
    img_npy = torch.permute(img_npy, (1, 2, 0))
    img_npy = img_npy.cpu().detach().numpy().astype(np.uint8)
    return img_npy

def feat_img_processor(image, chan_process="max"):
    """
    Reduces the channel dimension of feature images to 3.

    Args:
        image (tensor): .shape=(batch_size, C, H, W)
    
    Returns:
        image (tensor): float -1 +1 tensor .shape=(batch_size, 3, H, W)

    Written by: Alex Berian <berian@arizona.edu>
    """
    assert len(image.shape) == 4
    if chan_process == "first3":
        image = image[:, :3]
    else:
        # break up the channels into 3 sections
        n_chan = image.shape[1]
        chan_size = n_chan // 3
        chan_size = np.repeat(chan_size, 3)
        n_current_chan = chan_size.sum()
        n_missing_chan = n_chan - n_current_chan
        chan_size[:n_missing_chan] += 1
        assert(chan_size.sum() == n_chan), "chan_size.sum() != n_chan\n%d != %d" % (chan_size.sum(), n_chan)
        image = torch.split(image, chan_size.tolist(), dim=1)  # [(C1,H,W),(C2,H,W),(C3,H,W)]

        # take max or mean of each section over the channel dimension
        if chan_process == "max":
            image = [chan.max(dim=1, keepdim=True)[0] for chan in image]  # [(1,H,W),(1,H,W),(1,H,W)]
        elif chan_process == "mean":
            image = [chan.mean(dim=1, keepdim=True) for chan in image]  # [(1,H,W),(1,H,W),(1,H,W)]
        else:
            raise ValueError("chan_process = %s is not supported" % chan_process)
        image = torch.cat(image,dim=1)  # (B,3,H,W)

    # get it in -1 +1 range
    if torch.any(image < 0):
        image -= image.min()
    image /= (image.max() + 1e-8)
    image = 2 * image - 1
    return image

class AddGaussianNoise():
    """
    Class for adding gaussian noise to tensors.
    """
    
    def __init__(self, mean=0.0):
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * torch.rand(1, device=tensor.device).item() * 0.5 + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0})'.format(self.mean)

# copied from https://github.com/NVlabs/edm/blob/main/generate.py
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

# copied from https://github.com/NVlabs/edm/blob/main/generate.py
def edm_sampler(
    net, latents, feat_imgs, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(torch.cat((x_hat, feat_imgs), dim=1), t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(torch.cat((x_next, feat_imgs), dim=1), t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def one_step_edm_sampler(
    net, latents, feat_imgs, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    sigma = torch.tensor([sigma_max], device=feat_imgs.device)
    denoised = net(torch.cat((latents * sigma, feat_imgs), dim=1), sigma, class_labels)
    return denoised

def archimedean_spiral(radius=1, num_points=1000, num_turns=10):
    """
    Generates poses along a 3D Archimedean spiral.
    +Z is up direction.

    Args:
        radius (float): radius of the sphere to revolve around
        num_points (int): number of poses to generate
        num_turns (int): number of turns in the spiral

    Returns:
        poses (tensor): .shape=(num_points, 4, 4)
    """
    # 1. generate points on the spiral
    theta = np.linspace(0, 2 * np.pi * num_turns, num_points)
    z = np.linspace(radius, -radius, num_points)  # z-coordinates from radius to -radius
    r = np.sqrt(radius**2 - z**2)  # 2d radius from z axis
    x = r * np.cos(theta)  # convert spherical coordinates to cartesian coordinates
    y = r * np.sin(theta)

    # 2. get the rotations of the poses
    up = np.array([0, 0, 1]).reshape(1, 3)
    pts = np.stack((x, y, z), axis=1)  # (num_points, 3)
    w = pts - 0  # (num_points, 3)
    w = w / (np.linalg.norm(w, axis=-1).reshape(-1, 1) + 1e-6)
    u = np.cross(up, w)  # (num_points, 3)
    u = u / (np.linalg.norm(u, axis=-1).reshape(-1, 1) + 1e-6)
    v = np.cross(w, u)  # (num_points, 3)
    v = v / (np.linalg.norm(v, axis=-1).reshape(-1, 1) + 1e-6)

    # 3. format into poses matrices
    poses = np.zeros((num_points, 4, 4))
    poses[:, 3, 3] = 1
    poses[:, :3, 0] = u
    poses[:, :3, 1] = v
    poses[:, :3, 2] = w
    poses[:, :3, 3] = pts

    return poses


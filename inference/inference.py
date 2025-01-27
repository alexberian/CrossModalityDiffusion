# written by JhihYang Wu <jhihyangwu@arizona.edu>
# scripts for generating videos or images from trained GeNVS model

import click
import os
import sys
import pickle
import torch
import imageio
from tqdm import tqdm
import numpy as np
import glob
from torchvision import transforms
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from genvs.utils.utils import StackedRandomGenerator, edm_sampler, one_step_edm_sampler, feat_img_processor, archimedean_spiral
from genvs.models.genvs import GeNVS

# dataset specific parameters
VIDEO_RADIUS = 1.3  # radius of archimedean spiral when generating video
Z_NEAR = 0.8
Z_FAR = 1.8
COORD_TRANSFORM = torch.diag(
    torch.tensor([1, -1, -1, 1], dtype=torch.float32)
)  # for preprocessing poses loaded from txt
IMG_TO_TENSOR_BALANCED = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])  # for preprocessing images loaded from png

@click.command()
@click.option("--in_dir", help="What to input into GeNVS", required=True)
@click.option("--out_dir", help="Where to save output of GeNVS", required=True)
@click.option("--denoising_steps", help="How many denoising steps to use for diffusion model", required=True, type=int)
@click.option("--cam_weighting", help="Which camera weighting algorithm to use", required=True)
@click.option("--ckpt_path", help="Path to trained model", required=True)
@click.option("--gpu_id", help="Which GPU to use", required=True)
@click.option("--batch_size", help="How many target views to generate at once", required=True, type=int)
@click.option('--cond_method', help="Matters for video generation only: whether to condition on previous frames to generate next frame", type=click.Choice(["autoreg", "normal"]), required=True)
@click.option('--video', help="Whether to generate archimedean spiral video or just images of target_pose", is_flag=True)

def main(in_dir, out_dir, denoising_steps, cam_weighting, ckpt_path, gpu_id, batch_size, cond_method, video):
    # make out directory
    os.makedirs(out_dir, exist_ok=True)

    # setup CUDA device
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using device: {device}")

    # load test scene
    scene_data = load_input(in_dir)
    images = scene_data["images"].to(device)
    poses = scene_data["poses"].to(device)
    target_poses = scene_data["target_poses"].to(device)
    focal = scene_data["focal"].item()
    z_near = scene_data["z_near"]
    z_far = scene_data["z_far"]
    target_pose_paths = scene_data["target_pose_paths"]
    _, _, H, W = images.shape

    # if generating video, override poses with archimedean spiral
    if video:
        target_poses = archimedean_spiral(radius=VIDEO_RADIUS, num_points=500)[1:-1]  # don't include the first and last pose because they don't work really well (nan values)
        target_poses = torch.from_numpy(target_poses).to(torch.float32).to(device)  # (NUM_IMAGES, 4, 4)

    # load trained GeNVS
    net = GeNVS(None, cam_weighting_method=cam_weighting)
    with open(ckpt_path, "rb") as f:
        tmp_net = pickle.load(f)["ema"].cpu()
        net.denoiser = tmp_net.denoiser
        net.load_state_dict(tmp_net.state_dict())
        del tmp_net
    net = net.to(device)
    net.eval()
    
    # figure out real batch size to use
    if video and cond_method == "autoreg":
        batch_size = 1  # if autoreg, batch size must be 1 because next view is conditioned on prev generated view
        prev_output = [] 
        prev_poses = []

    # encode input views
    inputs_encoded = net.pixel_nerf_net.encode_input_views(images[None],
                                                           poses[None])

    # prepare a constant latent to be used by denoiser
    rnd = StackedRandomGenerator(device, [0])
    latents = rnd.randn([1, 3, H, W], device=device)

    # start generating video frames
    video_frames = []
    all_target_views = torch.tensor(list(range(target_poses.shape[0])))
    with torch.no_grad():
        for target_views in tqdm(torch.split(all_target_views, batch_size)):
            # 1. render feature images at target_views
            feature_images = net.pixel_nerf_net(inputs_encoded, target_poses[target_views][None],
                                                focal, z_near, z_far)[0]  # (len(target_views), 16, H, W)
            # 2. use feature images and denoiser to generate novel views
            sampler_fn = edm_sampler if denoising_steps > 1 else one_step_edm_sampler
            novel_images = sampler_fn(net.denoiser, latents.expand(len(target_views), -1, -1, -1), feature_images, None, num_steps=denoising_steps)  # (len(target_views), 3, H, W)
            novel_images = torch.clamp(novel_images, min=-1, max=1)
            # 3. save video frame(s)
            new_frames = torch.cat((feat_img_processor(feature_images), novel_images), dim=-1)
            video_frames.append(new_frames)

            # 4. if autoregressive, encode again for next iter
            if video and cond_method == "autoreg":
                prev_output.append(novel_images)  # appended (1, 3, H, W)
                prev_poses.append(target_poses[target_views])  # appended (1, 4, 4)
                # prepare images to encode
                to_encode_imgs = []
                to_encode_poses = []
                to_encode_imgs.append(images)  # original input images
                to_encode_poses.append(poses)
                to_encode_imgs.append(novel_images)  # last output
                to_encode_poses.append(target_poses[target_views])
                # 5 random previous outputs
                if len(prev_output) > 1:
                    indices = np.random.choice(list(range(len(prev_output) - 1)), replace=False, size=min(len(prev_output) - 1, 5))
                    for i in indices:
                        to_encode_imgs.append(prev_output[i])
                        to_encode_poses.append(prev_poses[i])
                # concatenate and finally encode
                to_encode_imgs = torch.cat(to_encode_imgs, dim=0).to(torch.float32)
                to_encode_poses = torch.cat(to_encode_poses, dim=0).to(torch.float32)
                inputs_encoded = net.pixel_nerf_net.encode_input_views(to_encode_imgs[None],
                                                                    to_encode_poses[None])
    
    # save the frames as video or individual images
    video_frames = torch.cat(video_frames, dim=0)  # (len(all_target_views), 3, H, W*2)
    # convert from float tensor to uint8 numpy array
    video_frames = torch.permute(video_frames, (0, 2, 3, 1))
    video_frames = torch.clip(video_frames, min=-1, max=1)
    video_frames = video_frames * 127.5 + 127.5  # range 0 to 255
    video_frames = video_frames.cpu().detach().numpy().astype(np.uint8)  # (len(all_target_views), H, W*2, 3)
    if video:
        # save as video
        video_name = f"steps={denoising_steps}_cond_method={cond_method}.mp4"
        video_path = os.path.join(out_dir, video_name)
        imageio.mimwrite(video_path,
                        video_frames,
                        fps=30,
                        quality=8)
        print("Successfully wrote video to", video_path)
    else:
        pred_out_dir = os.path.join(out_dir, f"steps={denoising_steps}", "pred")
        feat_img_out_dir = os.path.join(out_dir, f"steps={denoising_steps}", "feature_image")
        os.makedirs(pred_out_dir, exist_ok=True)
        os.makedirs(feat_img_out_dir, exist_ok=True)
        # save as individual images
        for i, pose_path in enumerate(target_pose_paths):
            txt_filename = os.path.basename(pose_path)
            png_filename = txt_filename.replace(".txt", ".png")
            imageio.imwrite(os.path.join(pred_out_dir, png_filename), video_frames[i, :, W:, :])
            imageio.imwrite(os.path.join(feat_img_out_dir, png_filename), video_frames[i, :, :W, :])
        print("Successfully saved images to", pred_out_dir)

def load_input(in_dir):
    # get important paths
    intrinsic_path = os.path.join(in_dir, "intrinsics.txt")
    rgb_paths = sorted(glob.glob(
        os.path.join(in_dir, "rgb", "*.png")))
    pose_paths = sorted(glob.glob(
        os.path.join(in_dir, "pose", "*.txt")))
    target_pose_paths = sorted(glob.glob(
        os.path.join(in_dir, "target_pose", "*.txt")))
    # checks
    assert len(rgb_paths) == len(pose_paths)
    for i in range(len(rgb_paths)):
        assert (os.path.basename(rgb_paths[i]).replace(".png", ".txt") ==
                os.path.basename(pose_paths[i]))

    # load intrinsics
    with open(intrinsic_path, "r") as file:
        lines = file.readlines()
        focal_len, ox, oy, _ = map(float, lines[0].split())  # ox, oy is offset for center of image
        height, width = map(float, lines[-1].split())
        focal_len = torch.tensor(2 * focal_len / height, dtype=torch.float32)  # normalize focal length, * 2 because -1 to + 1 is 2

    # load rgb and pose data
    all_imgs = []
    all_poses = []
    all_target_poses = []
    for rgb_path, pose_path in zip(rgb_paths, pose_paths):
        img = imageio.imread(rgb_path)[..., :3]
        img = IMG_TO_TENSOR_BALANCED(img)  # img now tensor with range -1 to +1

        pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        pose = torch.from_numpy(pose)
        # convert to our right-handed coordinate system
        # +x is right
        # +y is forward
        # +z is up
        pose = pose @ COORD_TRANSFORM
        
        all_imgs.append(img)
        all_poses.append(pose)

    # load target poses (if any)
    for pose_path in target_pose_paths:
        pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        pose = torch.from_numpy(pose)
        pose = pose @ COORD_TRANSFORM
        all_target_poses.append(pose)
    
    all_imgs = torch.stack(all_imgs)
    all_poses = torch.stack(all_poses)
    all_target_poses = None if len(all_target_poses) == 0 else torch.stack(all_target_poses)
    
    # return
    return {
        "images": all_imgs,
        "poses": all_poses,
        "target_poses": all_target_poses,
        "focal": focal_len,
        "z_near": Z_NEAR / focal_len.item(),
        "z_far": Z_FAR / focal_len.item(),
        "target_pose_paths": target_pose_paths,
    }

if __name__ == "__main__":
    main()

# written by JhihYang Wu <jhihyangwu@arizona.edu>
# generates multiple videos demonstrating ability of GeNVS to transform any domain to any domain
# to run simply execute: python any_dom_to_any_dom_1.py
# generated video columns are: input view(s), feature image, pred, gt

import os
import sys
import pickle
import torch
import imageio
from tqdm import tqdm
import numpy as np
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from genvs.data.cross_domain_srncars import CrossDomainNVS
from genvs.utils.utils import StackedRandomGenerator, edm_sampler, one_step_edm_sampler, feat_img_processor
from genvs.models.joint_genvs import JointGeNVS

DOMAINS = ["rgb", "sonar", "lidar_depth", "raysar"]
INPUT_VIEWS = [60, 120]  # which input view indices to use to generate novel views
SCENE_INDEX = 10  # which scene to use
COND_METHOD = "normal"  # which method to use for conditioning GeNVS
assert COND_METHOD in ["normal", "autoreg"]
DENOISING_STEPS = 25

# where to save output video
OUT_DIR = os.path.join("output_1", f"{SCENE_INDEX}_steps={DENOISING_STEPS}_cond_method={COND_METHOD}")
GPU_ID = 0  # which GPU to use
DATA_PATH = "/workspace/data/srncars/cars"
CKPT_PATH = "../weights/network-snapshot.pkl"

def main():
    # make out directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # setup CUDA device
    device = torch.device(f"cuda:{GPU_ID}")
    print(f"Using device: {device}")

    # load test scene
    dataset = CrossDomainNVS(DATA_PATH, distr="test", load_everything=True)
    scene_data = dataset[SCENE_INDEX]
    print("Scene:", scene_data["path"])
    images = {}
    for domain in DOMAINS:
        images[domain] = scene_data[f"{domain}_images"].to(device)  # (imgs_per_scene, C, H, W)
    poses = scene_data["input_poses"].to(device)  # (imgs_per_scene, 4, 4)
    focal = scene_data["focal"].item()
    z_near = scene_data["z_near"]
    z_far = scene_data["z_far"]
    _, _, H, W = images[list(images.keys())[0]].shape

    # load trained GeNVS
    net = JointGeNVS(None)
    with open(CKPT_PATH, "rb") as f:
        tmp_net = pickle.load(f)["ema"].cpu()
        for domain in DOMAINS:
            net.models[domain].denoiser = tmp_net.models[domain].denoiser
        net.load_state_dict(tmp_net.state_dict())
        del tmp_net
    with open(CKPT_PATH.replace("pkl", "pt").replace("network-snapshot", "training-state"), "rb") as f:
        dict_  = torch.load(f)["net"]
        net.load_state_dict(dict_, strict=False)
    net = net.to(device)
    net.eval()
    
    # generate all combinations of input and target domains
    for input_domain in DOMAINS:
        for target_domain in DOMAINS:
            print(f"Generating video from {input_domain} to {target_domain}")
            tmp = os.path.join(OUT_DIR, f"in_domain={input_domain}_out_domain={target_domain}.mp4")
            if os.path.exists(tmp):
                continue
            os.system(f"touch {tmp}")

            # figure out real batch size to use
            if COND_METHOD == "autoreg":
                batch_size = 1  # if autoreg, batch size must be 1 because next view is conditioned on prev generated view
                prev_output = []
                prev_poses = []
            else:
                assert COND_METHOD == "normal"
                batch_size = 1  # JointGeNVS asserts number of target views be 1 at all times

            # encode input views
            input_views = torch.tensor(INPUT_VIEWS)
            inputs_encoded = net.models[input_domain].pixel_nerf_net.encode_input_views(
                                                                images[input_domain][input_views][None],
                                                                poses[input_views][None])

            # prepare a constant latent to be used by denoiser
            rnd = StackedRandomGenerator(device, [0])
            latents = rnd.randn([1, 3, H, W], device=device)

            # start generating video frames
            video_frames = []
            all_target_views = torch.tensor(list(range(images[target_domain].shape[0])))
            with torch.no_grad():
                for target_views in tqdm(torch.split(all_target_views, batch_size)):
                    # 1. render feature images at target_views
                    range_angle_feat_img = {
                        "rgb": False,
                        "sonar": True,
                        "lidar_depth": False,
                        "raysar": False
                    }
                    feature_images = net.models["rgb"].pixel_nerf_net(inputs_encoded, poses[target_views][None],
                                                        focal, z_near, z_far, range_angle_feat_img=range_angle_feat_img[target_domain])[0]  # (len(target_views), 16, H, W)
                    # 2. use feature images and denoiser to generate novel views
                    sampler_fn = edm_sampler if DENOISING_STEPS > 1 else one_step_edm_sampler
                    novel_images = sampler_fn(net.models[target_domain].denoiser, latents.expand(len(target_views), -1, -1, -1), feature_images, None, num_steps=DENOISING_STEPS)  # (len(target_views), 3, H, W)
                    novel_images = torch.clamp(novel_images, min=-1, max=1)
                    # 3. save video frame(s)
                    gt_images = images[target_domain][target_views]
                    new_frames = torch.cat((feat_img_processor(feature_images), novel_images, gt_images), dim=-1)
                    video_frames.append(new_frames)

                    # 4. if autoregressive, encode again for next iter
                    assert COND_METHOD == "normal"
                    #if COND_METHOD == "autoreg":
                    #    prev_output.append(novel_images)  # appended (1, 3, H, W)
                    #    prev_poses.append(poses[target_views])  # appended (1, 4, 4)
                    #    # prepare images to encode
                    #    to_encode_imgs = []
                    #    to_encode_poses = []
                    #    to_encode_imgs.append(images[input_views])  # original input images
                    #    to_encode_poses.append(poses[input_views])
                    #    to_encode_imgs.append(novel_images)  # last output
                    #    to_encode_poses.append(poses[target_views])
                    #    # 5 random previous outputs
                    #    if len(prev_output) > 1:
                    #        indices = np.random.choice(list(range(len(prev_output) - 1)), replace=False, size=min(len(prev_output) - 1, 5))
                    #        for i in indices:
                    #            to_encode_imgs.append(prev_output[i])
                    #            to_encode_poses.append(prev_poses[i])
                    #    # concatenate and finally encode
                    #    to_encode_imgs = torch.cat(to_encode_imgs, dim=0).to(torch.float32)
                    #    to_encode_poses = torch.cat(to_encode_poses, dim=0).to(torch.float32)
                    #    inputs_encoded = net.pixel_nerf_net.encode_input_views(to_encode_imgs[None],
                    #                                                        to_encode_poses[None])
            
            # save video
            video_frames = torch.cat(video_frames, dim=0)  # (len(all_target_views), 3, H, W*3)
            # append still input view images
            input_view_frames = images[input_domain][input_views]  # (num_input_views, 3, H, W)
            input_view_frames = [input_view_frames[i] for i in range(input_view_frames.shape[0])]
            input_view_frames = torch.cat(input_view_frames, dim=-1)[None]  # (1, 3, H, W * num_input_views)
            input_view_frames = input_view_frames.expand(video_frames.shape[0], -1, -1, -1)  # (len(all_target_views), 3, H, W * num_input_views)
            video_frames = torch.cat((input_view_frames, video_frames), dim=-1)
            # convert from float tensor to uint8 numpy array
            video_frames = torch.permute(video_frames, (0, 2, 3, 1))
            video_frames = torch.clip(video_frames, min=-1, max=1)
            video_frames = video_frames * 127.5 + 127.5  # range 0 to 255
            video_frames = video_frames.cpu().detach().numpy().astype(np.uint8)
            # write to disk
            video_name = f"in_domain={input_domain}_out_domain={target_domain}.mp4"
            video_path = os.path.join(OUT_DIR, video_name)
            imageio.mimwrite(video_path,
                            video_frames,
                            fps=30,
                            quality=8)
            print("Successfully wrote video to", video_path)

if __name__ == "__main__":
    main()

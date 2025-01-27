# written by JhihYang Wu <jhihyangwu@arizona.edu>
# script for evaluating the performance of JointGeNVS

import click
import os
import sys
import pickle
import torch
import skimage.metrics
from tqdm import tqdm
import imageio
import numpy as np
import random
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
from genvs.data.cross_domain_srncars import CrossDomainNVS
from genvs.utils.utils import StackedRandomGenerator, edm_sampler, one_step_edm_sampler, img_tensor_to_npy
from genvs.models.joint_genvs import JointGeNVS

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# should not change
DOMAINS = ["rgb", "sonar", "lidar_depth", "raysar"]  # all domains it is trained on
NUM_INPUT_VIEWS = 3  # how many input views to use to generate novel views for each scene
NUM_TARGET_VIEWS = 3  # how many target views to test against for each scene

@click.command()
@click.option("--out_dir", help="Where to save results", required=True)
@click.option("--denoising_steps", help="How many denoising steps to use for diffusion model", required=True)
@click.option("--cam_weighting", help="Which camera weighting algorithm to use", required=True)
@click.option("--ckpt_path", help="Path to trained model", required=True)
@click.option("--gpu_id", help="Which GPU to use", required=True)
@click.option("--data_path", help="Path to dataset", required=True)
@click.option("--input_domains", help="Which input domain(s) to use", required=True, type=str)  # split by commas
@click.option("--target_domains", help="Which target domain(s) to use", required=True, type=str)

def main(**kwargs):
    kwargs["input_domains"] = kwargs["input_domains"].split(",")
    kwargs["target_domains"] = kwargs["target_domains"].split(",")

    # make out directory
    os.makedirs(kwargs["out_dir"], exist_ok=False)
    finish_file = open(os.path.join(kwargs["out_dir"], "finish.txt"), "a", buffering=1)

    # setup CUDA device
    device = torch.device(f"cuda:{int(kwargs['gpu_id'])}")
    print(f"Using device: {device}")

    # get dataset
    dataset = CrossDomainNVS(kwargs["data_path"], distr="test", load_everything=True)

    # load trained GeNVS
    net = JointGeNVS(None, cam_weighting_method=kwargs["cam_weighting"])
    with open(kwargs["ckpt_path"], "rb") as f:
        tmp_net = pickle.load(f)["ema"].cpu()
        for domain in DOMAINS:
            net.models[domain].denoiser = tmp_net.models[domain].denoiser
        net.load_state_dict(tmp_net.state_dict())
        del tmp_net
    with open(kwargs["ckpt_path"].replace("pkl", "pt").replace("network-snapshot", "training-state"), "rb") as f:
        dict_  = torch.load(f)["net"]
        net.load_state_dict(dict_, strict=False)
    net = net.to(device)
    net.eval()

    # prepare a constant latent to be used by denoiser
    rnd = StackedRandomGenerator(device, [0])
    _, _, H, W = dataset[0]["rgb_images"].shape
    latents = rnd.randn([1, 3, H, W], device=device)

    # evaluate on one scene at a time
    total_psnr = 0.0
    total_ssim = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            # load data
            scene_data = dataset.get_some_images(i, input_domains=kwargs["input_domains"], target_domains=kwargs["target_domains"],
                                                 num_input_views=NUM_INPUT_VIEWS, num_target_views=NUM_TARGET_VIEWS)
            poses = scene_data["input_poses"].to(device)  # (imgs_per_scene, 4, 4)
            focal = scene_data["focal"].item()
            z_near = scene_data["z_near"]
            z_far = scene_data["z_far"]

            input_imgs = scene_data["input_images"].to(device)  # (num_input_views, C, H, W)
            input_poses = scene_data["input_poses"].to(device)  # (num_input_views, 4, 4)

            target_imgs = scene_data["target_images"].to(device)  # (num_target_views, C, H, W)
            target_poses = scene_data["target_poses"].to(device)  # (num_target_views, 4, 4)

            input_info = []
            for idx, domain_info in zip(scene_data["input_indices"], scene_data["input_domain_info"]):
                input_info.append((domain_info["domain"], idx))
            target_info = []
            for idx, domain_info in zip(scene_data["target_indices"], scene_data["target_domain_info"]):
                target_info.append((domain_info["domain"], idx))

            obj_name = os.path.basename(scene_data["path"])
            print(f"OBJECT {i} OF {len(dataset)} {scene_data['path']}")
            os.makedirs(os.path.join(kwargs["out_dir"], obj_name), exist_ok=True)

            # encode input views
            in_encs = []
            for j in range(len(input_info)):
                dom, idx = input_info[j]
                enc, _ = net.models[dom].pixel_nerf_net.encode_input_views(input_imgs[j][None][None], None)
                in_encs.append(enc)
            in_encs = torch.cat(in_encs, dim=1)  # (1, num_input_views, feat_dim, num_depths, H, W)
            inputs_encoded = (in_encs, input_poses[None])

            # generate novel views
            ssims, psnrs = [], []
            for target_view_i in tqdm(range(target_imgs.shape[0])):
                # 1. render feature images at target_view_i
                range_angle_feat_img = {
                    "rgb": False,
                    "sonar": True,
                    "lidar_depth": False,
                    "raysar": False
                }
                dom, target_idx = target_info[target_view_i]
                feature_images = net.models["rgb"].pixel_nerf_net(inputs_encoded, target_poses[target_view_i][None][None].to(device),
                                                    focal, z_near, z_far, range_angle_feat_img=range_angle_feat_img[dom])[0]  # (1, 16, H, W)
                # 2. use feature images and denoiser to generate novel views
                steps = int(kwargs["denoising_steps"])
                sampler_fn = edm_sampler if steps > 1 else one_step_edm_sampler
                novel_images = sampler_fn(net.models[dom].denoiser, latents.expand(1, -1, -1, -1), feature_images, None, num_steps=steps)  # (1, 3, H, W)
                novel_images = torch.clamp(novel_images, min=-1, max=1).cpu()  # (1, 3, H, W)
                # 3. evaluate and save images
                pred = novel_images[0]  # (3, H, W)
                gt = target_imgs[target_view_i]  # (3, H, W)
                pred = img_tensor_to_npy(pred)  # (H, W, 3)
                gt = img_tensor_to_npy(gt)  # (H, W, 3)

                # get ssim and psnr
                cur_ssim = skimage.metrics.structural_similarity(
                    pred,
                    gt,
                    multichannel=True,
                    data_range=255,
                )
                cur_psnr = skimage.metrics.peak_signal_noise_ratio(
                    pred,
                    gt,
                    data_range=255
                )
                ssims.append(cur_ssim)
                psnrs.append(cur_psnr)
                # save images
                imageio.imwrite(os.path.join(kwargs["out_dir"], obj_name, f"{dom}_{target_idx:0>6}.png"),
                                pred)

            # record and print metrics
            ssim = sum(ssims) / len(ssims)
            psnr = sum(psnrs) / len(psnrs)
            total_psnr += psnr
            total_ssim += ssim
            cnt += 1
            print(
                "curr psnr",
                psnr,
                "ssim",
                ssim,
                "running psnr",
                total_psnr / cnt,
                "running ssim",
                total_ssim / cnt,
            )
            finish_file.write(
                "{} {} {}\n".format(obj_name, psnr, ssim)
            )
            tmp = [f"{dom}_{idx}" for dom, idx in input_info]
            finish_file.write(f"{' '.join(tmp)}\n")
    
    # final record and print
    print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
    finish_file.write(f"final_psnr {total_psnr / cnt} final_ssim {total_ssim / cnt}\n")
    finish_file.close()

if __name__ == "__main__":
    main()

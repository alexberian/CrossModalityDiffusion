# written by JhihYang Wu <jhihyangwu@arizona.edu>
# inspired by https://github.com/sxyu/pixel-nerf/blob/master/src/data/MultiObjectDataset.py
# dataset class for multi chair srnchairs dataset

import json
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import imageio

class SRNMultiChairs(Dataset):
    """
    Dataset class for multi chair SRNChairs dataset.
    Can be downloaded from multi_chair_*.zip https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR
    """

    def __init__(self, path, distr="train"):
        """
        Constructor for multi chair SRNChairs class.

        Args: 
            path (str): path to multi chair srnchairs dataset
            distr (str): which distribution of dataset to load (train, val, test)
        """
        assert distr in ["train", "val", "test"]

        self.base_path = os.path.join(path, distr)
        self.distr = distr
        self.transform_files = []
        for scene_folder in os.listdir(self.base_path):
            path = os.path.join(self.base_path, scene_folder, "transforms.json")
            if os.path.exists(path):
                self.transform_files.append(path)
        self.transform_files = sorted(self.transform_files)

        self.img_to_tensor_balanced = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.z_near = 4
        self.z_far = 9
        self.resolution = 128

    def __len__(self):
        """
        Returns number of scenes in this dataset distribution.
        """
        return len(self.transform_files)

    def __getitem__(self, index):
        """
        Loads all images, poses, etc of a scene.

        Returns:
            path (str): path to scene folder
            images (tensor): all images in the scene .shape=(N, 3, H, W)
            poses (tensor): all camera poses associated with images .shape=(N, 4, 4)
            focal (tensor): focal length of camera .shape=(1)
            z_near (float): z near of this dataset
            z_far (float): z far of this dataset
        """
        # get important paths
        transform_path = self.transform_files[index]
        scene_path = os.path.dirname(transform_path)

        with open(transform_path, "r") as file:
            transform = json.load(file)

        camera_angle_x = transform["camera_angle_x"]  # horizontal FOV in radians
        focal_len = 1 / np.tan(camera_angle_x / 2)  # normalized focal length
        focal_len = torch.tensor(focal_len, dtype=torch.float32)

        # load rgb and pose data
        all_imgs = []
        all_poses = []
        for frame in transform["frames"]:
            pose = np.array(frame["transform_matrix"])
            assert pose.shape == (4, 4)
            pose = torch.from_numpy(pose).to(torch.float32)

            img_filename = os.path.basename(frame["file_path"]) + "_obj.png"
            img = imageio.imread(os.path.join(scene_path, img_filename))  # (H, W, 4)
            img[img[..., 3] == 0] = 255  # convert transparent places to white background
            img = img[..., :3]  # first 3 channels (H, W, 3)
            img = self.img_to_tensor_balanced(img)  # img now tensor with range -1 to +1
            
            all_imgs.append(img)
            all_poses.append(pose)
        
        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        
        # return
        return {
            "path": scene_path,
            "images": all_imgs,
            "poses": all_poses,
            "focal": focal_len,
            "z_near": self.z_near / focal_len.item(),
            "z_far": self.z_far / focal_len.item(),
        }

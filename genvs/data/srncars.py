# written by JhihYang Wu <jhihyangwu@arizona.edu>
# inspired by https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py
# dataset class for srncars dataset

import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
import imageio
import numpy as np
import torch

class SRNCars(Dataset):
    """
    Dataset class for SRNCars dataset.
    Can be downloaded from srn_cars.zip https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR
    """
    
    def __init__(self, path, distr="train"):
        """
        Constructor for SRNCars class.

        Args: 
            path (str): path to srncars dataset
            distr (str): which distribution of dataset to load (train, val, test)
        """
        assert distr in ["train", "val", "test"]

        self.base_path = path + "_" + distr
        self.distr = distr
        self.intrinsics = sorted(glob.glob(
            os.path.join(self.base_path, "*", "intrinsics.txt")))

        self.img_to_tensor_balanced = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.coord_transform = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )
        self.z_near = 0.8
        self.z_far = 1.8
        self.resolution = 128

    def __len__(self):
        """
        Returns number of scenes in this dataset distribution.
        """
        return len(self.intrinsics)

    def __getitem__(self, index):
        """
        Loads all images, poses, etc of a scene.

        Returns:
            path (str): path to scene folder
            images (tensor): all images in the scene .shape=(N, 3, H, W)
            poses (tensor): all camera poses associated with images .shape=(N, 4, 4)
            focal (tensor): focal length of camera .shape=(1)
            offset (tensor): offset for center of camera .shape=(2)
            z_near (float): z near of this dataset
            z_far (float): z far of this dataset
        """
        # get important paths
        intrinsic_path = self.intrinsics[index]
        scene_path = os.path.dirname(intrinsic_path)
        rgb_paths = sorted(glob.glob(
            os.path.join(scene_path, "rgb", "*.png")))
        pose_paths = sorted(glob.glob(
            os.path.join(scene_path, "pose", "*.txt")))
        # checks
        assert len(rgb_paths) == len(pose_paths)
        for i in range(len(rgb_paths)):
            assert (os.path.basename(rgb_paths[i]).replace(".png", ".txt") ==
                    os.path.basename(pose_paths[i]))

        # load intrinsics
        with open(intrinsic_path, "r") as file:
            lines = file.readlines()
            focal_len, ox, oy, _ = map(float, lines[0].split())  # ox, oy is offset for center of image
            offset = torch.tensor([ox, oy], dtype=torch.float32)
            height, width = map(float, lines[-1].split())
            focal_len = torch.tensor(2 * focal_len / height, dtype=torch.float32)  # normalize focal length, * 2 because -1 to + 1 is 2

        # load rgb and pose data
        all_imgs = []
        all_poses = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img = self.img_to_tensor_balanced(img)  # img now tensor with range -1 to +1

            pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            pose = torch.from_numpy(pose)
            # convert to our right-handed coordinate system
            # +x is right
            # +y is forward
            # +z is up
            pose = pose @ self.coord_transform
            
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
            "offset": offset,
            "z_near": self.z_near / focal_len.item(),
            "z_far": self.z_far / focal_len.item(),
        }

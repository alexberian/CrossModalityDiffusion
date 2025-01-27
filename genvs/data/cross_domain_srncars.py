# written by JhihYang Wu and Alex Berian 2024
# inspired by https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py
# dataset class for cross-domain srncars dataset

import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
import imageio
import numpy as np
import torch
import random

class CrossDomainNVS(Dataset):
    """
    Dataset class for cross-domain NVS datasets. The dataset should be organized
    in a similar manner to the SRNCars dataset. Instead of just a "pose" and "rgb" 
    folder, any folder besides "pose" will be considered an image domain.
    """
    
    def __init__(   self,
                    path,
                    distr="train",
                    z_near=0.8,
                    z_far=1.8,
                    resolution=128,
                    domains=["rgb", "sonar", "lidar_depth", "raysar"],
                    load_everything=False,
                ):
        """
        Constructor for CrossDomainNVS class.

        Args: 
            path (str): path to the dataset
            distr (str): which distribution of dataset to load (train, val, test)
            domains (List[str]): list of domains that are available
            load_everything (bool): whether to return all views or randomly choose 3
        """
        self.base_path = path + "_" + distr
        self.intrinsics = sorted(glob.glob(
            os.path.join(self.base_path, "*", "intrinsics.txt")))

        self.img_to_tensor_balanced = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.coord_transform = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = z_near
        self.z_far = z_far
        self.resolution = resolution

        self.domains = domains
        self.load_everything = load_everything

    def _rand_domain(self):
        return random.choice(self.domains)

    def __len__(self):
        """
        Returns number of scenes in this dataset distribution.
        """
        return len(self.intrinsics)

    def __getitem__(self, index):
        """
        Loads some images, poses, etc of a scene.

        Returns:
            path (str): path to scene folder
            focal (tensor): focal length of camera .shape=(1)
            offset (tensor): offset for center of camera .shape=(2)
            z_near (float): z near of this dataset
            z_far (float): z far of this dataset

            (new from cross-domain)
            input_images (tensor): images to input for training .shape=(n, 3, H, W)
            target_images (tensor): image to try to pred .shape=(3, H, W)
            input_domain_info (List[dict]): n dictionaries that tell the domain info of the n input_images
            target_domain_info (dict): dictionary that tells the domain info of the target_image
            input_poses (tensor): camera poses associated with input_images (n, 4, 4)
            target_poses (tensor): camera pose associated with target_images (4, 4)
        """
        # get important paths
        intrinsic_path = self.intrinsics[index]
        scene_path = os.path.dirname(intrinsic_path)

        image_paths = {}
        for domain in self.domains:
            image_paths[domain] = sorted(glob.glob(
                os.path.join(scene_path, domain, "*.png")))

        pose_paths = sorted(glob.glob(
            os.path.join(scene_path, "pose", "*.txt")))

        # checks
        for domain in self.domains:
            assert len(image_paths[domain]) == len(pose_paths)
            for i in range(len(image_paths[domain])):
                assert(os.path.basename(image_paths[domain][i]).replace(".png", ".txt") ==
                        os.path.basename(pose_paths[i])), \
                        "There is a mismatch between image and pose filenames in scene %s domain %s" % (scene_path, domain)

        # load intrinsics
        with open(intrinsic_path, "r") as file:
            lines = file.readlines()
            focal_len, ox, oy, _ = map(float, lines[0].split())  # ox, oy is offset for center of image
            offset = torch.tensor([ox, oy], dtype=torch.float32)
            height, width = map(float, lines[-1].split())
            focal_len = torch.tensor(2 * focal_len / height, dtype=torch.float32)  # normalize focal length, * 2 because -1 to + 1 is 2

        # choose number of input views (1 - 3)
        num_input_views = 3  # this has to be fixed for dataloader to create a batch
        if self.load_everything:
            num_input_views = len(image_paths[self.domains[0]])
        # pick domain for n input images
        input_domain_info = [{"domain": self._rand_domain()} for _ in range(num_input_views)]
        # pick domain for target image
        target_domain_info = {"domain": self._rand_domain()}

        # choose which views to use as input
        views_per_scene = len(image_paths[self.domains[0]])
        input_indices = random.choices(list(range(views_per_scene)), k=num_input_views)
        if self.load_everything:
            input_indices = list(range(views_per_scene))
        # choose which view to use as output
        target_index = random.randint(0, views_per_scene-1)

        # load image data
        def load_proc_img(path):
            img = imageio.imread(path)
            if len(img.shape) == 3:  # (H, W, C)
                img = img[..., :3]
            else:
                assert len(img.shape) == 2  # (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            img = self.img_to_tensor_balanced(img)
            return img
        # input
        input_images = []
        for i, dom_info in zip(input_indices, input_domain_info):
            domain = dom_info["domain"]
            input_images.append(load_proc_img(image_paths[domain][i]))
        input_images = torch.stack(input_images)
        # target
        target_images = load_proc_img(image_paths[target_domain_info["domain"]][target_index])

        # load pose data
        def load_proc_pose(path):
            pose = np.loadtxt(path, dtype=np.float32).reshape(4, 4)
            pose = torch.from_numpy(pose)
            # convert to our right-handed coordinate system
            # +x is right
            # +y is forward
            # +z is up
            pose = pose @ self.coord_transform
            return pose
        # input
        input_poses = []
        for i in input_indices:
            input_poses.append(load_proc_pose(pose_paths[i]))
        input_poses = torch.stack(input_poses)
        # target
        target_poses = torch.tensor(load_proc_pose(pose_paths[target_index]))

        # return
        retval = {
            "path": scene_path,
            "focal": focal_len,
            "offset": offset,
            "z_near": self.z_near / focal_len.item(),
            "z_far": self.z_far / focal_len.item(),

            "input_images": input_images,
            "target_images": target_images,
            "input_domain_info": input_domain_info,
            "target_domain_info": target_domain_info,
            "input_poses": input_poses,
            "target_poses": target_poses,

            "input_indices": input_indices,
            "target_indices": [target_index],
        }

        # if load_everything, load all data from all domains
        if self.load_everything:
            for domain in self.domains:
                imgs = []
                for i in input_indices:
                    imgs.append(load_proc_img(image_paths[domain][i]))
                imgs = torch.stack(imgs)
                retval[domain + "_images"] = imgs

        return retval

    def get_some_images(self, index, input_domains, target_domains, num_input_views, num_target_views):
        """
        Loads some images, poses, etc of a scene.
        Randomly selects num_input_views from input_domains and num_target_views from target_domains.
        Mostly used by eval script to avoid loading all images.

        Returns:
            path (str): path to scene folder
            focal (tensor): focal length of camera .shape=(1)
            offset (tensor): offset for center of camera .shape=(2)
            z_near (float): z near of this dataset
            z_far (float): z far of this dataset

            (new from cross-domain)
            input_images (tensor): images to input for training .shape=(n, 3, H, W)
            target_images (tensor): image to try to pred .shape=(3, H, W)
            input_domain_info (List[dict]): n dictionaries that tell the domain info of the n input_images
            target_domain_info (dict): dictionary that tells the domain info of the target_image
            input_poses (tensor): camera poses associated with input_images (n, 4, 4)
            target_poses (tensor): camera pose associated with target_images (4, 4)
        """
        # get important paths
        intrinsic_path = self.intrinsics[index]
        scene_path = os.path.dirname(intrinsic_path)

        image_paths = {}
        for domain in self.domains:
            image_paths[domain] = sorted(glob.glob(
                os.path.join(scene_path, domain, "*.png")))

        pose_paths = sorted(glob.glob(
            os.path.join(scene_path, "pose", "*.txt")))

        # checks
        for domain in self.domains:
            assert len(image_paths[domain]) == len(pose_paths)
            for i in range(len(image_paths[domain])):
                assert(os.path.basename(image_paths[domain][i]).replace(".png", ".txt") ==
                        os.path.basename(pose_paths[i])), \
                        "There is a mismatch between image and pose filenames in scene %s domain %s" % (scene_path, domain)

        # load intrinsics
        with open(intrinsic_path, "r") as file:
            lines = file.readlines()
            focal_len, ox, oy, _ = map(float, lines[0].split())  # ox, oy is offset for center of image
            offset = torch.tensor([ox, oy], dtype=torch.float32)
            height, width = map(float, lines[-1].split())
            focal_len = torch.tensor(2 * focal_len / height, dtype=torch.float32)  # normalize focal length, * 2 because -1 to + 1 is 2

        # pick domain for n input images
        input_domain_info = [{"domain": random.choice(input_domains)} for _ in range(num_input_views)]
        # pick domain for target image
        target_domain_info = [{"domain": random.choice(target_domains)} for _ in range(num_target_views)]

        # choose which views to use as input
        views_per_scene = len(image_paths[self.domains[0]])
        input_indices = random.choices(list(range(views_per_scene)), k=num_input_views)
        # choose which view to use as output
        target_indices = random.choices(list(range(views_per_scene)), k=num_target_views)

        # load image data
        def load_proc_img(path):
            img = imageio.imread(path)
            if len(img.shape) == 3:  # (H, W, C)
                img = img[..., :3]
            else:
                assert len(img.shape) == 2  # (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            img = self.img_to_tensor_balanced(img)
            return img
        # input
        input_images = []
        for i, dom_info in zip(input_indices, input_domain_info):
            domain = dom_info["domain"]
            input_images.append(load_proc_img(image_paths[domain][i]))
        input_images = torch.stack(input_images)
        # target
        target_images = []
        for i, dom_info in zip(target_indices, target_domain_info):
            domain = dom_info["domain"]
            target_images.append(load_proc_img(image_paths[domain][i]))
        target_images = torch.stack(target_images)

        # load pose data
        def load_proc_pose(path):
            pose = np.loadtxt(path, dtype=np.float32).reshape(4, 4)
            pose = torch.from_numpy(pose)
            # convert to our right-handed coordinate system
            # +x is right
            # +y is forward
            # +z is up
            pose = pose @ self.coord_transform
            return pose
        # input
        input_poses = []
        for i in input_indices:
            input_poses.append(load_proc_pose(pose_paths[i]))
        input_poses = torch.stack(input_poses)
        # target
        target_poses = []
        for i in target_indices:
            target_poses.append(load_proc_pose(pose_paths[i]))
        target_poses = torch.stack(target_poses)

        # return
        retval = {
            "path": scene_path,
            "focal": focal_len,
            "offset": offset,
            "z_near": self.z_near / focal_len.item(),
            "z_far": self.z_far / focal_len.item(),

            "input_images": input_images,
            "target_images": target_images,
            "input_domain_info": input_domain_info,
            "target_domain_info": target_domain_info,
            "input_poses": input_poses,
            "target_poses": target_poses,

            "input_indices": input_indices,
            "target_indices": target_indices,
        }

        return retval

    def get_images_same_input_viewpoint(self, index, input_domains, target_domains, num_target_views):
        """
        Loads some images, poses, etc of a scene.
        Mostly used by eval script to avoid loading all images.

        num_input_views = len(input_domains)
        Loads num_input_views from same random viewpoint using domains in input_domains.
        Target views are still randomly selected from target_domains.

        Returns:
            path (str): path to scene folder
            focal (tensor): focal length of camera .shape=(1)
            offset (tensor): offset for center of camera .shape=(2)
            z_near (float): z near of this dataset
            z_far (float): z far of this dataset

            (new from cross-domain)
            input_images (tensor): images to input for training .shape=(n, 3, H, W)
            target_images (tensor): image to try to pred .shape=(3, H, W)
            input_domain_info (List[dict]): n dictionaries that tell the domain info of the n input_images
            target_domain_info (dict): dictionary that tells the domain info of the target_image
            input_poses (tensor): camera poses associated with input_images (n, 4, 4)
            target_poses (tensor): camera pose associated with target_images (4, 4)
        """
        num_input_views = len(input_domains)
        # get important paths
        intrinsic_path = self.intrinsics[index]
        scene_path = os.path.dirname(intrinsic_path)

        image_paths = {}
        for domain in self.domains:
            image_paths[domain] = sorted(glob.glob(
                os.path.join(scene_path, domain, "*.png")))

        pose_paths = sorted(glob.glob(
            os.path.join(scene_path, "pose", "*.txt")))

        # checks
        for domain in self.domains:
            assert len(image_paths[domain]) == len(pose_paths)
            for i in range(len(image_paths[domain])):
                assert(os.path.basename(image_paths[domain][i]).replace(".png", ".txt") ==
                        os.path.basename(pose_paths[i])), \
                        "There is a mismatch between image and pose filenames in scene %s domain %s" % (scene_path, domain)

        # load intrinsics
        with open(intrinsic_path, "r") as file:
            lines = file.readlines()
            focal_len, ox, oy, _ = map(float, lines[0].split())  # ox, oy is offset for center of image
            offset = torch.tensor([ox, oy], dtype=torch.float32)
            height, width = map(float, lines[-1].split())
            focal_len = torch.tensor(2 * focal_len / height, dtype=torch.float32)  # normalize focal length, * 2 because -1 to + 1 is 2

        # pick domain for n input images
        input_domain_info = [{"domain": dom} for dom in input_domains]
        # pick domain for target image
        target_domain_info = [{"domain": random.choice(target_domains)} for _ in range(num_target_views)]

        # choose which views to use as input
        views_per_scene = len(image_paths[self.domains[0]])
        input_indices = [random.choice(list(range(views_per_scene)))] * num_input_views
        # choose which view to use as output
        target_indices = random.choices(list(range(views_per_scene)), k=num_target_views)

        # load image data
        def load_proc_img(path):
            img = imageio.imread(path)
            if len(img.shape) == 3:  # (H, W, C)
                img = img[..., :3]
            else:
                assert len(img.shape) == 2  # (H, W)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            img = self.img_to_tensor_balanced(img)
            return img
        # input
        input_images = []
        for i, dom_info in zip(input_indices, input_domain_info):
            domain = dom_info["domain"]
            input_images.append(load_proc_img(image_paths[domain][i]))
        input_images = torch.stack(input_images)
        # target
        target_images = []
        for i, dom_info in zip(target_indices, target_domain_info):
            domain = dom_info["domain"]
            target_images.append(load_proc_img(image_paths[domain][i]))
        target_images = torch.stack(target_images)

        # load pose data
        def load_proc_pose(path):
            pose = np.loadtxt(path, dtype=np.float32).reshape(4, 4)
            pose = torch.from_numpy(pose)
            # convert to our right-handed coordinate system
            # +x is right
            # +y is forward
            # +z is up
            pose = pose @ self.coord_transform
            return pose
        # input
        input_poses = []
        for i in input_indices:
            input_poses.append(load_proc_pose(pose_paths[i]))
        input_poses = torch.stack(input_poses)
        # target
        target_poses = []
        for i in target_indices:
            target_poses.append(load_proc_pose(pose_paths[i]))
        target_poses = torch.stack(target_poses)

        # return
        retval = {
            "path": scene_path,
            "focal": focal_len,
            "offset": offset,
            "z_near": self.z_near / focal_len.item(),
            "z_far": self.z_far / focal_len.item(),

            "input_images": input_images,
            "target_images": target_images,
            "input_domain_info": input_domain_info,
            "target_domain_info": target_domain_info,
            "input_poses": input_poses,
            "target_poses": target_poses,

            "input_indices": input_indices,
            "target_indices": target_indices,
        }

        return retval

# test the class
def _test():
    datapath = "/workspace/data/srncars/cars"
    dataset = CrossDomainNVS(datapath, distr="train", domains=["rgb","sonar"])
    #print("Number of scenes:", len(dataset))
    #data = dataset[0]
    #print("First scene:", data)
    #print("shapes:")
    #for key in data:
    #    if type(data[key]) is torch.Tensor:
    #        print(key, data[key].shape)

    # test with dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for data in dataloader:
        print(data.keys())
        print("shapes:")
        for key in data:
            if type(data[key]) is torch.Tensor:
                print(key, data[key].shape)
        print("domain info:")
        num_input_views = torch.randint(1, 3+1, (1,)).item()  # 1 to 3
        print("chosen num_input_views:", num_input_views)
        input_imgs  = data["input_images"]  # (batch_size, imgs_per_scene, C, H, W)
        input_domain_info = data["input_domain_info"]
        target_domain_info = data["target_domain_info"]
        # 1.01. format input and target domain info to match shape comment above
        batch_size = input_imgs.shape[0]
        domain_info_keys = list(target_domain_info.keys())
        new_input_dom_info = [[{}] * num_input_views for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(num_input_views):
                for key in domain_info_keys:
                    new_input_dom_info[i][j][key] = input_domain_info[j][key][i]
        new_target_dom_info = [{} for _ in range(batch_size)]
        for i in range(batch_size):
            for key in domain_info_keys:
                new_target_dom_info[i][key] = target_domain_info[key][i]
        input_domain_info = new_input_dom_info  # (batch_size, imgs_per_scene) size 2d array of dictionaries of domain information
        target_domain_info = new_target_dom_info  # (batch_size,) size 1d array of dictionaries of domain information
        
        print(input_domain_info)
        print(target_domain_info)

        break

if __name__ == "__main__":
    _test()

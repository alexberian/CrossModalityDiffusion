# written by JhihYang Wu <jhihyangwu@arizona.edu>
# prep a eval out folder for getting FID score
# usage: python prep_eval_fid.py path_to_folder_with_finish.txt
# does nothing if already ran before

import sys
import os
from tqdm import tqdm
import click

GT_DATA_PATH = "/workspace/data/srncars/cars_test"

@click.command()
@click.option("--gpu_id", help="Which GPU to use", required=True)
@click.option("--path", help="Which path to evaluate", required=True)

def main(**kwargs):
    # get finish.txt path
    path = kwargs["path"]
    gpu_id = kwargs["gpu_id"]
    assert path[-1] != "/"
    if not os.path.exists(os.path.join(path, "finish.txt")):
        print(f"Bad path {path}")
        sys.exit(1)

    # get imgs to test on
    scene_names = get_scene_names(path)
    # check if we even need to copy images
    path1 = path + "_tmp_fid"
    path2 = os.path.join(path1, "pred")
    path3 = os.path.join(path1, "ground")
    final_cmd = f"python -m pytorch_fid --device cuda:{gpu_id} {path2} {path3}"
    if os.path.exists(path1):
        print("Please run")
        print(final_cmd)
        sys.exit(0)
    # make folders
    os.makedirs(path1, exist_ok=False)
    os.makedirs(path2, exist_ok=False)
    os.makedirs(path3, exist_ok=False)
    # put all pred and ground in different folders
    count = 0
    for scene_name in tqdm(scene_names):
        img_filenames = os.listdir(os.path.join(path, scene_name))
        for filename in img_filenames:
            assert filename.endswith(".png")
            tmp = filename.split("_")
            filename = tmp[-1]
            dom = "_".join(tmp[:-1])
            os.system(f"cp {os.path.join(path, scene_name, dom + '_' + filename)} {os.path.join(path2, f'{count}.png')}")
            os.system(f"cp {os.path.join(GT_DATA_PATH, scene_name, dom, filename)} {os.path.join(path3, f'{count}.png')}")
            count += 1
    # after images are copied
    print("Please run")
    print(final_cmd)

def get_scene_names(path):
    scene_names = []
    with open(os.path.join(path, "finish.txt")) as file:
        for line in file:
            line = line.split(" ")
            try:
                float(line[-1])
            except ValueError:
                continue
            if len(line) == 3:
                scene_names.append(line[0])
            else:
                assert line[0] == "final_psnr"
                return scene_names
    print("Bad finish.txt")
    sys.exit(1)

if __name__ == "__main__":
    main()

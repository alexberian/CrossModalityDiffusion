# written by JhihYang Wu <jhihyangwu@arizona.edu>
# run all same viewpoint experiments to see if having multiple
# sensors at the same viewpoint can improve the performance of GeNVS

import os

OUT_DIR = "final_same_viewpoint_results"
GPU_ID = 1
DOMAINS = ["rgb", "sonar", "lidar_depth", "raysar"]
CKPT_PATH = "../weights/network-snapshot.pkl"

# all subsets of domains
for input_domains in [["rgb"],
                      ["sonar"],
                      ["lidar_depth"],
                      ["raysar"],
                      ["rgb", "sonar"],
                      ["rgb", "lidar_depth"],
                      ["rgb", "raysar"],
                      ["sonar", "lidar_depth"],
                      ["sonar", "raysar"],
                      ["lidar_depth", "raysar"],
                      ["rgb", "sonar", "lidar_depth"],
                      ["rgb", "sonar", "raysar"],
                      ["rgb", "lidar_depth", "raysar"],
                      ["sonar", "lidar_depth", "raysar"],
                      ["rgb", "sonar", "lidar_depth", "raysar"]]:
    this_out_dir = os.path.join(OUT_DIR, f"in_domains={'-'.join(input_domains)}")
    print(this_out_dir)
    if not os.path.exists(this_out_dir):
        os.system(f"""
                    python eval_same_viewpoint_4.py \
                        --out_dir={this_out_dir} \
                        --denoising_steps=25 \
                        --cam_weighting='baseline_mean' \
                        --ckpt_path='{CKPT_PATH}' \
                        --gpu_id={GPU_ID} \
                        --data_path=/workspace/data/srncars/cars \
                        --input_domains={','.join(input_domains)} \
                        --target_domains={','.join(DOMAINS)} \
                """)
    os.system(f"python eval_fid_lpips_dists.py --path {this_out_dir} --gpu_id {GPU_ID}")

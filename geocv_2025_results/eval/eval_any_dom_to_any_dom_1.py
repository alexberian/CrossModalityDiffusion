# written by JhihYang Wu <jhihyangwu@arizona.edu>
# script for evaluating the performance of JointGeNVS for every combination of domains

import os

DOMAINS = ["rgb", "sonar", "lidar_depth", "raysar"]
OUT_DIR = "final_any_dom_any_dom_results"
GPU_ID = 0
CKPT_PATH = "../weights/network-snapshot.pkl"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # run select_dom_to_dom_2.py for every combination
    for input_domain in DOMAINS:
        for target_domain in DOMAINS:
            print(f"Evaluating JointGeNVS from {input_domain} to {target_domain}")
            comb_out_dir = os.path.join(OUT_DIR, f"{input_domain}_to_{target_domain}")
            if not os.path.exists(comb_out_dir):
                os.system(f"""
                    python select_dom_to_dom_2.py \
                        --out_dir={comb_out_dir} \
                        --denoising_steps=25 \
                        --cam_weighting='baseline_mean' \
                        --ckpt_path='{CKPT_PATH}' \
                        --gpu_id={GPU_ID} \
                        --data_path=/workspace/data/srncars/cars \
                        --input_domains={input_domain} \
                        --target_domains={target_domain}
                """)
            os.system(f"python eval_fid_lpips_dists.py --path {comb_out_dir} --gpu_id {GPU_ID}")

if __name__ == "__main__":
    main()

# written by JhihYang Wu <jhihyangwu@arizona.edu>
# run all data fusion experiments

import os

OUT_DIR = "final_data_fusion_results"
GPU_ID = 1
DOMAINS = ["rgb", "sonar", "lidar_depth", "raysar"]
CKPT_PATH = "../weights/network-snapshot.pkl"

for num_input_views in [2, 3, 4, 5]:
    for use_input_indices in [[i] for i in range(num_input_views)] + [list(range(num_input_views))]:
        this_out_dir = os.path.join(OUT_DIR, f"num_in_views={num_input_views}_use={'-'.join(map(str, use_input_indices))}")
        print(this_out_dir)
        if not os.path.exists(this_out_dir):
            os.system(f"""
                        python eval_data_fusion_3.py \
                            --out_dir={this_out_dir} \
                            --denoising_steps=25 \
                            --cam_weighting='baseline_mean' \
                            --ckpt_path='{CKPT_PATH}' \
                            --gpu_id={GPU_ID} \
                            --data_path=/workspace/data/srncars/cars \
                            --input_domains={','.join(DOMAINS)} \
                            --target_domains={','.join(DOMAINS)} \
                            --use_input_indices={','.join(map(str, use_input_indices))} \
                            --num_input_views={num_input_views}
                    """)
        os.system(f"python eval_fid_lpips_dists.py --path {this_out_dir} --gpu_id {GPU_ID}")

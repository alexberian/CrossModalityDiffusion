#!/bin/sh

python inference.py \
    --in_dir=input \
    --out_dir=output \
    --denoising_steps=1 \
    --cam_weighting='baseline_mean' \
    --ckpt_path=../training_runs/00000-uncond-ddpmpp-edm-gpus8-batch96-fp32/network-snapshot-004000.pkl \
    --gpu_id=0 \
    --batch_size=16 \
    --cond_method='autoreg' \
    #--video \

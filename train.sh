#!/bin/sh

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --standalone \
            --nproc_per_node=6 train.py --outdir=training_runs \
                                        --cond=0 \
                                        --lr=2e-5 \
                                        --arch=ddpmpp \
                                        --batch=72 \
                                        --batch-gpu=12 \
                                        --duration=200 \
                                        --log_freq=1000 \
                                        --tick=400 \
                                        --snap=5 \
                                        --dump=5 \
                                        --seed=321 \
                                        --data=/workspace/data/srncars/cars \
                                        --cam_weighting_method=baseline_mean \

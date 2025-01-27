# Unofficial Implementation of GeNVS
University of Arizona's implementation of GeNVS  
Alex Berian, Daniel Brignac, JhihYang Wu, Natnael Daba, Abhijit Mahalanobis  
Built off of [EDM](https://github.com/NVlabs/edm) from Nvidia  
**Please consider citing our paper CrossModalityDiffusion if you found our code useful.**

---
### Files and Purpose
```
dnnlib/ -- unmodified code from EDM
torch_utils/ -- unmodified code from EDM
training/ -- modified code from EDM, where training loop is stored
genvs/ -- new code needed for GeNVS

inference/ -- where inference script is stored

train.py -- script to initiate training of GeNVS
train.sh -- to easily run train.py
```

---
### Environment Setup
```
conda env create -f environment.yml
conda activate genvs
```

---
### Downloading Data to Train On
`srn_cars.zip` from https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR  
Hosted by authors of pixelNeRF  

---
### Downloading our Trained Weights
Download https://drive.google.com/file/d/1ESsrKM99MK3wBPmgSNthaGNdPXUM1WCm/view?usp=sharing  
unzip such that `/training_runs/00000-uncond-ddpmpp-edm-gpus8-batch96-fp32/network-snapshot-004000.pkl` exists  

---
### Inference our Trained Model
```
cd inference/
./inference.sh
```
Customize options inside inference.sh to your preference  
if `--video` flag is present, inference.py generates video instead of novel views at target poses  

To improve image quality, increase the number of inferences in the denoising diffusion process with `--denoising_steps=30`

How to format input for inference:  
`inference/input/rgb` images to input into GeNVS  
`inference/input/pose` corresponding poses of those images  
`inference/input/target_pose` poses of novel views that you want GeNVS to generate  


---
### Training GeNVS
```
./train.sh
```
tensorboard logging stored inside `training_runs/run_name/`  
to launch web server for visualize training progress:  
```
tensorboard --logdir=training_runs/run_name/
```
if your training computer can't host web apps due to firewall, download the tensorboard log file(s) inside `training_runs/run_name/` and run tensorboard on your PC  

Training progress should look something like https://wandb.ai/natedaba/GeNVS/runs/6fq07yj0  
You might need to change `--seed` a few times in `train.sh` if it seems like GeNVS is not training at all  

To specify which GPUs on your system you'd like to use, adjust the environment variable `CUDA_VISIBLE_DEVICES` in the `train.sh` script as well as the option  `--nproc_per_node=X` according to the number of GPUs you want to use. 
Example: 
```
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --standalone \
          --nproc_per_node=7 train.py --whatever_options_you_want\
```
Select the batch size according to the amount of RAM your GPUs have and the number of GPUs. It must be divisible by the number of GPUs.  

---
### Warnings for Custom Dataset
Your custom dataset class in `genvs/data/your_dataset.py` should return  
`focal` as: normalized focal length (do necessary computation inside getitem method)  
`z_near` as: absolute `z_near` / normalized focal length  
`z_far` as: absolute `z_far` / normalized focal length  

Make sure to briefly skim the first few lines of any script for dataset specific constants such as:  
Radius of sphere at which the images were taken/to be inferenced  
`z_near` `z_far` values  
`coord_transform`  

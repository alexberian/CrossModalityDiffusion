# CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation

Alex Berian, Daniel Brignac, JhihYang Wu, Natnael Daba, Abhijit Mahalanobis  
University of Arizona

![Teaser](readme_imgs/teaser.gif)

Accepted to WACV 2025 GeoCV Workshop  
arXiv: https://arxiv.org/abs/2501.09838

Abstract: Geospatial imaging leverages data from diverse sensing modalities-such as EO, SAR, and LiDAR, ranging from ground-level drones to satellite views. These heterogeneous inputs offer significant opportunities for scene understanding but present challenges in interpreting geometry accurately, particularly in the absence of precise ground truth data. To address this, we propose CrossModalityDiffusion, a modular framework designed to generate images across different modalities and viewpoints without prior knowledge of scene geometry. CrossModalityDiffusion employs modality-specific encoders that take multiple input images and produce geometry-aware feature volumes that encode scene structure relative to their input camera positions. The space where the feature volumes are placed acts as a common ground for unifying input modalities. These feature volumes are overlapped and rendered into feature images from novel perspectives using volumetric rendering techniques. The rendered feature images are used as conditioning inputs for a modality-specific diffusion model, enabling the synthesis of novel images for the desired output modality. In this paper, we show that jointly training different modules ensures consistent geometric understanding across all modalities within the framework. We validate CrossModalityDiffusion's capabilities on the synthetic ShapeNet cars dataset, demonstrating its effectiveness in generating accurate and consistent novel views across multiple imaging modalities and perspectives.

This is the official repository for our paper, CrossModalityDiffusion.  
Even though we mention GeNVS throughout our code base, it is our implementation of GeNVS (not official) so it may be different from what the authors of GeNVS have. Just like the authors of GeNVS, we built our project on top of [EDM](https://github.com/NVlabs/edm) from Nvidia. Check their repo out to understand what we changed and what we kept the same. For help running our code, please email jhihyangwu å† arizona ∂ø† edu.

# Environment setup

To start, create an environment using conda:
```
conda env create -f environment.yml
conda activate genvs
```

# Logging

To log training progress on wandb, fill in your wandb project information in training/training_loop.py and uncomment lines related to wandb.

# Data

We start by downloading pre-rendered ShapeNet cars images from https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR, hosted by the authors of pixelNeRF.  
Those images will be used for the EO sensing modality.

We download 3D facet files of ShapeNet cars from https://huggingface.co/ShapeNet and pass those 3D models into BLAINDER and RaySAR to generate LiDAR and SAR images.

We use the following directory structure to store our data.  
```
srncars
  - cars_train
    - 100715345ee54d7ae38b52b4ee9d36a3
      - lidar_depth
        - 000000.png
        ...
        - 000049.png
      - pose
        - 000000.txt
        ...
        - 000049.txt
      - raysar
        - 000000.png
        ...
        - 000049.png
      - rgb
      - sonar
      - intrinsics.txt
    - 100c3076c74ee1874eb766e5a46fceab
    - 10247b51a42b41603ffe0e5069bf1eb5
    ...
  - cars_val
  - cars_test
```

# Generating a video (inference)

First, download our pretrained weights from https://drive.google.com/drive/folders/1Xhqect85JzOyf8hVRUmNHurwY31SE7uk?usp=sharing, so that geocv_2025_results/weights/training-state.pt and geocv_2025_results/weights/network-snapshot.pkl exists.  

All video generation scripts are in `geocv_2025_results/video/` directory.  

# Evaluation (inference)

All evaluation code is in `geocv_2025_results/eval/` directory.

# Training

Modify `train.sh` to your liking and run the script to start training.  

Flag definitions can be found in train.py. Checkpoints are saved every few thousand kilo-iterations.  

# BibTeX

```
@misc{berian2025crossmodalitydiffusionmultimodalnovelview,
      title={CrossModalityDiffusion: Multi-Modal Novel View Synthesis with Unified Intermediate Representation}, 
      author={Alex Berian and Daniel Brignac and JhihYang Wu and Natnael Daba and Abhijit Mahalanobis},
      year={2025},
      eprint={2501.09838},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.09838}, 
}
```

# Acknowledgements

Our code is built on top of EDM from Nvidia: https://github.com/NVlabs/edm  
Parts of the code were inspired by pixelNeRF: https://github.com/sxyu/pixel-nerf  
The architecture of our pipeline is heavily based off of GeNVS: https://nvlabs.github.io/genvs/  

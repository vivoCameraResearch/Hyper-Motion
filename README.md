# HyperMotion: DiT-Based Pose-Guided Human Image Animation of Complex Motions
This repository is the official implementation of [HyperMotion](https://vivocameraresearch.github.io/hypermotion/)

<a href="https://arxiv.org/abs/2505.22977"><img src='https://img.shields.io/badge/arXiv-2505.22977-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>&nbsp;
<a href="https://vivocameraresearch.github.io/hypermotion/"><img src='https://img.shields.io/badge/Project-Page-Green' alt='GitHub'></a>&nbsp;
<a href="http://www.apache.org/licenses/LICENSE-2.0"><img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'></a>&nbsp;
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfWK4a7GqI-Yc8GIWcYmUcmZgdnI-vIYQZ1wrXJNQCrDtABQA/viewform?usp=header"><img src='https://img.shields.io/badge/Dataset-HypermotionX-Green' alt='GitHub'></a>&nbsp;

## 📣 News:
We'll be open-sourcing model weights, inference/train scripts, and methods for obtaining pose sequences in June 2025.
### The Open-HypermotionX training dataset and the HypermotionX bench are available at [this application link](https://docs.google.com/forms/d/e/1FAIpQLSfWK4a7GqI-Yc8GIWcYmUcmZgdnI-vIYQZ1wrXJNQCrDtABQA/viewform?usp=header)!!!
We will complete the review of the training dataset in the near future, and we will release the bench data to the applicants first.
## ✅ To-Do List for HyperMotion Release

- [✅] Release the Open-HypermotionX dataset
- [✅] Release the HypermotionX bench
- [✅] Release the source code
- [ ] Release the pretrained weights
- [ ] Release the testing file
- [ ] Release the training file
      
## ⚙ Install
We have verified this repo execution on the following environment:

The detailed of Windows:
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G & Nvidia-3090 24G

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

We need about 60GB available on disk (for saving weights), please check!

### ⚠ About H20 GPU's bug, if yor meet the error about bf16 for train or inference please reference the [Link](https://github.com/vllm-project/vllm/issues/4392)

## 😁 Inference


## 😘 Acknowledgement
Our code is modified based on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main). We adopt [Wan2.1-I2V-14B](https://github.com/Wan-Video/Wan2.1) as the base model. And we referenced [UniAnimate](https://github.com/ali-vilab/UniAnimate), Our data is inherited from [Motion-X](https://github.com/IDEA-Research/Motion-X), then we use [EasyOCR](https://github.com/JaidedAI/EasyOCR) to deal with videos, and [InternVL2](https://github.com/OpenGVLab/InternVL) to generate text dic. We use [Xpose](https://github.com/IDEA-Research/X-Pose) to generate pose video. Thanks to all the contributors! Special thanks to [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main), without that work as a foundation there would be no work for us!

<pre> 
  @misc{xu2025hypermotion,
    title={HyperMotion: DiT-Based Pose-Guided Human Image Animation of Complex Motions}, 
    author={Shuolin Xu and Siming Zheng and Ziyi Wang and HC Yu and Jinwei Chen and Huaqi Zhang and Bo Li and Peng-Tao Jiang},
    year={2025},
    eprint={2505.22977},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2505.22977}, 
  }
</pre>

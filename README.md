# HyperMotion: DiT-Based Pose-Guided Human Image Animation of Complex Motions
This repository is the official implementation of [HyperMotion](https://vivocameraresearch.github.io/hypermotion/)

<a href="https://arxiv.org/abs/2505.22977"><img src='https://img.shields.io/badge/arXiv-2505.22977-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>&nbsp;
<a href='https://vivocameraresearch.github.io/hypermotion/'>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'></a>
<a href="http://www.apache.org/licenses/LICENSE-2.0"><img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'></a>&nbsp;
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfWK4a7GqI-Yc8GIWcYmUcmZgdnI-vIYQZ1wrXJNQCrDtABQA/viewform?usp=header"><img src="https://img.shields.io/static/v1?label=HypermotionX&message=Dataset&color=green"></a> &ensp;

## 📣 News:
We'll be open-sourcing model weights, inference/train scripts, and methods for obtaining pose sequences in June 2025.
### The Open-HypermotionX training dataset and the HypermotionX bench are available at [this application link](https://docs.google.com/forms/d/e/1FAIpQLSfWK4a7GqI-Yc8GIWcYmUcmZgdnI-vIYQZ1wrXJNQCrDtABQA/viewform?usp=header)!!!
We will complete the review of the training dataset in the near future, and we will release the bench data to the applicants first.
## ✅ To-Do List for HyperMotion Release

- [✅] Release the Open-HypermotionX dataset
- [✅] Release the HypermotionX bench
- [✅] Release the source code
- [✅] Release the inference file
- [ ] Release the pretrained weights
- [ ] Release the training file & details (wan-2.1_14B 8*H20 96G sft)

## 👀 How to get Open-HyperMotionX training dataset from [Motion-X](https://github.com/IDEA-Research/Motion-X) dataset
We are so sorry that due to force majeure caused by company's regulations, we can't upload the processed training set‘s videos directly, but we will give you the complete ways to get the HypermotionX training data from Motion-X.  Including video name ID, original pose annotation, **Follow these steps to process the Motion-X dataset:**
### 1. Download Motion-X Dataset (The completed form will be sent immediately)
Please fill out [this form](https://docs.google.com/forms/d/e/1FAIpQLSeb1DwnzGPxXWWjXr8cLFPAYd3ZHlWUtRDAzYoGvAKmS4uBlA/viewform) to request authorization to use Motion-X for non-commercial purposes. Then you will receive an email and please download the motion and text labels from the provided downloading links. The pose texts can be downloaded from [here](https://drive.google.com/file/d/168ja-oBTHM0QDKFIcRriQFPew5gUlZkQ/view?usp=sharing).

- Please collect them as the following directory structure, We only use the following parts of the data:
```
../Motion-X++ 

├──  video
  ├── perform.zip
  ├── music.zip
  ├── Kungfu.zip
  ├── idea400.zip
  ├── humman.zip
  ├── haa500.zip
  ├── animation.zip
  ├── fitness.zip(no need)
├──  text
  ├── wholebody_pose_description(no need)
  ├── semantic_label
    ├── perform.zip
    ├── music.zip
    ├── Kungfu.zip
    ├── idea400.zip
    ├── humman.zip
    ├── haa500.zip
    ├── animation.zip
├── motion
  ├──  motiion_generation(no need)
  ├──  mesh_recovery(no_need)
  ├──  keypoints
    ├── perform.zip
    ├── music.zip
    ├── Kungfu.zip
    ├── idea400.zip
    ├── humman.zip
    ├── haa500.zip
    ├── animation.zip
```
Unzip all files.
### 2. Filter the required source video based on the video ID list provided
```
cd train_data_processing
python fetch_videos_by_id.py \
  --json ./video_metadata.json \
  --source /data/motionX/video/ \
  --target /data/datasets/filtered_videos/ \
  --ext .mp4
```
### 3. Filter the required source kepoints files based on the json ID list provided

## ⚙ Install
We have verified this repo execution on the following environment:

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1 & 12.3
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

```shell
# python==3.12.9 cuda==12.3 torch==2.2
conda create -n hypermotion python==3.12.9
conda activate hypermotion
pip install -r requirements.txt
```
⚠ If you encounter an error while installing Flash Attention, please [**manually download**](https://github.com/Dao-AILab/flash-attention/releases) the installation package based on your Python version, CUDA version, and Torch version, and install it using `pip install flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`.

⚠ We need about 60GB available on disk (for saving weights), please check!

⚠ But just for inference only need < 24GB, even 16GB! So single RTX4090 is enough.

⚠ About H20 GPU's bug, if yor meet the error about bf16 for train or inference please reference the [Link](https://github.com/vllm-project/vllm/issues/4392)

## ⬇️ Checkpoint download

hyper-wan2.1-14B(asap,Review and upload will take some time)

## 😁 Inference
### First step
- Go to scripts/inference.py and set the path of model weights and input conditions correctly.
```
# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "........./model/hypermotion_14B" # model checkpoints
.......
# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
control_video           = "......./test.mp4" # guided pose video
ref_image               = "......./image.jpg" # reference image
```
### Second step
- Running inference script.
```
python inference.py
```
- Batch inference script.
```
json_path = "............../inference_config.json"
save_path = "......../sample/results"
config_path = "config/wan2.1/wan_civitai.yaml"
model_name = "............../hypermotion_14B"
```
```
python inference_batch.py
```

## 😘 Acknowledgement
Our code is modified based on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main). We adopt [Wan2.1-I2V-14B](https://github.com/Wan-Video/Wan2.1) as the base model. And we referenced [UniAnimate](https://github.com/ali-vilab/UniAnimate), Our data is inherited from [Motion-X](https://github.com/IDEA-Research/Motion-X), then we use [EasyOCR](https://github.com/JaidedAI/EasyOCR) to deal with videos, and [InternVL2](https://github.com/OpenGVLab/InternVL) to generate text dic. We use [Xpose](https://github.com/IDEA-Research/X-Pose) to generate pose video. Thanks to all the contributors! Special thanks to [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main), without that work as a foundation there would be no work for us!

## 🌏 Citation
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

import os
import sys
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hyper.dist import set_multi_gpus_devices
from hyper.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from hyper.models.cache_utils import get_teacache_coefficients
from hyper.pipeline import WanhyperPipeline
from hyper.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from hyper.utils.lora_utils import merge_lora, unmerge_lora
from hyper.utils.utils import (filter_kwargs, get_video_to_video_latent,
                                    save_videos_grid)

json_path = "............../inference_config.json"
save_path = "......../sample/results"
config_path = "config/wan2.1/wan_civitai.yaml"
model_name = "............../hypermotion_14B"
transformer_path = None
vae_path = None
lora_path = None

guidance_scale = 6.0
seed = 43
num_inference_steps = 50
lora_weight = 0.55
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，运镜，镜头移动，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，残缺的脚，没有脚的腿，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，模糊，突变，变形，失真"

device = set_multi_gpus_devices(1, 1)
weight_dtype = torch.bfloat16
GPU_memory_mode = "sequential_cpu_offload"
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False
enable_riflex = False
riflex_k = 6

config = OmegaConf.load(config_path)
transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).eval()
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype).eval()
scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
pipeline = WanhyperPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
)

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",])
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
else:
    pipeline.to(device=device)

if enable_teacache:
    coefficients = get_teacache_coefficients(model_name)
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

generator = torch.Generator(device=device).manual_seed(seed)
if lora_path:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)

with open(json_path, 'r') as f:
    cases = json.load(f)
os.makedirs(save_path, exist_ok=True)

processed_files = []
if os.path.exists(save_path):
    # 获取已经处理过的文件名
    processed_files = [os.path.splitext(f)[0] for f in os.listdir(save_path) if f.endswith('.mp4')]
    print(f"找到{len(processed_files)}个已处理的文件")

for idx, case in enumerate(cases):
    control_video_path = case['control_file_path']
    name = os.path.splitext(os.path.basename(control_video_path))[0]
    if name in processed_files:
        print(f"[{idx+1}/{len(cases)}] 跳过已处理的文件: {name}")
        continue
    ref_image_path = case['file_path']
    frame_count = case['frame_count']
    fps_in = case['fps']
    width = case['width']
    height = case['height']
    prompt = f"Static shot, no camera movement, {case.get('text', ' ')}"#case.get('text', " ")  # 使用每个样本自己的文本
    sample_size = [height, width]

    video_length = frame_count
    fps = fps_in

    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    if enable_riflex:
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)

    input_video, _, ref_image_tensor, clip_image_tensor = get_video_to_video_latent(
        control_video_path,
        video_length=video_length,
        sample_size=sample_size,
        fps=fps,
        ref_image=ref_image_path
    )

    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            control_video=input_video,
            ref_image=ref_image_tensor,
            clip_image=clip_image_tensor
        ).videos

    sample = sample[:, :, 1:]  # 去掉首帧
    name = os.path.splitext(os.path.basename(control_video_path))[0]
    out_path = os.path.join(save_path, f"{name}.mp4")
    save_videos_grid(sample, out_path, fps=fps)
    print(f"[{idx+1}/{len(cases)}] Saved to {out_path}")

if lora_path:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)


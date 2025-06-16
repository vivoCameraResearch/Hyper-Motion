import argparse
import os
import torch
from natsort import natsorted
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import custom utilities
from utils.video_dataset import VideoDataset, collate_fn

def recaption_batch_video(llm, batch_video_frames, prompt, sampling_params):
    """Generate descriptions for a batch of video frames"""
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": video_frames
            },
        }
        for video_frames in batch_video_frames
    ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    batch_output = []
    for o in outputs:
        generated_text = o.outputs[0].text
        batch_output.append(generated_text)

    return batch_output

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate video descriptions with InternVL2.")
    parser.add_argument(
        "--video_folder",
        type=str,
        required=True,
        help="The folder containing video files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="The folder to save description txt files."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for vllm inference."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="The number of workers for the video dataset."
    )
    parser.add_argument(
        "--input_prompt",
        type=str,
        default="Describe this video in detail. Don't repeat."
    )
    parser.add_argument(
        "--frame_sample_method",
        type=str,
        choices=["mid", "uniform", "image"],
        default="uniform",
        help="Method to sample frames from videos"
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use (default: 0)"
    )
   
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
   
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
   
    # Get video file list
    video_files = []
    for root, _, files in os.walk(args.video_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
   
    # Sort video files for consistency
    video_files = natsorted(video_files)
   
    # Skip already processed videos
    processed_videos = []
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        txt_path = os.path.join(args.output_folder, f"{video_name}.txt")
        if os.path.exists(txt_path):
            processed_videos.append(video_path)
   
    if processed_videos:
        logger.info(f"Skipping {len(processed_videos)} already processed videos.")
        video_files = [v for v in video_files if v not in processed_videos]
   
    logger.info(f"Found {len(video_files)} videos to process.")
   
    if not video_files:
        logger.info("No videos to process. Exiting.")
        return
   
    # Create video dataset and dataloader
    video_dataset = VideoDataset(
        dataset_inputs={"video_path": video_files},
        video_folder="",  # video_files already contains full paths
        sample_method=args.frame_sample_method,
        num_sampled_frames=args.num_sampled_frames
    )
    video_loader = DataLoader(
        video_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
   
    # Initialize VLLM inference pipeline
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", None)
    tensor_parallel_size = torch.cuda.device_count() if CUDA_VISIBLE_DEVICES is None else len(CUDA_VISIBLE_DEVICES.split(","))
    logger.info(f"Using tensor_parallel_size={tensor_parallel_size} based on available devices.")
   
    # Load model from specified path
    logger.info(f"Loading model from: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": args.num_sampled_frames},
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
        quantization="awq",
        dtype="float16",
        mm_processor_kwargs={"max_dynamic_patch": 1}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
   
    # Prepare prompt
    placeholders = "".join(f"Frame{i}: <image>\n" for i in range(1, args.num_sampled_frames + 1))
    messages = [{'role': 'user', 'content': f"{placeholders}{args.input_prompt}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   
    # Set stop tokens
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(temperature=0, max_tokens=512, stop_token_ids=stop_token_ids)
   
    # Process videos and generate descriptions
    for batch in tqdm(video_loader, desc="Processing videos"):
        if len(batch) > 0:
            batch_video_path = batch["path"]
            batch_frame = batch["sampled_frame"]  # [batch_size, num_sampled_frames, H, W, C]
            batch_caption = recaption_batch_video(llm, batch_frame, prompt, sampling_params)
           
            # Save descriptions to individual txt files
            for video_path, caption in zip(batch_video_path, batch_caption):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                txt_path = os.path.join(args.output_folder, f"{video_name}.txt")
               
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
               
                logger.info(f"Saved description for {video_name}")
           
            # Clear GPU memory
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


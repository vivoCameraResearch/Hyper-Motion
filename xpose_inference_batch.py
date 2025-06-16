import argparse
import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import clip
import transforms as T
from models import build_model
from predefined_keypoints import *
from util import box_ops
from util.config import Config
from util.utils import clean_state_dict
from torch.utils.data import Dataset
from torchvision.ops import nms
from tqdm import tqdm

class VideoFrameDataset(Dataset):
    """Dataset for processing videos in a folder"""
    def __init__(self, video_folder):
        self.video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                           if f.lower().endswith(('.mp4', '.avi'))]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).split('.')[0]
        return video_path, video_name

def text_encoding(instance_names, keypoints_names, model, device):
    """Encode text descriptions for instances and keypoints using CLIP"""
    # Encode instance text
    ins_text_embeddings = []
    for cat in instance_names:
        instance_description = f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
        text = clip.tokenize(instance_description).to(device)
        text_features = model.encode_text(text)
        ins_text_embeddings.append(text_features)
    ins_text_embeddings = torch.cat(ins_text_embeddings, dim=0)

    # Encode keypoint text
    kpt_text_embeddings = []
    for kpt in keypoints_names:
        kpt_description = f"a photo of {kpt.lower().replace('_', ' ')}"
        text = clip.tokenize(kpt_description).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        kpt_text_embeddings.append(text_features)
    kpt_text_embeddings = torch.cat(kpt_text_embeddings, dim=0)

    return ins_text_embeddings, kpt_text_embeddings

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    """Load UniPose model from checkpoint"""
    args = Config.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded: {load_res}")
    _ = model.eval()
    return model

def get_unipose_output(model, frames, instance_text_prompt, keypoint_text_prompt, box_threshold, iou_threshold, cpu_only=False):
    """Process frames with UniPose model to detect instances and keypoints"""
    instance_list = instance_text_prompt.split(',')
    device = "cuda" if not cpu_only else "cpu"
   
    # Load CLIP model for text encoding
    clip_model, _ = clip.load("ViT-B/32", device=device)
    ins_text_embeddings, kpt_text_embeddings = text_encoding(instance_list, keypoint_text_prompt, clip_model, device)
   
    # Prepare target dictionary for model input
    target = {
        "instance_text_prompt": instance_list,
        "keypoint_text_prompt": keypoint_text_prompt,
        "object_embeddings_text": ins_text_embeddings.float(),
        "kpts_embeddings_text": torch.cat((kpt_text_embeddings.float(),
                                         torch.zeros(100 - kpt_text_embeddings.shape[0], 512, device=device)), dim=0),
        "kpt_vis_text": torch.cat((torch.ones(kpt_text_embeddings.shape[0], device=device),
                                 torch.zeros(100 - kpt_text_embeddings.shape[0], device=device)), dim=0)
    }
   
    # Move model and data to device
    model = model.to(device)
    frames = frames.to(device)
   
    # Process frames in batch
    with torch.no_grad():
        targets = [target for _ in range(frames.shape[0])]
        outputs = model(frames, targets)
   
    # Process model outputs
    all_filtered_boxes = []
    all_filtered_keypoints = []
    all_filtered_bbox_scores = []

    batch_size = outputs["pred_logits"].shape[0]

    for i in range(batch_size):
        # Extract predictions for current frame
        logits = outputs["pred_logits"][i].sigmoid()
        boxes = outputs["pred_boxes"][i]
        keypoints = outputs["pred_keypoints"][i][:, :2*len(keypoint_text_prompt)]
       
        # Filter by confidence threshold
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        keypoints_filt = keypoints.cpu().clone()

        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        bbox_score_filt = logits_filt.max(dim=1)[0]
        boxes_filt = boxes_filt[filt_mask]
        keypoints_filt = keypoints_filt[filt_mask]
       
        # Apply NMS
        keep_indices = nms(box_ops.box_cxcywh_to_xyxy(boxes_filt), bbox_score_filt, iou_threshold=iou_threshold)

        # Filter results
        filtered_boxes = boxes_filt[keep_indices]
        filtered_keypoints = keypoints_filt[keep_indices]
        filtered_bbox_scores = bbox_score_filt[keep_indices]

        all_filtered_boxes.append(filtered_boxes)
        all_filtered_keypoints.append(filtered_keypoints)
        all_filtered_bbox_scores.append(filtered_bbox_scores)
   
    return all_filtered_boxes, all_filtered_keypoints, all_filtered_bbox_scores

def parse_args():
    parser = argparse.ArgumentParser(description="UniPose keypoint detection for videos")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSON files")
    parser.add_argument("--instance_prompt", type=str, default="face", help="Instance text prompt")
    parser.add_argument("--keypoint_type", type=str, default="face", help="Keypoint skeleton type")
    parser.add_argument("--box_threshold", type=float, default=0.15, help="Box confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--frames_per_clip", type=int, default=8, help="Number of frames to process at once")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    parser.add_argument("--cpu_only", action="store_true", help="Use CPU only")
   
    return parser.parse_args()

def main():
    args = parse_args()
   
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
   
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.config, args.checkpoint, cpu_only=args.cpu_only)
   
    # Create dataset
    dataset = VideoFrameDataset(args.video_folder)
    print(f"Found {len(dataset)} videos to process")
   
    # Define transform
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
   
    # Get keypoint configuration
    if args.keypoint_type in globals():
        keypoint_dict = globals()[args.keypoint_type]
        keypoint_text_prompt = keypoint_dict.get("keypoints")
        keypoint_skeleton = keypoint_dict.get("skeleton")
    else:
        print(f"Keypoint type '{args.keypoint_type}' not found. Exiting.")
        return
   
    # Process each video
    for video_idx in tqdm(range(len(dataset)), desc="Processing videos"):
        video_path, video_name = dataset[video_idx]
        print(f"\nProcessing video: {video_name}")
       
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        detection_results = []
       
        with tqdm(desc="Frames", leave=False) as pbar:
            while True:
                frames = []
                for _ in range(args.frames_per_clip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame, _ = transform(frame, None)
                    frames.append(frame)
               
                if not frames:
                    break
               
                frames = torch.stack(frames, dim=0)
               
                # Get model predictions
                all_filtered_boxes, all_filtered_keypoints, all_filtered_bbox_scores = get_unipose_output(
                    model, frames, args.instance_prompt, keypoint_text_prompt,
                    args.box_threshold, args.iou_threshold, cpu_only=args.cpu_only
                )
               
                # Process results for each frame
                for i in range(len(frames)):
                    detected_results = {
                        "frame_id": frame_id,
                        "instances": []
                    }
                   
                    for bbox, bbox_scores, keypoints in zip(
                        all_filtered_boxes[i], all_filtered_bbox_scores[i], all_filtered_keypoints[i]
                    ):
                        keypoints_reshaped = [[keypoints[k].item(), keypoints[k+1].item()]
                                             for k in range(0, len(keypoints), 2)]
                       
                        instance = {
                            "boxes": bbox.cpu().numpy().tolist(),
                            "bbox_score": bbox_scores.cpu().numpy().tolist(),
                            "keypoints": keypoints_reshaped
                        }
                       
                        detected_results["instances"].append(instance)
                   
                    detection_results.append(detected_results)
                    frame_id += 1
                    pbar.update(1)
               
                if len(frames) < args.frames_per_clip:
                    break
       
        # Save results to JSON
        json_output_path = os.path.join(args.output_dir, f"{video_name}_{args.instance_prompt}.json")
        with open(json_output_path, 'w') as json_file:
            json.dump(detection_results, json_file, indent=4)
       
        cap.release()
        print(f"JSON saved to {json_output_path}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import easyocr
import argparse
from torchvision.datasets.utils import download_url
from tqdm import tqdm

def init_ocr_reader(root: str = "~/.cache/easyocr", device: str = "gpu"):
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)

    download_url(
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/easyocr/craft_mlt_25k.pth",
        root,
        filename="craft_mlt_25k.pth",
        md5="2f8227d2def4037cdb3b34389dcf9ec1",
    )

    ocr_reader = easyocr.Reader(
        lang_list=["en", "ch_sim"],
        gpu=(device == "gpu"),
        recognizer=False,
        verbose=False,
        model_storage_directory=root,
    )
    return ocr_reader

def process_video(input_video_path: str, output_video_path: str, ocr_reader, blur_strength=15):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_video_path)}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        horizontal_list, free_list = ocr_reader.detect(frame)
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        for box in horizontal_list[0]:
            xmin, xmax, ymin, ymax = box
            if xmax > xmin and ymax > ymin:
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, thickness=cv2.FILLED)

        for points in free_list[0]:
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points], 255)

        blurred_frame = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        result = frame.copy()
        result[mask == 255] = blurred_frame[mask == 255]

        out.write(result)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Batch process videos to blur text regions')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input video files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed videos')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to use for OCR')
    parser.add_argument('--blur_strength', type=int, default=15, help='Blur strength (odd number)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Initializing OCR model...")
    ocr_reader = init_ocr_reader(device=args.device)

    video_files = [f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"Found {len(video_files)} videos to process")

    for video_file in video_files:
        input_path = os.path.join(args.input_dir, video_file)
        output_path = os.path.join(args.output_dir, video_file)
        print(f"\nProcessing: {video_file}")
        process_video(input_path, output_path, ocr_reader, args.blur_strength)
        print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()

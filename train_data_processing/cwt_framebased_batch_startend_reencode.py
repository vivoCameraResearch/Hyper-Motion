#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import json
import numpy as np
import cv2
import pywt
import shutil
from scipy.signal import find_peaks, peak_widths

def read_keypoints_from_json(json_path, keypoint_type='body', max_points=17):
    """
    Read keypoint time series from JSON. Returns (sorted_fids, all_x, all_y).
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
        print(f"[Error] Cannot parse JSON: {json_path}, error: {e}")
        return [], [], []

    ann_list = data.get('annotations', [])
    if not ann_list:
        return [], [], []

    ann_dict = {}
    for ann in ann_list:
        fname = ann.get('file_name', '')
        base = os.path.basename(fname)
        frame_str = os.path.splitext(base)[0]
        try:
            frame_id = int(frame_str)
        except ValueError:
            continue
        ann_dict[frame_id] = ann

    sorted_fids = sorted(ann_dict.keys())
    if not sorted_fids:
        return [], [], []

    all_x = [[] for _ in range(max_points)]
    all_y = [[] for _ in range(max_points)]

    for fid in sorted_fids:
        ann = ann_dict[fid]
        kpts = np.array(ann.get(f"{keypoint_type}_kpts", []))
        if len(kpts) < max_points:
            for i in range(max_points):
                all_x[i].append(np.nan)
                all_y[i].append(np.nan)
            continue

        kpts = kpts[:max_points]
        for i in range(max_points):
            x, y, score = kpts[i]
            if score > 0:
                all_x[i].append(x)
                all_y[i].append(y)
            else:
                all_x[i].append(np.nan)
                all_y[i].append(np.nan)

    return sorted_fids, all_x, all_y


def compute_velocity(all_x, all_y, fps, joint_index=0):
    """
    Calculate velocity sequence speed(t) for a single joint. Returns np.array (length n-1).
    """
    x_seq = np.nan_to_num(all_x[joint_index])
    y_seq = np.nan_to_num(all_y[joint_index])
    n = len(x_seq)
    if n < 2:
        return np.array([])

    vx = np.diff(x_seq) * fps
    vy = np.diff(y_seq) * fps
    speed = np.sqrt(vx**2 + vy**2)
    return speed


def cwt_allfreq_energy(speed, fps, wavelet='morl', max_scale=128):
    """
    Perform Continuous Wavelet Transform (CWT), without frequency filtering,
    sum absolute coefficients for all scales => energy_all.
    Returns (energy_all, n_samples).
    """
    n_samples = len(speed)
    if n_samples < 1:
        return np.array([]), 0

    scales = np.arange(1, min(max_scale, n_samples))
    cwt_coeffs, _ = pywt.cwt(speed, scales, wavelet, sampling_period=1.0/fps)
    cabs = np.abs(cwt_coeffs)
    energy_all = np.sum(cabs, axis=0)  # shape=(n_samples,)
    return energy_all, n_samples


def remove_short_spikes(energy, min_width_frames=3):
    """
    Filter minimum peak width (frames), set peaks with width < min_width_frames to 0.
    """
    energy_reduced = energy.copy()
    peak_inds, _ = find_peaks(energy_reduced)
    widths, _, left_ips, right_ips = peak_widths(energy_reduced, peak_inds, rel_height=0.5)

    for i, w in enumerate(widths):
        if w < min_width_frames:
            li = int(np.floor(left_ips[i]))
            ri = int(np.ceil(right_ips[i]))
            energy_reduced[li:ri+1] = 0.0
    return energy_reduced


def find_best_window_in_frames(energy, window_frames):
    """
    Sliding window search for interval, returns (best_start, best_end).
    """
    N = len(energy)
    if window_frames >= N:
        return 0, N-1

    prefix_sum = np.cumsum(energy)
    prefix_sum = np.insert(prefix_sum, 0, 0)  # shape=(N+1,)

    best_sum = -1.0
    best_start = 0
    for start_idx in range(0, N - window_frames + 1):
        end_idx = start_idx + window_frames
        cur_sum = prefix_sum[end_idx] - prefix_sum[start_idx]
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_start = start_idx

    return best_start, best_start + window_frames - 1


def save_clipped_video_ffmpeg_reencode(input_video, output_video, start_frame, end_frame, fps):
    """
    Re-encode: -c:v libx264 -crf 23 -c:a aac -b:a 128k
    Allows precise trimming, no longer affected by keyframe alignment.
    """
    start_time = start_frame / fps
    end_time = (end_frame + 1) / fps

    cmd = (
        f'ffmpeg -y -i "{input_video}" '
        f'-ss {start_time} -to {end_time} '
        f'-c:v libx264 -crf 23 -c:a aac -b:a 128k '
        f'"{output_video}"'
    )
    print("[Info] FFmpeg command:", cmd)
    os.system(cmd)


def process_one_video_framebased(
    input_video,
    json_path,
    output_dir,
    keypoint_type='body',
    max_points=17,
    joint_index=0,
    clip_seconds=6.0,
    wavelet='morl',
    max_scale=128,
    min_spike_width=3,
    shift_margin=10
):
    """
    Process a single video:
    1) If video <= clip_seconds, copy directly
    2) Otherwise CWT + frame sliding window to find the best interval
    3) If interval is close to beginning/end => clip 6s from beginning or end
    4) Use save_clipped_video_ffmpeg_reencode() for output, re-encoding
    """
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    out_mp4 = os.path.join(output_dir, base_name + ".mp4")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {input_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = total_frames / fps if fps > 0 else 0.0
    if duration_sec <= clip_seconds:
        # Video too short, copy directly
        print(f"[Info] Video({input_video}) duration {duration_sec:.2f}s <= {clip_seconds}, copying directly.")
        shutil.copy2(input_video, out_mp4)
        return

    # Read JSON => velocity
    _, all_x, all_y = read_keypoints_from_json(
        json_path=json_path,
        keypoint_type=keypoint_type,
        max_points=max_points
    )
    if len(all_x[0]) < 2:
        print(f"[Warning] JSON {json_path} insufficient data, skipping.")
        return

    speed = compute_velocity(all_x, all_y, fps, joint_index)
    n_samples = len(speed)
    if n_samples < 2:
        print(f"[Warning] Velocity sequence too short, skipping: {input_video}")
        return

    # Wavelet analysis => full frequency energy
    energy_all, n_samples = cwt_allfreq_energy(speed, fps, wavelet, max_scale)
    if n_samples < 1:
        print("[Warning] Wavelet energy empty, skipping.")
        return

    # Remove narrow peaks
    energy_filter = remove_short_spikes(energy_all, min_width_frames=min_spike_width)

    # 6-second frame window
    window_frames = int(round(clip_seconds * fps))
    if window_frames >= n_samples:
        print(f"[Warning] clip_seconds={clip_seconds}, window>=n_samples, skipping.")
        return

    # Frame sliding window to find interval
    best_start_f, best_end_f = find_best_window_in_frames(energy_filter, window_frames)

    # If close to beginning or end => stick to edge
    if best_start_f < shift_margin:
        # Beginning
        best_start_f = 0
        best_end_f = window_frames - 1
    elif best_end_f > total_frames - shift_margin:
        # End
        best_end_f = total_frames - 1
        best_start_f = best_end_f - (window_frames - 1)

    # Clamp
    if best_end_f >= total_frames:
        best_end_f = total_frames - 1
        best_start_f = max(0, best_end_f - (window_frames - 1))

    # Re-encode and clip output
    save_clipped_video_ffmpeg_reencode(
        input_video=input_video,
        output_video=out_mp4,
        start_frame=best_start_f,
        end_frame=best_end_f,
        fps=fps
    )
    print(f"[Info] {input_video} => {out_mp4}")
    print(f"[Info] Frame interval [{best_start_f}, {best_end_f}] ~ {clip_seconds} seconds (re-encoded)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True, help='folder with .mp4')
    parser.add_argument('--json_dir', required=True, help='folder with .json (same name)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--keypoint_type', default='body')
    parser.add_argument('--max_points', type=int, default=17)
    parser.add_argument('--joint_index', type=int, default=0)
    parser.add_argument('--clip_seconds', type=float, default=6.0)
    parser.add_argument('--wavelet', default='morl')
    parser.add_argument('--max_scale', type=int, default=128)
    parser.add_argument('--min_spike_width', type=int, default=3,
                      help='minimum peak width in frames')
    parser.add_argument('--shift_margin', type=int, default=10,
                      help='how many frames near edge to just clip from edge')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_list = sorted(glob.glob(os.path.join(args.video_dir, '*.mp4')))
    if not video_list:
        print(f"[Error] No mp4 found in {args.video_dir}.")
        return

    for vid_path in video_list:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        json_path = os.path.join(args.json_dir, base_name + '.json')
        if not os.path.exists(json_path):
            print(f"[Warning] JSON doesn't exist: {json_path}, skipping.")
            continue

        process_one_video_framebased(
            input_video=vid_path,
            json_path=json_path,
            output_dir=args.output_dir,
            keypoint_type=args.keypoint_type,
            max_points=args.max_points,
            joint_index=args.joint_index,
            clip_seconds=args.clip_seconds,
            wavelet=args.wavelet,
            max_scale=args.max_scale,
            min_spike_width=args.min_spike_width,
            shift_margin=args.shift_margin
        )

    print("[Info] All videos processed.")


if __name__ == '__main__':
    main()

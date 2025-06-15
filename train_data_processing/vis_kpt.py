import os
import cv2
import json
import math
import numpy as np
import argparse
import matplotlib
from pathlib import Path

eps = 0.01

def draw_bodypose(canvas, candidate, subset):
    """
    Draw body skeletons including keypoints and bone connections.

    Parameters:
    - canvas: image canvas to draw on (H, W, C)
    - candidate: list of keypoints, each as [x_pixel, y_pixel, score]
    - subset: list of keypoint indices per person
    """
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = max(2, min(4, int(min(W, H) / 300)))
    point_radius = max(2, min(4, int(min(W, H) / 300)))

    # COCO 17-point skeleton definition (1-based indexing)
    limbSeq = [
        [1, 2], [1, 3], [2, 4], [3, 5],
        [6, 8], [7, 9],
        [8, 10], [9, 11],
        [12, 14], [13, 15],
        [14, 16], [15, 17]
    ]

    # Use the first 14 colors
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255]
    ]

    # Draw bone lines
    for i, limb in enumerate(limbSeq):
        for n in range(len(subset)):
            index = subset[n][np.array(limb) - 1]
            if -1 in index:
                continue
            X = candidate[index.astype(int), 0]
            Y = candidate[index.astype(int), 1]

            mX = np.mean(X)
            mY = np.mean(Y)
            length = math.sqrt((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2)
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))

            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)),
                (int(length / 2), stickwidth),
                int(angle),
                0,
                360,
                1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])

    # Calculate midpoint between shoulders
    left_shoulder = candidate[5]
    right_shoulder = candidate[6]
    mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    def draw_line(p1, p2, color, width=stickwidth):
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.line(canvas, p1, p2, color, thickness=width, lineType=cv2.LINE_AA)

    # Draw center connections with distinctive colors
    mid_l_color = (0, 255, 0)
    mid_r_color = (255, 0, 85)
    mid_t_color = (255, 170, 0)
    mid_d_color = (255, 0, 255)
    mid_c_color = (0, 0, 255)

    draw_line(left_shoulder, [mid_shoulder_x, mid_shoulder_y], mid_l_color, stickwidth)
    draw_line(right_shoulder, [mid_shoulder_x, mid_shoulder_y], mid_r_color, stickwidth)
    nose = candidate[0]
    draw_line([mid_shoulder_x, mid_shoulder_y], nose, mid_t_color, stickwidth)
    left_hip = candidate[11]
    draw_line([mid_shoulder_x, mid_shoulder_y], left_hip, mid_d_color, stickwidth)
    right_hip = candidate[12]
    draw_line([mid_shoulder_x, mid_shoulder_y], right_hip, mid_c_color, stickwidth)

    # Draw keypoints
    for i in range(17):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y, s = candidate[index]
            if s > 0:
                x = int(x)
                y = int(y)
                cv2.circle(canvas, (x, y), point_radius, colors[i % len(colors)], thickness=-1)

    # Darken background to highlight keypoints
    canvas = (canvas * 0.6).astype(np.uint8)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    from matplotlib import colors
    H, W, C = canvas.shape

    line_thickness = max(1, min(2, int(min(W, H) / 500)))
    point_radius = max(2, min(4, int(min(W, H) / 400)))

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)
        num_points = len(peaks)

        for ie, e in enumerate(edges):
            if e[0] < num_points and e[1] < num_points:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)
                if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                    cv2.line(
                        canvas,
                        (x1, y1),
                        (x2, y2),
                        colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
                        thickness=line_thickness,
                    )

        for i, keypoint in enumerate(peaks):
            if i < num_points:
                x, y = keypoint
                x = int(x * W)
                y = int(y * H)
                if x > eps and y > eps:
                    cv2.circle(canvas, (x, y), point_radius, (0, 0, 255), thickness=-1)

    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    point_radius = max(1, min(3, int(min(W, H) / 500)))
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), point_radius, (255, 255, 255), thickness=-1)
    return canvas


def process_video(video_path, json_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, video_name + '.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError as e:
        print(f"Failed to parse JSON file {json_path}: {e}")
        return

    annotations = data['annotations']
    ann_dict = {}

    for ann in annotations:
        fname = ann['file_name']
        base = os.path.basename(fname)
        frame_str = os.path.splitext(base)[0]
        frame_id = int(frame_str)
        ann_dict[frame_id] = ann

    for frame_id in range(1, frame_count + 1):
        ret, frame = cap.read()
        if not ret:
            break

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if frame_id in ann_dict:
            ann = ann_dict[frame_id]

            body_kpts = np.array(ann.get('body_kpts', []))
            foot_kpts = np.array(ann.get('foot_kpts', []))
            lefthand_kpts = np.array(ann.get('lefthand_kpts', []))
            righthand_kpts = np.array(ann.get('righthand_kpts', []))
            face_kpts = np.array(ann.get('face_kpts', []))

            if len(body_kpts) >= 17:
                body_kpts_17 = body_kpts[:17]
                candidate = [[x, y, s] for x, y, s in body_kpts_17]
                candidate = np.array(candidate)
                subset = [list(range(len(candidate)))]
                canvas = draw_bodypose(canvas, candidate, subset)

            if len(lefthand_kpts) > 0:
                lefthand_peaks = [[x / width, y / height] for x, y, s in lefthand_kpts if s > 0]
                if len(lefthand_peaks) > 0:
                    canvas = draw_handpose(canvas, [lefthand_peaks])

            if len(righthand_kpts) > 0:
                righthand_peaks = [[x / width, y / height] for x, y, s in righthand_kpts if s > 0]
                if len(righthand_peaks) > 0:
                    canvas = draw_handpose(canvas, [righthand_peaks])

            if len(face_kpts) > 0:
                face_landmarks = [[x / width, y / height] for x, y, s in face_kpts if s > 0]
                if len(face_landmarks) > 0:
                    canvas = draw_facepose(canvas, [face_landmarks])

        out_writer.write(canvas)

    cap.release()
    out_writer.release()
    print(f"Output saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output videos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_paths = [p for p in os.listdir(args.video_dir) if p.endswith('.mp4')]
    for video_file in video_paths:
        video_path = os.path.join(args.video_dir, video_file)
        json_file = os.path.splitext(video_file)[0] + '.json'
        json_path = os.path.join(args.json_dir, json_file)

        if os.path.exists(json_path):
            process_video(video_path, json_path, args.output_dir)
        else:
            print(f"Skipping {video_file}: No corresponding JSON file found.")

if __name__ == '__main__':
    main()

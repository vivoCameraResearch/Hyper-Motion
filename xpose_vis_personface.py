import os
import cv2
import json
import math
import numpy as np
import argparse
from pathlib import Path

eps = 0.01

def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    stickwidth = max(2, min(4, int(min(W, H) / 300)))
    point_radius = max(2, min(4, int(min(W, H) / 300)))

    limbSeq = [
        [1, 2], [1, 3], [2, 4], [3, 5],
        [6, 8], [7, 9],
        [8, 10], [9, 11],
        [12, 14], [13, 15],
        [14, 16], [15, 17]
    ]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255]
    ]

    candidate = np.array(candidate)
    subset = np.array(subset)

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
                0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])

    # 额外连接肩膀中点
    left_shoulder = candidate[5]
    right_shoulder = candidate[6]
    mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    def draw_line(p1, p2, color, width=stickwidth):
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        cv2.line(canvas, p1, p2, color, thickness=width, lineType=cv2.LINE_AA)

    draw_line(left_shoulder, [mid_shoulder_x, mid_shoulder_y], (0, 255, 0))
    draw_line(right_shoulder, [mid_shoulder_x, mid_shoulder_y], (255, 0, 85))
    nose = candidate[0]
    draw_line([mid_shoulder_x, mid_shoulder_y], nose, (255, 170, 0))
    left_hip = candidate[11]
    right_hip = candidate[12]
    draw_line([mid_shoulder_x, mid_shoulder_y], left_hip, (255, 0, 255))
    draw_line([mid_shoulder_x, mid_shoulder_y], right_hip, (0, 0, 255))

    for i in range(17):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y, s = candidate[index]
            if s > 0:
                cv2.circle(canvas, (int(x), int(y)), point_radius, colors[i % len(colors)], thickness=-1)

    return canvas

def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    point_radius = max(1, min(3, int(min(W, H) / 500)))

    for lmks in all_lmks:
        lmks = np.array(lmks)
        for x, y in lmks:
            if x > eps and y > eps:
                cv2.circle(canvas, (int(x * W), int(y * H)), point_radius, (255, 255, 255), thickness=-1)

    return canvas

def process_video(video_path, body_json_path, face_json_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, video_name + '.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    with open(body_json_path, 'r') as f:
        body_data = json.load(f)
    with open(face_json_path, 'r') as f:
        face_data = json.load(f)

    body_ann = {ann['frame_id']: ann.get('instances', []) for ann in body_data}
    face_ann = {ann['frame_id']: ann.get('instances', []) for ann in face_data}

    for frame_id in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Body
        body_instances = body_ann.get(frame_id, [])
        for instance in body_instances:
            keypoints = instance.get('keypoints', [])
            if len(keypoints) >= 17:
                candidate = [[x * width, y * height, 1.0] for x, y in keypoints[:17]]
                candidate = np.array(candidate)
                subset = [list(range(len(candidate)))]
                canvas = draw_bodypose(canvas, candidate, subset)

        # Face
        face_instances = face_ann.get(frame_id, [])
        all_face_lmks = []
        for instance in face_instances:
            keypoints = instance.get('keypoints', [])
            if len(keypoints) > 0:
                face_lmks = [[x, y] for x, y in keypoints]
                all_face_lmks.append(face_lmks)
        canvas = draw_facepose(canvas, all_face_lmks)

        canvas = (canvas * 0.6).astype(np.uint8)
        out_writer.write(canvas)

    cap.release()
    out_writer.release()
    print(f"Output saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--body_json_dir', type=str, required=True)
    parser.add_argument('--face_json_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_files = [p for p in os.listdir(args.video_dir) if p.endswith('.mp4')]

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(args.video_dir, video_file)
        body_json = os.path.join(args.body_json_dir, video_name + '_person.json')
        face_json = os.path.join(args.face_json_dir, video_name + '_face.json')

        if all(os.path.exists(p) for p in [body_json, face_json]):
            process_video(video_path, body_json, face_json, args.output_dir)
        else:
            print(f"Skipping {video_file}: missing JSON files.")

if __name__ == '__main__':
    main()

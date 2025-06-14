import os
import shutil
import json
import argparse

def move_videos_by_json_ids(json_path, source_root, target_dir, file_ext=".mp4"):
    # load JSON
    with open(json_path, 'r') as f:
        video_dict = json.load(f)

    video_ids = set(video_dict.keys())
    print(f" Loaded {len(video_ids)} video IDs from {json_path}")

    os.makedirs(target_dir, exist_ok=True)

    moved_count = 0

    for root, _, files in os.walk(source_root):
        for file in files:
            filename_no_ext, ext = os.path.splitext(file)
            if ext.lower() == file_ext and filename_no_ext in video_ids:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.move(src_path, dst_path)
                moved_count += 1
                print(f"Moved: {file}")

    print(f"\nDone! Moved {moved_count} files to {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Move video files by ID list in JSON.")
    parser.add_argument("--json", required=True, help="Path to JSON file with video_id -> text.")
    parser.add_argument("--source", required=True, help="Root directory to search for video files.")
    parser.add_argument("--target", required=True, help="Directory to move matched video files to.")
    parser.add_argument("--ext", default=".mp4", help="Video file extension to match (default: .mp4)")

    args = parser.parse_args()

    move_videos_by_json_ids(
        json_path=args.json,
        source_root=args.source,
        target_dir=args.target,
        file_ext=args.ext
    )

if __name__ == "__main__":
    main()

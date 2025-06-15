import os
import shutil
import json
import argparse

def move_matching_files(json_path, source_root, target_dir, video_ext=".mp4", extra_exts=[]):
    #
    with open(json_path, 'r') as f:
        video_dict = json.load(f)

    video_ids = set(video_dict.keys())
    print(f"Loaded {len(video_ids)} video IDs from {json_path}")
    os.makedirs(target_dir, exist_ok=True)

    moved_count = 0

    #
    for root, _, files in os.walk(source_root):
        for file in files:
            filename_no_ext, ext = os.path.splitext(file)
            ext = ext.lower()

            
            if filename_no_ext in video_ids:
                if ext == video_ext or ext in extra_exts:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                    print(f"Moved: {file}")

    print(f"Done! Total moved: {moved_count} files to {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Move files by video_id list in JSON.")
    parser.add_argument("--json", required=True, help="Path to video_text_dict.json (contains video IDs).")
    parser.add_argument("--source", required=True, help="Root directory to search for files.")
    parser.add_argument("--target", required=True, help="Target directory to move matched files.")
    parser.add_argument("--video_ext", default=".mp4", help="Main video file extension (e.g. .mp4)")
    parser.add_argument("--extra_exts", nargs="*", default=[], help="Extra file extensions to move (e.g. .json)")

    args = parser.parse_args()

    move_matching_files(
        json_path=args.json,
        source_root=args.source,
        target_dir=args.target,
        video_ext=args.video_ext,
        extra_exts=args.extra_exts
    )

if __name__ == "__main__":
    main()


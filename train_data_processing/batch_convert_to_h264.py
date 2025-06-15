import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def convert_video(input_path, output_path, codec='libx264', preset='fast', crf=23):
    """
    Convert a video to H.264 encoding using FFmpeg.

    Parameters:
    - input_path (str): Path to the input video file.
    - output_path (str): Path to the output video file.
    - codec (str): Codec to use. Default is 'libx264'.
    - preset (str): Encoding preset. Default is 'fast'. Options include 'ultrafast', 'superfast', 'veryfast', 
                    'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.
    - crf (int): Constant Rate Factor for quality (range: 0–51). Lower means better quality and larger file.
    """
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', codec,
        '-preset', preset,
        '-crf', str(crf),
        '-c:a', 'copy',  # Keep original audio stream
        '-y',  # Overwrite output file if it exists
        str(output_path)
    ]

    try:
        # Execute FFmpeg command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error: Failed to convert {input_path.name}.\n{result.stderr}")
            return False
        else:
            print(f"Success: {input_path.name} converted to {output_path.name}")
            return True
    except Exception as e:
        print(f"Exception: Error occurred while converting {input_path.name}: {e}")
        return False

def batch_convert(input_dir, output_dir, codec='libx264', preset='fast', crf=23, extensions=['.mp4', '.avi', '.mov', '.mkv']):
    """
    Batch convert all video files in a directory to H.264 encoding.

    Parameters:
    - input_dir (str): Path to the folder containing input videos.
    - output_dir (str): Path to the folder to save converted videos.
    - codec (str): Codec to use. Default is 'libx264'.
    - preset (str): Encoding preset. Default is 'fast'.
    - crf (int): Constant Rate Factor for quality. Default is 23.
    - extensions (list): List of video file extensions to process. Default is ['.mp4', '.avi', '.mov', '.mkv'].
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all matching video files
    video_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    if not video_files:
        print(f"Warning: No matching video files found in {input_dir}.")
        return

    print(f"Found {len(video_files)} video files. Starting conversion...")

    # Use ThreadPoolExecutor for parallel conversion
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all conversion tasks
        future_to_file = {
            executor.submit(convert_video, file, output_dir / file.name, codec, preset, crf): file for file in video_files
        }

        # Wait for all tasks to complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                success = future.result()
                if not success:
                    print(f"Conversion failed: {file.name}")
            except Exception as exc:
                print(f"Exception occurred during conversion of {file.name}: {exc}")

    print("Batch conversion completed.")

def main():
    parser = argparse.ArgumentParser(description='Batch convert videos from MP4V to H.264 (libx264).')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing videos to convert.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted videos.')
    parser.add_argument('--codec', type=str, default='libx264', help='Video codec to use. Default is "libx264".')
    parser.add_argument('--preset', type=str, default='fast', help='Encoding preset. Default is "fast".')
    parser.add_argument('--crf', type=int, default=23, help='Constant Rate Factor (0-51). Default is 23.')
    parser.add_argument('--extensions', type=str, nargs='+', default=['.mp4', '.avi', '.mov', '.mkv'],
                        help='List of video file extensions to process. Default is [".mp4", ".avi", ".mov", ".mkv"].')

    args = parser.parse_args()

    # Check if FFmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in system PATH. Please install FFmpeg and ensure it is accessible via command line.")
        return

    # Run batch conversion
    batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
        extensions=args.extensions
    )

if __name__ == "__main__":
    main()

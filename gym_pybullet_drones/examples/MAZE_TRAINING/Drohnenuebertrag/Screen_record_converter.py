#!/usr/bin/env python3
"""
WebM to MP4 Converter for Ubuntu Screen Recordings

This script converts a WebM video file to MP4 format using the FFmpeg library.
Requires FFmpeg to be installed on the system and the python-ffmpeg-video-streaming package.
"""

import argparse
import os
import subprocess
from datetime import datetime


def check_ffmpeg_installed():
    """Check if FFmpeg is installed on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def convert_webm_to_mp4(input_file, output_file=None):
    """
    Convert a WebM file to MP4 format using FFmpeg.

    Args:
        input_file (str): Path to the input WebM file
        output_file (str, optional): Path for the output MP4 file.
                                    If None, will use the same name with .mp4 extension.

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False

    if not input_file.lower().endswith(".webm"):
        print(f"Warning: Input file '{input_file}' doesn't have a .webm extension.")

    # Create output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.mp4"

    print(f"Converting '{input_file}' to '{output_file}'...")
    start_time = datetime.now()

    try:
        # Run FFmpeg command to convert the file
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-c:v",
            "libx264",  # Video codec
            "-preset",
            "medium",  # Encoding speed/compression trade-off
            "-crf",
            "23",  # Constant Rate Factor (quality: lower is better)
            "-c:a",
            "aac",  # Audio codec
            "-b:a",
            "128k",  # Audio bitrate
            "-y",  # Overwrite output file if it exists
            output_file,
        ]

        # Execute the command
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if os.path.exists(output_file):
            print(f"Conversion completed successfully in {duration:.2f} seconds.")
            print(f"Output file: {output_file}")
            return True
        else:
            print("Error: Conversion seemed to complete but output file was not created.")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print(f"FFmpeg error output: {e.stderr}")
        return False


def main():
    """Main function to parse arguments and initiate conversion."""
    parser = argparse.ArgumentParser(description="Convert WebM files to MP4 format.")
    parser.add_argument("input_file", help="Path to the input WebM file")
    parser.add_argument("-o", "--output", help="Path for the output MP4 file (optional)")

    args = parser.parse_args()

    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        print("Error: FFmpeg is not installed. Please install it using:")
        print("sudo apt update && sudo apt install ffmpeg")
        return

    # Perform the conversion
    convert_webm_to_mp4(args.input_file, args.output)


if __name__ == "__main__":
    main()

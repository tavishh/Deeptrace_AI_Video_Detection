"""
reencode.py - Normalize all clips to common format via ffmpeg

Owner: Tavish
Week:  1

Re-encodes every clip to H.264 CRF 23 at 720p BEFORE any other processing.
This eliminates compression-based leakage between real (Kinetics-400, YouTube-
compressed) and AI-generated (often high-bitrate demo exports) sources.

Run this FIRST on every clip before frame extraction or preprocessing.

Usage:
    python src/data_pipeline/reencode.py --input data/raw --output data/reencoded
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

FFMPEG_PARAMS = {
    "codec":   "libx264",
    "crf":     "23",
    "width":   "1280",
    "height":  "720",
    "audio":   "none",
}


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def reencode_clip(input_path: Path, output_path: Path) -> bool:
    """
    Re-encode a single clip to H.264 / CRF 23 / 720p, stripping audio.

    Args:
        input_path:  Path to original clip
        output_path: Path to write re-encoded clip (.mp4)

    Returns:
        True if re-encoding succeeded, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already processed
    if output_path.exists():
        return True

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", f"scale={FFMPEG_PARAMS['width']}:{FFMPEG_PARAMS['height']}:"
               f"force_original_aspect_ratio=decrease,"
               f"pad={FFMPEG_PARAMS['width']}:{FFMPEG_PARAMS['height']}:"
               f"(ow-iw)/2:(oh-ih)/2",
        "-c:v", FFMPEG_PARAMS["codec"],
        "-crf", FFMPEG_PARAMS["crf"],
        "-preset", "fast",
        "-an",                   # strip audio
        "-movflags", "+faststart",
        "-y",                    # overwrite without prompting
        "-loglevel", "error",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"\n  [ffmpeg error] {input_path.name}: {result.stderr.strip()}")
            # Remove partial output if it exists
            if output_path.exists():
                output_path.unlink()
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"\n  [timeout] {input_path.name}")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        print(f"\n  [exception] {input_path.name}: {e}")
        return False


def _reencode_worker(args: tuple) -> tuple:
    """Worker function for multiprocessing. Returns (input_path, success)."""
    input_path, output_path = args
    success = reencode_clip(input_path, output_path)
    return str(input_path), success


def reencode_directory(input_dir: Path, output_dir: Path,
                       workers: int = 4) -> None:
    """
    Re-encode all video files in input_dir recursively.
    Preserves directory structure in output_dir.

    Args:
        input_dir:  Root directory of raw clips
        output_dir: Root directory for re-encoded output
        workers:    Number of parallel ffmpeg processes
    """
    # Collect all video files
    clips = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not clips:
        print(f"No video files found in {input_dir}")
        return

    print(f"Re-encoding {len(clips)} clips "
          f"({workers} workers) -> {output_dir}")

    # Build (input, output) pairs preserving subdirectory structure
    jobs = []
    for clip in clips:
        relative = clip.relative_to(input_dir)
        output_path = (output_dir / relative).with_suffix(".mp4")
        jobs.append((clip, output_path))

    success_count, fail_count = 0, 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_reencode_worker, job): job for job in jobs}
        with tqdm(total=len(jobs), unit="clip", desc="Re-encoding") as pbar:
            for future in as_completed(futures):
                input_path_str, ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"  [failed] {Path(input_path_str).name}")
                pbar.update(1)

    print(f"\nDone. {success_count} succeeded, {fail_count} failed.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-encode clips to H.264/CRF23/720p."
    )
    parser.add_argument(
        "--input",  type=str, required=True,
        help="Root directory of raw clips."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Root directory for re-encoded output."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel ffmpeg processes (default: 4)."
    )
    args = parser.parse_args()
    reencode_directory(Path(args.input), Path(args.output), args.workers)
"""
extract_frames.py - Extract frames from clips at 1 FPS using OpenCV

Owner: Tavish
Week:  1

Samples one frame per second from each clip. For a 5-second clip this
produces 5 frames. Run AFTER reencode.py.

Output structure:
    data/frames/{clip_id}/frame_0001.jpg
    data/frames/{clip_id}/frame_0002.jpg
    ...

Usage:
    python src/data_pipeline/extract_frames.py \
        --input data/reencoded --output data/frames
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SAMPLE_RATE_FPS  = 1    # frames per second to extract
FRAME_SIZE       = 224  # resize longest side to this (square crop for CLIP)
JPEG_QUALITY     = 95   # JPEG compression quality


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def extract_frames(clip_path: Path, output_dir: Path,
                   fps: int = SAMPLE_RATE_FPS) -> list[Path]:
    """
    Extract frames from a single clip at the given FPS.

    Frames are resized and center-cropped to FRAME_SIZE x FRAME_SIZE
    for direct CLIP compatibility.

    Args:
        clip_path:  Path to re-encoded clip
        output_dir: Directory to save extracted frames (one subdir per clip)
        fps:        Sampling rate in frames per second

    Returns:
        List of paths to saved frame images
    """
    clip_id    = clip_path.stem
    frame_dir  = output_dir / clip_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  [error] Cannot open {clip_path.name}")
        return []

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        print(f"  [warning] Invalid FPS for {clip_path.name}, skipping.")
        cap.release()
        return []

    # Sample every Nth frame to achieve target FPS
    frame_interval = max(1, round(video_fps / fps))

    saved_paths   = []
    frame_idx     = 0
    saved_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize and center-crop to FRAME_SIZE x FRAME_SIZE
            processed = _resize_and_crop(frame, FRAME_SIZE)

            frame_name   = f"frame_{saved_idx:04d}.jpg"
            frame_path   = frame_dir / frame_name
            cv2.imwrite(
                str(frame_path), processed,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            saved_paths.append(frame_path)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return saved_paths


def _resize_and_crop(frame, size: int):
    """
    Resize the shorter side of a frame to `size`, then center-crop to size x size.
    Matches CLIP ViT-B/32 preprocessing expectations.
    """
    h, w = frame.shape[:2]

    # Scale so shortest side == size
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Center crop to size x size
    top  = (new_h - size) // 2
    left = (new_w - size) // 2
    cropped = resized[top:top + size, left:left + size]

    return cropped


def extract_all(input_dir: Path, output_dir: Path,
                fps: int = SAMPLE_RATE_FPS) -> None:
    """
    Extract frames from all clips in input_dir recursively.

    Args:
        input_dir:  Root directory of re-encoded clips
        output_dir: Root directory for extracted frames
        fps:        Sampling rate in frames per second
    """
    clips = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not clips:
        print(f"No video files found in {input_dir}")
        return

    print(f"Extracting frames from {len(clips)} clips at {fps} FPS -> {output_dir}")

    total_frames = 0
    for clip_path in tqdm(clips, unit="clip", desc="Extracting"):
        # Skip if already extracted
        clip_id   = clip_path.stem
        frame_dir = output_dir / clip_id
        if frame_dir.exists() and any(frame_dir.iterdir()):
            continue

        frames = extract_frames(clip_path, output_dir, fps)
        total_frames += len(frames)

    print(f"\nDone. {total_frames} total frames saved to {output_dir}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from re-encoded clips at 1 FPS."
    )
    parser.add_argument(
        "--input",  type=str, required=True,
        help="Root directory of re-encoded clips."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Root directory for extracted frames."
    )
    parser.add_argument(
        "--fps", type=int, default=SAMPLE_RATE_FPS,
        help=f"Frames per second to extract (default: {SAMPLE_RATE_FPS})."
    )
    args = parser.parse_args()
    extract_all(Path(args.input), Path(args.output), args.fps)
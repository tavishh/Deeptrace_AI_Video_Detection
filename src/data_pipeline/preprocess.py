"""
preprocess.py - Face detection, cropping, resizing, and CLIP normalization

Owner: Tavish
Week:  1

For clips with faces: detect face using RetinaFace, crop, resize to 224x224.
For scene-only clips: resize full frame to 224x224.
Apply CLIP preprocessing normalization to all frames.

Run AFTER extract_frames.py.

Usage:
    python src/data_pipeline/preprocess.py \
        --input data/frames --output data/processed \
        --manifest data/manifests/deeptrace_gv.csv
"""

import argparse
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# CLIP normalization constants (from OpenAI)
CLIP_MEAN  = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
CLIP_STD   = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGE_SIZE = 224

# Lazy-loaded RetinaFace model (initialized once on first use)
_retina_model = None


def _get_retina_model():
    """Initialize and return the RetinaFace model (loaded once)."""
    global _retina_model
    if _retina_model is None:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _retina_model = app
            print("  [RetinaFace] model loaded.")
        except Exception as e:
            print(f"  [RetinaFace] failed to load: {e}. "
                  f"Face clips will fall back to full-frame resize.")
            _retina_model = None
    return _retina_model


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def detect_and_crop_face(frame: np.ndarray) -> np.ndarray | None:
    """
    Detect the largest face in a frame using RetinaFace and return the crop.
    Adds a 20% padding around the bounding box before cropping.
    Returns None if no face is detected or model unavailable.

    Args:
        frame: BGR image as numpy array (H x W x 3)

    Returns:
        Cropped face region as numpy array, or None
    """
    model = _get_retina_model()
    if model is None:
        return None

    try:
        # insightface expects BGR
        faces = model.get(frame)
        if not faces:
            return None

        # Pick the largest face by bounding box area
        largest = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))

        x1, y1, x2, y2 = [int(v) for v in largest.bbox]
        h, w = frame.shape[:2]

        # Add 20% padding
        pad_x = int((x2 - x1) * 0.20)
        pad_y = int((y2 - y1) * 0.20)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    except Exception as e:
        print(f"  [face detect error] {e}")
        return None


def _center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize shortest side to size, then center-crop to size x size."""
    h, w = frame.shape[:2]
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    top  = (new_h - size) // 2
    left = (new_w - size) // 2
    return resized[top:top + size, left:left + size]


def _clip_normalize(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 frame to CLIP-normalized float32 tensor (3, H, W).
    Applies ImageNet-style normalization using CLIP's mean and std.
    """
    # BGR -> RGB, scale to [0, 1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Normalize
    normalized = (rgb - CLIP_MEAN) / CLIP_STD
    # HWC -> CHW
    return normalized.transpose(2, 0, 1)


def preprocess_frame(frame_path: Path, has_faces: bool) -> np.ndarray | None:
    """
    Load a frame, apply face detection (if applicable), resize to 224x224,
    and apply CLIP normalization.

    Args:
        frame_path: Path to extracted frame image
        has_faces:  Whether to attempt face detection

    Returns:
        Preprocessed frame as float32 numpy array (3, 224, 224), or None on error
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"  [error] Cannot read {frame_path.name}")
        return None

    if has_faces:
        crop = detect_and_crop_face(frame)
        region = crop if crop is not None else frame
    else:
        region = frame

    resized    = _center_crop_resize(region, IMAGE_SIZE)
    normalized = _clip_normalize(resized)
    return normalized.astype(np.float32)


def preprocess_all(input_dir: Path, output_dir: Path,
                   manifest_path: Path) -> None:
    """
    Preprocess all frames according to the manifest metadata.
    Saves each preprocessed frame as a .npy file alongside the original.

    Args:
        input_dir:     Root directory of extracted frames (one subdir per clip)
        output_dir:    Root directory for preprocessed .npy output
        manifest_path: Path to DeepTrace-GV CSV manifest
    """
    # Load manifest to get has_faces per clip
    has_faces_map: dict[str, bool] = {}
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id   = row.get("clip_id", "").strip()
                has_faces = row.get("has_faces", "").strip().lower()
                has_faces_map[clip_id] = has_faces in ("true", "1", "yes")
    else:
        print(f"  [warning] Manifest not found at {manifest_path}. "
              f"Defaulting to full-frame resize for all clips.")

    # Collect all frame images
    frame_paths = sorted(input_dir.rglob("*.jpg")) + \
                  sorted(input_dir.rglob("*.png"))

    if not frame_paths:
        print(f"No frame images found in {input_dir}")
        return

    print(f"Preprocessing {len(frame_paths)} frames -> {output_dir}")
    success, failed = 0, 0

    for frame_path in tqdm(frame_paths, unit="frame", desc="Preprocessing"):
        # clip_id is the parent directory name
        clip_id   = frame_path.parent.name
        has_faces = has_faces_map.get(clip_id, False)

        # Mirror directory structure in output
        relative   = frame_path.relative_to(input_dir)
        output_path = (output_dir / relative).with_suffix(".npy")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed
        if output_path.exists():
            success += 1
            continue

        tensor = preprocess_frame(frame_path, has_faces)
        if tensor is not None:
            np.save(str(output_path), tensor)
            success += 1
        else:
            failed += 1

    print(f"\nDone. {success} succeeded, {failed} failed.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess frames: face detection, resize, CLIP normalization."
    )
    parser.add_argument(
        "--input",    type=str, required=True,
        help="Root directory of extracted frames."
    )
    parser.add_argument(
        "--output",   type=str, required=True,
        help="Root directory for preprocessed .npy output."
    )
    parser.add_argument(
        "--manifest", type=str,
        default="data/manifests/deeptrace_gv.csv",
        help="Path to DeepTrace-GV manifest CSV."
    )
    args = parser.parse_args()
    preprocess_all(Path(args.input), Path(args.output), Path(args.manifest))
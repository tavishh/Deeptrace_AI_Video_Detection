"""
collect.py - AI-generated clip collection via yt-dlp

Owner: Tavish
Week:  1

Downloads clips from official generator channels and verified community sources.
Only clips with explicit generator attribution from the uploader or a visible
platform watermark are included.

Usage:
    python src/data_pipeline/collect.py --generator sora --output data/raw/sora
    python src/data_pipeline/collect.py --generator kling --output data/raw/kling
    python src/data_pipeline/collect.py --generator veo   --output data/raw/veo
"""

import argparse
import csv
import json
import subprocess
import uuid
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# SOURCE LISTS
# Add URLs here as you collect them. Only include clips with explicit
# generator attribution from the uploader or a visible platform watermark.
# ---------------------------------------------------------------------------

OFFICIAL_SOURCES = {
    "sora":  [
        # Example: "https://www.youtube.com/watch?v=XXXXXXX",
    ],
    "kling": [
        # Example: "https://www.youtube.com/watch?v=XXXXXXX",
    ],
    "veo":   [
        # Example: "https://www.youtube.com/watch?v=XXXXXXX",
    ],
}

COMMUNITY_SOURCES = {
    "sora":  [
        # Reddit/X/YouTube Shorts - only verified AI-generated posts
    ],
    "kling": [],
    "veo":   [],
}

# Staging manifest written during collection - one row per downloaded clip.
# Fields that require manual annotation (scene_type, motion_level, difficulty)
# are left blank and filled in during the annotation step.
STAGING_MANIFEST_FIELDS = [
    "clip_id", "label", "generator", "scene_type", "motion_level",
    "has_faces", "resolution", "duration_sec", "source_url",
    "collection_date", "split", "difficulty",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _probe(clip_path: Path) -> dict:
    """
    Use ffprobe to extract resolution and duration from a clip.
    Returns a dict with keys: resolution (e.g. '1280x720'), duration_sec (float).
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(clip_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        width, height, duration = None, None, None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                width = stream.get("width")
                height = stream.get("height")
                duration = float(stream.get("duration", 0))
                break
        if duration is None:
            duration = float(info.get("format", {}).get("duration", 0))
        resolution = f"{width}x{height}" if width and height else "unknown"
        return {"resolution": resolution, "duration_sec": round(duration, 2)}
    except Exception as e:
        print(f"  [probe error] {clip_path.name}: {e}")
        return {"resolution": "unknown", "duration_sec": 0.0}


def _append_manifest_row(manifest_path: Path, row: dict) -> None:
    """Append a single row to the staging manifest CSV."""
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STAGING_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def download_clip(url: str, output_dir: Path, generator: str,
                  manifest_path: Path) -> bool:
    """
    Download a single clip using yt-dlp and log metadata to the manifest.

    Args:
        url:           Source URL of the clip
        output_dir:    Directory to save the downloaded clip
        generator:     Generator label (sora | kling | veo)
        manifest_path: Path to the staging manifest CSV

    Returns:
        True if download succeeded, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id = f"{generator}_{uuid.uuid4().hex[:8]}"
    output_template = str(output_dir / f"{clip_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    print(f"  Downloading {url} -> {clip_id}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [yt-dlp error] {result.stderr.strip()}")
            return False

        # Find the downloaded file (yt-dlp determines the final extension)
        matches = list(output_dir.glob(f"{clip_id}.*"))
        if not matches:
            print(f"  [error] downloaded file not found for {clip_id}")
            return False
        clip_path = matches[0]

        # Probe for resolution and duration
        probe = _probe(clip_path)

        # Write staging manifest row
        row = {
            "clip_id":        clip_id,
            "label":          "fake",
            "generator":      generator,
            "scene_type":     "",          # manual annotation later
            "motion_level":   "",          # manual annotation later
            "has_faces":      "",          # RetinaFace pass later
            "resolution":     probe["resolution"],
            "duration_sec":   probe["duration_sec"],
            "source_url":     url,
            "collection_date": str(date.today()),
            "split":          "",          # assigned after full collection
            "difficulty":     "",          # team annotation later
        }
        _append_manifest_row(manifest_path, row)
        print(f"  [ok] {clip_id} | {probe['resolution']} | {probe['duration_sec']}s")
        return True

    except subprocess.TimeoutExpired:
        print(f"  [timeout] {url}")
        return False
    except Exception as e:
        print(f"  [exception] {url}: {e}")
        return False


def collect(generator: str, output_dir: Path, manifest_path: Path) -> None:
    """
    Download all clips for a given generator from official and community sources.

    Args:
        generator:     Generator label (sora | kling | veo)
        output_dir:    Root output directory for this generator
        manifest_path: Path to the staging manifest CSV
    """
    all_sources = OFFICIAL_SOURCES.get(generator, []) + \
                  COMMUNITY_SOURCES.get(generator, [])

    if not all_sources:
        print(f"[{generator}] No URLs configured yet. "
              f"Add URLs to OFFICIAL_SOURCES / COMMUNITY_SOURCES in collect.py.")
        return

    print(f"\n[{generator}] Downloading {len(all_sources)} clips -> {output_dir}")
    success, failed = 0, 0
    for url in all_sources:
        ok = download_clip(url, output_dir, generator, manifest_path)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n[{generator}] Done. {success} succeeded, {failed} failed.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AI-generated clips for DeepTrace-GV."
    )
    parser.add_argument(
        "--generator", choices=["sora", "kling", "veo"], required=True,
        help="Generator to collect clips for."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for raw clips (e.g. data/raw/sora)."
    )
    parser.add_argument(
        "--manifest", type=str, default="data/manifests/staging_manifest.csv",
        help="Path to staging manifest CSV (default: data/manifests/staging_manifest.csv)."
    )
    args = parser.parse_args()
    collect(args.generator, Path(args.output), Path(args.manifest))
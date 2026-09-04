"""
annotate_ai_clips.py - Manual annotation for AI-generated clips

Usage:
    python3 src/data_pipeline/annotate_ai_clips.py \
        --manifest data/manifests/staging_manifest.csv
"""

import argparse
import csv
import subprocess
from pathlib import Path

SCENE_TYPES   = ["face", "landscape", "object", "urban", "mixed"]
MOTION_LEVELS = ["static", "low", "high"]


def annotate(manifest_path: Path) -> None:
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fields = list(rows[0].keys())
    unannotated = [r for r in rows if r["scene_type"] == "" and r["generator"] != "real"]
    print(f"{len(unannotated)} AI clips need annotation\n")

    for i, row in enumerate(unannotated):
        clip_id   = row["clip_id"]
        generator = row["generator"]

        clip_path = Path(f"data/reencoded/{generator}/{clip_id}.mp4")
        if clip_path.exists():
            subprocess.Popen(["open", str(clip_path)])

        print(f"[{i+1}/{len(unannotated)}] {clip_id}")

        while True:
            print("  0=face  1=landscape  2=object  3=urban  4=mixed")
            try:
                scene_input = input("  scene_type: ").strip()
                row["scene_type"] = SCENE_TYPES[int(scene_input)]
                break
            except (ValueError, IndexError):
                print("  Invalid input, try again.")

        while True:
            print("  0=static  1=low  2=high")
            try:
                motion_input = input("  motion_level: ").strip()
                row["motion_level"] = MOTION_LEVELS[int(motion_input)]
                break
            except (ValueError, IndexError):
                print("  Invalid input, try again.")

        while True:
            faces_input = input("  has_faces (y/n): ").strip().lower()
            if faces_input in ("y", "n"):
                row["has_faces"] = "true" if faces_input == "y" else "false"
                break
            print("  Enter y or n.")

        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  Saved.\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str,
        default="data/manifests/staging_manifest.csv"
    )
    args = parser.parse_args()
    annotate(Path(args.manifest))

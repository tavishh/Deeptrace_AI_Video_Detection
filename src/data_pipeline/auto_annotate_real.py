"""
auto_annotate_real.py - Auto-annotate real clips based on Kinetics category names

Owner: Tavish
Week:  1

Real clips have descriptive names (real_surfing_water_xxx) so scene_type,
motion_level, and has_faces can be inferred without manual review.

Usage:
    python3 src/data_pipeline/auto_annotate_real.py \
        --manifest data/manifests/staging_manifest.csv
"""

import argparse
import csv
from pathlib import Path

# Mapping from Kinetics category -> (scene_type, motion_level, has_faces)
CATEGORY_ANNOTATIONS = {
    "driving_car":    ("urban",     "high",   "false"),
    "surfing_water":  ("landscape", "high",   "false"),
    "sailing":        ("landscape", "low",    "false"),
    "jogging":        ("mixed",     "high",   "true"),
    "walking_the_dog":("mixed",     "low",    "true"),
    "petting_cat":    ("object",    "static", "true"),
    "scuba_diving":   ("landscape", "high",   "true"),
    "paragliding":    ("landscape", "high",   "true"),
    "motorcycling":   ("urban",     "high",   "false"),
    "skateboarding":  ("urban",     "high",   "true"),
    "tai_chi":        ("mixed",     "low",    "true"),
    "flying_kite":    ("landscape", "low",    "true"),
    "feeding_birds":  ("landscape", "low",    "true"),
}


def infer_category(clip_id: str) -> str | None:
    """
    Extract Kinetics category from clip_id by matching known category prefixes.
    e.g. real_surfing_water_MM-HW0kdYy0 -> surfing_water
         real_driving_car_QkLN_QBF1hI   -> driving_car
         real_sailing__3U9TMb-jFI       -> sailing
    """
    if not clip_id.startswith("real_"):
        return None
    remainder = clip_id[len("real_"):]
    for category in CATEGORY_ANNOTATIONS:
        if remainder.startswith(category + "_") or remainder.startswith(category + "-"):
            return category
    return None


def auto_annotate(manifest_path: Path) -> None:
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fields = list(rows[0].keys())
    updated = 0
    skipped = 0

    for row in rows:
        if row["generator"] != "real":
            continue
        if row["scene_type"] != "":
            skipped += 1
            continue

        category = infer_category(row["clip_id"])
        if category not in CATEGORY_ANNOTATIONS:
            print(f"  [unknown category] {row['clip_id']} -> {category}")
            continue

        scene_type, motion_level, has_faces = CATEGORY_ANNOTATIONS[category]
        row["scene_type"]   = scene_type
        row["motion_level"] = motion_level
        row["has_faces"]    = has_faces
        updated += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Auto-annotated {updated} real clips.")
    print(f"Skipped {skipped} already-annotated clips.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=str,
        default="data/manifests/staging_manifest.csv"
    )
    args = parser.parse_args()
    auto_annotate(Path(args.manifest))
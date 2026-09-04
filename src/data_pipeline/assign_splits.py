"""
assign_splits.py - Assign 70/15/15 train/val/test splits stratified by generator

Owner: Tavish
Week:  1

Splits are stratified by generator so each split contains a proportional
representation of Sora, Veo, Kling, and Real clips.

Usage:
    python3 src/data_pipeline/assign_splits.py \
        --manifest data/manifests/staging_manifest.csv
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


def assign_splits(manifest_path: Path) -> None:
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fields = list(rows[0].keys())

    # Group by generator
    by_generator = defaultdict(list)
    for row in rows:
        by_generator[row["generator"]].append(row)

    random.seed(RANDOM_SEED)

    total_train, total_val, total_test = 0, 0, 0

    for generator, clips in by_generator.items():
        random.shuffle(clips)
        n = len(clips)
        n_train = round(n * TRAIN_RATIO)
        n_val   = round(n * VAL_RATIO)
        n_test  = n - n_train - n_val

        for i, clip in enumerate(clips):
            if i < n_train:
                clip["split"] = "train"
            elif i < n_train + n_val:
                clip["split"] = "val"
            else:
                clip["split"] = "test"

        total_train += n_train
        total_val   += n_val
        total_test  += n_test

        print(f"  {generator:8s}: {n} clips -> "
              f"train={n_train}, val={n_val}, test={n_test}")

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal: train={total_train}, val={total_val}, test={total_test}")
    print(f"Splits saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign 70/15/15 train/val/test splits stratified by generator."
    )
    parser.add_argument(
        "--manifest", type=str,
        default="data/manifests/staging_manifest.csv"
    )
    args = parser.parse_args()
    assign_splits(Path(args.manifest))
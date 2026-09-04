"""
select_difficulty_subset.py - Select a 60-clip random subset for difficulty annotation

Owner: Tavish
Phase: End of project (post Phase 3)

Selects a random 60-clip subset from DeepTrace-GV, stratified roughly
proportional to generator representation, for all four team members to
independently rate as easy/medium/hard. Produces one blank rating sheet
per team member.

Usage:
    python3 src/data_pipeline/select_difficulty_subset.py \
        --manifest data/manifests/staging_manifest.csv \
        --output-dir data/difficulty_annotation \
        --n 60
"""

import argparse
import csv
import random
from pathlib import Path

TEAM_MEMBERS = ["tavish", "zihao", "jiajun", "xijia"]
RANDOM_SEED  = 42


def select_subset(manifest_path: Path, n: int) -> list:
    """
    Select n clips from the manifest, stratified proportionally by generator.
    """
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_generator = {}
    for row in rows:
        by_generator.setdefault(row["generator"], []).append(row)

    random.seed(RANDOM_SEED)
    total = len(rows)
    selected = []

    for generator, clips in by_generator.items():
        proportion = len(clips) / total
        n_select = max(1, round(n * proportion))
        random.shuffle(clips)
        selected.extend(clips[:n_select])

    # Trim or pad to exactly n if rounding caused drift
    random.shuffle(selected)
    selected = selected[:n]

    return selected


def write_rating_sheets(selected: list, output_dir: Path) -> None:
    """
    Write one blank rating sheet per team member, plus a combined reference sheet.
    Each sheet has clip_id, generator, source_url, and a blank 'difficulty' column.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "generator", "source_url", "difficulty"]

    for member in TEAM_MEMBERS:
        sheet_path = output_dir / f"difficulty_ratings_{member}.csv"
        with open(sheet_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for clip in selected:
                writer.writerow({
                    "clip_id":    clip["clip_id"],
                    "generator":  clip["generator"],
                    "source_url": clip["source_url"],
                    "difficulty": "",  # easy | medium | hard - to be filled in
                })
        print(f"  Wrote {sheet_path} ({len(selected)} clips)")

    # Reference sheet listing the selected subset (for tracking/auditing)
    ref_path = output_dir / "difficulty_subset_reference.csv"
    with open(ref_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "generator", "label", "source_url"])
        writer.writeheader()
        for clip in selected:
            writer.writerow({
                "clip_id":    clip["clip_id"],
                "generator":  clip["generator"],
                "label":      clip["label"],
                "source_url": clip["source_url"],
            })
    print(f"\n  Reference sheet: {ref_path}")


def main(manifest_path: str, output_dir: str, n: int) -> None:
    manifest_path = Path(manifest_path)
    output_dir    = Path(output_dir)

    selected = select_subset(manifest_path, n)

    print(f"Selected {len(selected)} clips for difficulty annotation.\n")
    by_gen = {}
    for clip in selected:
        by_gen[clip["generator"]] = by_gen.get(clip["generator"], 0) + 1
    for gen, count in sorted(by_gen.items()):
        print(f"  {gen:8s}: {count} clips")

    print(f"\nWriting rating sheets to {output_dir}/ ...")
    write_rating_sheets(selected, output_dir)

    print(f"\nDone. Rubric reminder:")
    print(f"  Rate each clip as easy / medium / hard based on whether it")
    print(f"  would fool a naive human viewer with no AI-detection training.")
    print(f"  Rate independently - do not discuss with teammates before submitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a random clip subset for team difficulty annotation."
    )
    parser.add_argument("--manifest",   type=str,
                        default="data/manifests/staging_manifest.csv")
    parser.add_argument("--output-dir", type=str,
                        default="data/difficulty_annotation")
    parser.add_argument("--n",          type=int, default=60)
    args = parser.parse_args()
    main(args.manifest, args.output_dir, args.n)
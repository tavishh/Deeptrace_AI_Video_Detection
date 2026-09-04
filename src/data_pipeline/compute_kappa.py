"""
compute_kappa.py - Compute Cohen's kappa for inter-rater agreement on difficulty annotation

Owner: Tavish
Phase: End of project (post Phase 3)

Once all four team members have filled in their difficulty_ratings_<name>.csv
sheets, run this to compute pairwise and overall (Fleiss') agreement and merge
the final difficulty label into the main manifest via majority vote.

Usage:
    python3 src/data_pipeline/compute_kappa.py \
        --ratings-dir data/difficulty_annotation \
        --manifest data/manifests/staging_manifest.csv
"""

import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

TEAM_MEMBERS = ["tavish", "zihao", "jiajun", "xijia"]
DIFFICULTY_MAP = {"easy": 0, "medium": 1, "hard": 2}


def load_ratings(ratings_dir: Path) -> dict:
    """
    Load each team member's ratings.
    Returns dict: {member: {clip_id: difficulty}}
    """
    ratings = {}
    for member in TEAM_MEMBERS:
        path = ratings_dir / f"difficulty_ratings_{member}.csv"
        if not path.exists():
            print(f"  [warning] {path} not found - skipping {member}")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            member_ratings = {}
            for row in reader:
                diff = row["difficulty"].strip().lower()
                if diff not in DIFFICULTY_MAP:
                    print(f"  [warning] {member}: missing/invalid rating for {row['clip_id']}")
                    continue
                member_ratings[row["clip_id"]] = diff
            ratings[member] = member_ratings
    return ratings


def compute_pairwise_kappa(ratings: dict) -> dict:
    """Compute Cohen's kappa for every pair of raters on shared clips."""
    pairwise = {}
    members = list(ratings.keys())

    for m1, m2 in combinations(members, 2):
        shared_clips = set(ratings[m1].keys()) & set(ratings[m2].keys())
        if not shared_clips:
            continue
        y1 = [DIFFICULTY_MAP[ratings[m1][c]] for c in shared_clips]
        y2 = [DIFFICULTY_MAP[ratings[m2][c]] for c in shared_clips]
        kappa = cohen_kappa_score(y1, y2)
        pairwise[(m1, m2)] = (kappa, len(shared_clips))

    return pairwise


def majority_vote_difficulty(ratings: dict, clip_id: str) -> str:
    """Return the majority-vote difficulty label for a clip across raters."""
    votes = [ratings[m][clip_id] for m in ratings if clip_id in ratings[m]]
    if not votes:
        return ""
    return Counter(votes).most_common(1)[0][0]


def merge_into_manifest(ratings: dict, manifest_path: Path) -> None:
    """Merge majority-vote difficulty labels into the main manifest."""
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    all_clip_ids = set()
    for member_ratings in ratings.values():
        all_clip_ids.update(member_ratings.keys())

    updated = 0
    for row in rows:
        if row["clip_id"] in all_clip_ids:
            row["difficulty"] = majority_vote_difficulty(ratings, row["clip_id"])
            updated += 1

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMerged difficulty labels for {updated} clips into {manifest_path}")


def main(ratings_dir: str, manifest_path: str) -> None:
    ratings_dir   = Path(ratings_dir)
    manifest_path = Path(manifest_path)

    print("Loading ratings...")
    ratings = load_ratings(ratings_dir)

    n_raters = len(ratings)
    print(f"\nLoaded ratings from {n_raters} raters: {list(ratings.keys())}")

    if n_raters < 2:
        print("Need at least 2 raters to compute kappa. Aborting.")
        return

    print("\nPairwise Cohen's kappa:")
    pairwise = compute_pairwise_kappa(ratings)
    kappas = []
    for (m1, m2), (kappa, n_shared) in pairwise.items():
        print(f"  {m1} vs {m2}: kappa={kappa:.4f}  (n={n_shared})")
        kappas.append(kappa)

    if kappas:
        mean_kappa = sum(kappas) / len(kappas)
        print(f"\nMean pairwise kappa: {mean_kappa:.4f}")
        print(_interpret_kappa(mean_kappa))

    merge_into_manifest(ratings, manifest_path)


def _interpret_kappa(kappa: float) -> str:
    """Standard Landis & Koch interpretation."""
    if kappa < 0:
        return "Interpretation: poor agreement (worse than chance)"
    elif kappa < 0.20:
        return "Interpretation: slight agreement"
    elif kappa < 0.40:
        return "Interpretation: fair agreement"
    elif kappa < 0.60:
        return "Interpretation: moderate agreement"
    elif kappa < 0.80:
        return "Interpretation: substantial agreement"
    else:
        return "Interpretation: almost perfect agreement"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa from team difficulty ratings."
    )
    parser.add_argument("--ratings-dir", type=str,
                        default="data/difficulty_annotation")
    parser.add_argument("--manifest",    type=str,
                        default="data/manifests/staging_manifest.csv")
    args = parser.parse_args()
    main(args.ratings_dir, args.manifest)
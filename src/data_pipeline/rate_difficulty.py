"""
rate_difficulty.py - Interactive difficulty rating tool for team members

Owner: Tavish
Phase: End of project (post Phase 3)

Opens each assigned clip in your default video player one at a time and
prompts you to rate it as easy/medium/hard. Saves progress after every
clip, so you can stop and resume anytime.

Usage:
    python3 rate_difficulty.py --name zihao --clips-dir data/reencoded
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

DIFFICULTY_OPTIONS = ["easy", "medium", "hard"]


def rate(name: str, clips_dir: Path, ratings_dir: Path) -> None:
    sheet_path = ratings_dir / f"difficulty_ratings_{name}.csv"

    if not sheet_path.exists():
        print(f"Error: {sheet_path} not found.")
        print("Make sure you've downloaded your rating sheet from the team Drive folder.")
        sys.exit(1)

    with open(sheet_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fields = list(rows[0].keys())
    unrated = [r for r in rows if r["difficulty"].strip() == ""]
    total   = len(rows)

    if not unrated:
        print(f"All {total} clips already rated. Nothing to do.")
        return

    print(f"\n{name.title()}'s difficulty rating session")
    print(f"{len(unrated)} of {total} clips remaining.\n")
    print("Rubric: would this clip fool a naive viewer with no AI-detection training?")
    print("Rate independently - do not discuss with teammates before finishing.\n")

    for i, row in enumerate(unrated):
        clip_id   = row["clip_id"]
        generator = row["generator"]
        clip_path = clips_dir / generator / f"{clip_id}.mp4"

        if not clip_path.exists():
            print(f"[{i+1}/{len(unrated)}] {clip_id} - FILE NOT FOUND at {clip_path}, skipping.")
            continue

        # Open in default video player
        try:
            subprocess.Popen(["open", str(clip_path)])   # macOS
        except FileNotFoundError:
            try:
                subprocess.Popen(["xdg-open", str(clip_path)])  # Linux
            except FileNotFoundError:
                subprocess.Popen(["start", str(clip_path)], shell=True)  # Windows

        print(f"[{i+1}/{len(unrated)}] {clip_id} ({generator})")

        while True:
            answer = input("  Rate: (e)asy / (m)edium / (h)ard / (s)kip: ").strip().lower()
            mapping = {"e": "easy", "m": "medium", "h": "hard"}
            if answer == "s":
                print("  Skipped.\n")
                break
            if answer in mapping:
                row["difficulty"] = mapping[answer]
                print(f"  Saved: {mapping[answer]}\n")
                break
            print("  Invalid input. Enter e, m, h, or s.")

        # Save progress after every clip
        with open(sheet_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    remaining = sum(1 for r in rows if r["difficulty"].strip() == "")
    print(f"\nSession complete. {total - remaining}/{total} clips rated.")
    if remaining > 0:
        print(f"{remaining} clips skipped or remaining - re-run this script to continue.")
    print(f"\nWhen finished, upload {sheet_path.name} back to the shared Drive folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively rate assigned clips for difficulty."
    )
    parser.add_argument("--name", type=str, required=True,
                        choices=["tavish", "zihao", "jiajun", "xijia"],
                        help="Your name (matches your rating sheet filename)")
    parser.add_argument("--clips-dir", type=str, default="data/reencoded",
                        help="Path to the reencoded clips folder (downloaded from Drive)")
    parser.add_argument("--ratings-dir", type=str, default="data/difficulty_annotation",
                        help="Path to the folder containing your rating sheet")
    args = parser.parse_args()

    rate(args.name, Path(args.clips_dir), Path(args.ratings_dir))
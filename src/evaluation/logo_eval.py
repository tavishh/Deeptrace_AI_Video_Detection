"""
logo_eval.py - Leave-One-Generator-Out (LOGO) evaluation protocol

Owner: Jiajun
Week:  2 (baselines) / Week 4 (DeepTrace)

PRIMARY EXPERIMENT. Runs three evaluation rounds:
  Round 1: train Kling + Veo,  test Sora
  Round 2: train Sora + Veo,   test Kling
  Round 3: train Sora + Kling, test Veo

Each round loads a pre-trained LOGO checkpoint (trained with that generator
held out) and evaluates on ONLY the held-out generator's fake clips plus
all real test clips. This mirrors the corrected methodology from the
training notebook (Step 10b).

Reports mean LOGO AUC and standard deviation across rounds.

Usage:
    python src/evaluation/logo_eval.py --model deeptrace --held-out sora \
        --checkpoint models/checkpoints/deeptrace_joint_sora_best.pth
    python src/evaluation/logo_eval.py --model deeptrace --all-rounds \
        --checkpoint-dir models/checkpoints
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from clip_extractor import CLIPExtractor
from mlp_head import MLPHead, DeepTrace
from train import VideoClipDataset, CLIP_TRANSFORM

from metrics import compute_all_metrics, compute_logo_auc


GENERATORS = ["sora", "kling", "veo"]

LOGO_ROUNDS = [
    {"held_out": "sora",  "train_on": ["kling", "veo"]},
    {"held_out": "kling", "train_on": ["sora",  "veo"]},
    {"held_out": "veo",   "train_on": ["sora",  "kling"]},
]


def _sanitize(obj):
    """Convert numpy types and NaN to JSON-safe values (NaN -> null)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _load_logo_test_samples(manifest_path: str, deeptrace_root: str,
                            held_out: str) -> list:
    """
    Load the correct test set for a LOGO round: only the held-out
    generator's fake clips plus all real test clips.
    """
    samples = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            gen = row["generator"]
            if gen != held_out and gen != "real":
                continue
            clip_path = os.path.join(deeptrace_root, gen, f"{row['clip_id']}.mp4")
            if not os.path.exists(clip_path):
                continue
            label = 0 if row["label"] == "real" else 1
            samples.append((clip_path, label))
    return samples


def _load_model(checkpoint_path: str):
    """Load DeepTrace from a LOGO checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = CLIPExtractor().to(device)
    extractor.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    hidden_dim = saved_args.get("mlp_hidden_dim", 128)
    dropout = saved_args.get("mlp_dropout", 0.3)

    head = MLPHead(hidden_dim=hidden_dim, dropout=dropout).to(device)
    head.load_state_dict(ckpt["head_state"])

    model = DeepTrace(extractor, head).to(device)
    model.eval()
    return model, device


def _predict(model, loader, device) -> tuple:
    """Run inference and collect ground truth labels and predicted probabilities."""
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            logits = model.forward_frames(frames)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def run_logo_round(model_name: str, held_out: str, train_on: list,
                   checkpoint_path: str = None,
                   manifest_path: str = "data/manifests/staging_manifest.csv",
                   deeptrace_root: str = "data/reencoded",
                   batch_size: int = 16) -> dict:
    """
    Run one LOGO round: load a pre-trained checkpoint that excluded the
    held-out generator, evaluate on that generator's test clips + real clips.

    Args:
        model_name:      Model name (deeptrace)
        held_out:        Generator held out (sora | kling | veo)
        train_on:        Generators that were in the training set
        checkpoint_path: Path to the LOGO checkpoint for this round
        manifest_path:   Path to staging_manifest.csv
        deeptrace_root:  Root of reencoded clips
        batch_size:      Batch size for inference

    Returns:
        Dict with auc, accuracy, precision, recall, f1, confusion_matrix,
        held_out, n_fake, n_real
    """
    if not checkpoint_path:
        raise ValueError(
            f"checkpoint_path required. Expected the checkpoint trained "
            f"WITHOUT {held_out} clips (e.g. deeptrace_joint_{held_out}_best.pth)"
        )

    model, device = _load_model(checkpoint_path)
    samples = _load_logo_test_samples(manifest_path, deeptrace_root, held_out)

    if not samples:
        print(f"No test samples found for LOGO round (held_out={held_out})")
        return {"held_out": held_out, "trained_on": train_on,
                "n_fake": 0, "n_real": 0,
                "auc": float("nan"), "accuracy": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "confusion_matrix": None}

    n_fake = sum(1 for _, l in samples if l == 1)
    n_real = sum(1 for _, l in samples if l == 0)
    print(f"\nLOGO round: held_out={held_out}")
    print(f"  Trained on: {', '.join(train_on)} + real + Celeb-DF")
    print(f"  Test set: {n_fake} {held_out} fake + {n_real} real = {len(samples)} clips")

    loader = DataLoader(
        VideoClipDataset(samples, CLIP_TRANSFORM, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )

    y_true, y_scores = _predict(model, loader, device)
    metrics = compute_all_metrics(y_true, y_scores)

    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")

    return {
        "held_out": held_out,
        "trained_on": train_on,
        "n_fake": n_fake,
        "n_real": n_real,
        **metrics,
    }


def run_all_rounds(model_name: str,
                   checkpoint_dir: str = "models/checkpoints",
                   manifest_path: str = "data/manifests/staging_manifest.csv",
                   deeptrace_root: str = "data/reencoded",
                   output_dir: str = "data/results",
                   overwrite: bool = False) -> dict:
    """
    Run all three LOGO rounds and aggregate results.

    Args:
        model_name:     Model to evaluate
        checkpoint_dir: Directory containing the 3 LOGO checkpoints
        manifest_path:  Path to staging_manifest.csv
        deeptrace_root: Root of reencoded clips
        output_dir:     Where to save results JSON
        overwrite:      Allow replacing an existing logo_results.json
                        (the canonical results the guide says not to rerun)

    Returns:
        Dict with per-round results, mean AUC, and std
    """
    round_results = []
    round_aucs = []

    for r in LOGO_ROUNDS:
        ckpt_path = os.path.join(
            checkpoint_dir, f"deeptrace_joint_{r['held_out']}_best.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"[warning] Checkpoint not found: {ckpt_path}")
            continue

        result = run_logo_round(
            model_name, r["held_out"], r["train_on"],
            checkpoint_path=ckpt_path,
            manifest_path=manifest_path,
            deeptrace_root=deeptrace_root,
        )
        round_results.append(result)
        if not np.isnan(result["auc"]):
            round_aucs.append(result["auc"])

    if not round_aucs:
        print("No LOGO rounds completed successfully.")
        return {"rounds": [], "mean_auc": float("nan"), "std_auc": float("nan")}

    logo = compute_logo_auc(round_aucs)

    print(f"\n{'='*50}")
    print(f"LOGO Summary ({model_name})")
    print(f"{'='*50}")
    for r in round_results:
        print(f"  Held-out {r['held_out']:6s}: AUC={r['auc']:.4f}  "
              f"Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}")
    print(f"  Mean AUC: {logo['mean']:.4f} ± {logo['std']:.4f}")

    if len(round_aucs) < len(LOGO_ROUNDS):
        print(f"[warning] Only {len(round_aucs)}/{len(LOGO_ROUNDS)} rounds "
              "completed; mean/std are not comparable to the full protocol.")

    output = {
        "model": model_name,
        "rounds": round_results,
        "mean_auc": logo["mean"],
        "std_auc": logo["std"],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "logo_results.json")
    if os.path.exists(out_path) and not overwrite:
        out_path = os.path.join(output_dir, "logo_results_rerun.json")
        print(f"[warning] {output_dir}/logo_results.json already exists "
              "(canonical results - the guide says not to regenerate them). "
              f"Saving to {out_path} instead; pass --overwrite to replace it.")

    with open(out_path, "w") as f:
        json.dump(_sanitize(output), f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--held-out",   choices=GENERATORS, default=None)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (for single round)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="models/checkpoints",
                        help="Directory with LOGO checkpoints (for --all-rounds)")
    parser.add_argument("--manifest",   type=str,
                        default="data/manifests/staging_manifest.csv")
    parser.add_argument("--deeptrace-root", type=str, default="data/reencoded")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing logo_results.json")
    args = parser.parse_args()

    if args.all_rounds:
        run_all_rounds(args.model,
                       checkpoint_dir=args.checkpoint_dir,
                       manifest_path=args.manifest,
                       deeptrace_root=args.deeptrace_root,
                       overwrite=args.overwrite)
    elif args.held_out:
        run_logo_round(args.model, args.held_out,
                       [g for g in GENERATORS if g != args.held_out],
                       checkpoint_path=args.checkpoint,
                       manifest_path=args.manifest,
                       deeptrace_root=args.deeptrace_root)

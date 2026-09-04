"""
evaluate.py - Main evaluation runner (standard + per-generator + difficulty)

Owner: Jiajun
Week:  2 (baselines) / Week 4 (DeepTrace)

Runs Tests 1-2 and 4-7 from the evaluation plan:
  Test 1: In-distribution on Celeb-DF v2
  Test 2: Standard cross-generator gap on DeepTrace-GV
  Test 4: Per-generator breakdown (Sora / Kling / Veo)
  Test 5: Difficulty subset + failure analysis
  Test 6: Speed benchmark
  Test 7: Live demo validation (manual)

Usage:
    python src/evaluation/evaluate.py --model deeptrace --split test
    python src/evaluation/evaluate.py --model deeptrace --split test --difficulty hard
    python src/evaluation/evaluate.py --model deeptrace --cross-manipulation
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from clip_extractor import CLIPExtractor
from mlp_head import MLPHead, DeepTrace
from train import (
    VideoClipDataset,
    CLIP_TRANSFORM,
    load_celebdf_samples,
    load_deeptrace_samples,
)

from metrics import (
    EvalResult, compute_all_metrics, cross_manipulation_gap,
    compute_per_generator_auc, print_results_table,
)


MODELS = ["xception", "ojha_clip", "deeptrace"]

DEFAULT_MANIFEST = "data/manifests/staging_manifest.csv"
DEFAULT_DEEPTRACE_ROOT = "data/reencoded"
DEFAULT_CELEBDF_ROOT = "data/celebdf"
DEFAULT_CHECKPOINT_DIR = "models/checkpoints"


def load_model(model_name: str, checkpoint_path: str = None):
    """Load a model by name for evaluation."""
    if model_name != "deeptrace":
        raise ValueError(
            f"Model '{model_name}' is not supported for live evaluation. "
            "Baseline results are pre-computed in data/results/."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = CLIPExtractor().to(device)
    extractor.eval()

    hidden_dim = 128
    dropout = 0.3
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        hidden_dim = saved_args.get("mlp_hidden_dim", hidden_dim)
        dropout = saved_args.get("mlp_dropout", dropout)

    head = MLPHead(hidden_dim=hidden_dim, dropout=dropout).to(device)
    if checkpoint_path:
        head.load_state_dict(ckpt["head_state"])

    model = DeepTrace(extractor, head).to(device)
    model.eval()
    return model, device


def _load_deeptrace_test_samples(manifest_path: str, deeptrace_root: str,
                                 split: str = "test",
                                 difficulty: Optional[str] = None) -> tuple:
    """
    Load DeepTrace-GV samples with optional difficulty filtering.
    Returns (samples, generators).
    """
    samples = []
    generators = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if difficulty and row.get("difficulty", "") != difficulty:
                continue
            clip_path = os.path.join(
                deeptrace_root, row["generator"], f"{row['clip_id']}.mp4"
            )
            if not os.path.exists(clip_path):
                continue
            label = 0 if row["label"] == "real" else 1
            samples.append((clip_path, label))
            generators.append(row["generator"])
    return samples, generators


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


def _eval_on_samples(model, device, samples, batch_size=16) -> dict:
    """Evaluate model on a list of (path, label) samples. Returns metrics dict."""
    if not samples:
        return {"auc": float("nan"), "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "confusion_matrix": None}
    loader = DataLoader(
        VideoClipDataset(samples, CLIP_TRANSFORM, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    y_true, y_scores = _predict(model, loader, device)
    return compute_all_metrics(y_true, y_scores)


def run_evaluation(model_name: str, dataset: str, split: str,
                   difficulty: str = None,
                   checkpoint_path: str = None,
                   manifest_path: str = DEFAULT_MANIFEST,
                   deeptrace_root: str = DEFAULT_DEEPTRACE_ROOT,
                   batch_size: int = 16) -> EvalResult:
    """
    Run evaluation for one model on one dataset split.

    Args:
        model_name:      Model to evaluate (deeptrace)
        dataset:         Dataset to evaluate on (deeptrace_gv)
        split:           Data split (train | val | test)
        difficulty:      Optional difficulty filter (easy | medium | hard)
        checkpoint_path: Path to .pth checkpoint
        manifest_path:   Path to staging_manifest.csv
        deeptrace_root:  Root of reencoded clips
        batch_size:      Batch size for inference

    Returns:
        EvalResult with full metrics
    """
    random.seed(42)
    torch.manual_seed(42)

    model, device = load_model(model_name, checkpoint_path)
    samples, generators = _load_deeptrace_test_samples(
        manifest_path, deeptrace_root, split, difficulty
    )

    if not samples:
        print(f"No samples found for split={split}, difficulty={difficulty}")
        return EvalResult(model_name=model_name, dataset_name=dataset)

    # Single inference pass: frame sampling is random per pass, so the
    # per-generator breakdown must reuse the same predictions as the
    # headline metrics to stay mutually consistent.
    loader = DataLoader(
        VideoClipDataset(samples, CLIP_TRANSFORM, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )
    y_true, y_scores = _predict(model, loader, device)
    metrics = compute_all_metrics(y_true, y_scores)
    per_gen_auc = compute_per_generator_auc(y_true, y_scores, np.array(generators))

    dataset_label = dataset
    if difficulty:
        dataset_label = f"{dataset} ({difficulty})"

    result = EvalResult(
        model_name=model_name,
        dataset_name=dataset_label,
        auc=metrics["auc"],
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        confusion_mat=metrics["confusion_matrix"],
        per_generator_auc=per_gen_auc,
    )

    print_results_table([result])
    return result


def run_cross_manipulation(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
                           manifest_path: str = DEFAULT_MANIFEST,
                           deeptrace_root: str = DEFAULT_DEEPTRACE_ROOT,
                           celebdf_root: str = DEFAULT_CELEBDF_ROOT,
                           output_dir: str = "data/results",
                           batch_size: int = 16) -> dict:
    """
    Run the full cross-manipulation evaluation matrix.

    Tests each of the three DeepTrace variants (ai_video, faceswap, joint)
    against both DeepTrace-GV and Celeb-DF v2 test sets to measure the
    cross-manipulation AUC gap.

    Celeb-DF protocol: the ai_video variant never trained on Celeb-DF, so
    it is evaluated on the full 518-clip test list. The faceswap and joint
    variants used 80% of that list for training, so they are evaluated only
    on the held-out 20% slice, re-derived with the seed saved in each
    checkpoint (requires the same Celeb-DF files on disk as at training).

    Args:
        checkpoint_dir: Directory containing the 3 variant checkpoints
        manifest_path:  Path to staging_manifest.csv
        deeptrace_root: Root of reencoded clips
        celebdf_root:   Root of Celeb-DF v2 dataset
        output_dir:     Where to write evaluation_results.json
        batch_size:     Batch size for inference

    Returns:
        Dict matching the evaluation_results.json schema
    """
    dt_test_samples = load_deeptrace_samples(
        manifest_path, deeptrace_root, "test", "joint"
    )
    print(f"DeepTrace-GV test: {len(dt_test_samples)} clips")

    variants = {
        "ai_video": "deeptrace_ai_video_full_best.pth",
        "faceswap": "deeptrace_faceswap_full_best.pth",
        "joint": "deeptrace_joint_full_best.pth",
    }

    output = {}

    def _metric_dict(m, n):
        return {
            "auc": round(m["auc"], 4) if not np.isnan(m["auc"]) else None,
            "acc": round(m["accuracy"], 4),
            "f1": round(m["f1"], 4),
            "n_clips": n,
        }

    for variant_name, ckpt_filename in variants.items():
        ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
        if not os.path.exists(ckpt_path):
            print(f"\n[warning] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {variant_name}")
        print(f"{'='*50}")

        # Re-derive the Celeb-DF split exactly as training did: same seed
        # (stored in the checkpoint) immediately before the shuffle in
        # load_celebdf_samples. The on-disk file set must also match training.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_seed = ckpt.get("args", {}).get("seed", 42)
        random.seed(ckpt_seed)
        celebdf_train, celebdf_val = load_celebdf_samples(celebdf_root)

        if variant_name == "ai_video":
            # Never trained on Celeb-DF, so the full 518-clip test list is
            # leakage-free and matches the guide's stated sample size.
            celebdf_eval = celebdf_train + celebdf_val
        else:
            # celebdf_train was training data for faceswap/joint: only the
            # held-out 20% val slice is a valid test set.
            celebdf_eval = celebdf_val

        model, device = load_model("deeptrace", ckpt_path)

        dt_metrics = _eval_on_samples(model, device, dt_test_samples, batch_size)
        cdf_metrics = _eval_on_samples(model, device, celebdf_eval, batch_size)

        print(f"  DeepTrace-GV ({len(dt_test_samples)} clips): "
              f"AUC={dt_metrics['auc']:.4f} "
              f"Acc={dt_metrics['accuracy']:.4f} F1={dt_metrics['f1']:.4f}")
        print(f"  Celeb-DF v2 ({len(celebdf_eval)} clips): "
              f"AUC={cdf_metrics['auc']:.4f} "
              f"Acc={cdf_metrics['accuracy']:.4f} F1={cdf_metrics['f1']:.4f}")

        if variant_name == "ai_video":
            gap = cross_manipulation_gap(dt_metrics["auc"], cdf_metrics["auc"])
            output["ai_video_variant"] = {
                "in_dist_deeptrace_gv": _metric_dict(dt_metrics, len(dt_test_samples)),
                "cross_celeb_df": _metric_dict(cdf_metrics, len(celebdf_eval)),
                "cross_manipulation_gap": round(gap, 4) if not np.isnan(gap) else None,
            }
            print(f"  Cross-manipulation gap: {gap:.4f}")

        elif variant_name == "faceswap":
            gap = cross_manipulation_gap(cdf_metrics["auc"], dt_metrics["auc"])
            output["faceswap_variant"] = {
                "in_dist_celeb_df": _metric_dict(cdf_metrics, len(celebdf_eval)),
                "cross_deeptrace_gv": _metric_dict(dt_metrics, len(dt_test_samples)),
                "cross_manipulation_gap": round(gap, 4) if not np.isnan(gap) else None,
            }
            print(f"  Cross-manipulation gap: {gap:.4f}")

        elif variant_name == "joint":
            output["joint_variant"] = {
                "deeptrace_gv": _metric_dict(dt_metrics, len(dt_test_samples)),
                "celeb_df": _metric_dict(cdf_metrics, len(celebdf_eval)),
            }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


def run_failure_analysis(model_name: str, n_examples: int = 20,
                         checkpoint_path: str = None,
                         manifest_path: str = DEFAULT_MANIFEST,
                         deeptrace_root: str = DEFAULT_DEEPTRACE_ROOT,
                         output_dir: str = "data/results") -> dict:
    """
    Pull the n most-confident false positives and false negatives.
    Group by scene_type and generator.

    Args:
        model_name:      Model to analyze
        n_examples:      Number of failure cases to collect per category
        checkpoint_path: Path to .pth checkpoint
        manifest_path:   Path to staging_manifest.csv
        deeptrace_root:  Root of reencoded clips
        output_dir:      Where to write results JSON

    Returns:
        Dict with false_positives and false_negatives lists
    """
    model, device = load_model(model_name, checkpoint_path)

    rows_by_idx = []
    samples = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            clip_path = os.path.join(
                deeptrace_root, row["generator"], f"{row['clip_id']}.mp4"
            )
            if not os.path.exists(clip_path):
                continue
            label = 0 if row["label"] == "real" else 1
            samples.append((clip_path, label))
            rows_by_idx.append(row)

    if not samples:
        print("No test samples found for failure analysis.")
        return {"false_positives": [], "false_negatives": []}

    loader = DataLoader(
        VideoClipDataset(samples, CLIP_TRANSFORM, augment=False),
        batch_size=16, shuffle=False, num_workers=2,
    )
    y_true, y_scores = _predict(model, loader, device)
    y_pred = (y_scores >= 0.5).astype(int)

    false_positives = []
    false_negatives = []

    for i in range(len(y_true)):
        entry = {
            "clip_id":    rows_by_idx[i]["clip_id"],
            "generator":  rows_by_idx[i]["generator"],
            "scene_type": rows_by_idx[i].get("scene_type", ""),
            "true_label": int(y_true[i]),
            "pred_prob":  round(float(y_scores[i]), 4),
        }
        if y_true[i] == 0 and y_pred[i] == 1:
            false_positives.append(entry)
        elif y_true[i] == 1 and y_pred[i] == 0:
            false_negatives.append(entry)

    false_positives.sort(key=lambda x: x["pred_prob"], reverse=True)
    false_negatives.sort(key=lambda x: x["pred_prob"])

    result = {
        "false_positives": false_positives[:n_examples],
        "false_negatives": false_negatives[:n_examples],
        "total_test_clips": len(y_true),
        "total_fp": len(false_positives),
        "total_fn": len(false_negatives),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "failure_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Failure analysis saved to {out_path}")
    print(f"  False positives: {len(false_positives)}")
    print(f"  False negatives: {len(false_negatives)}")

    return result


def run_speed_benchmark(model_name: str, n_clips: int = 100,
                        checkpoint_path: str = None,
                        frames_per_clip: int = 5,
                        output_dir: str = "data/results") -> dict:
    """
    Measure end-to-end inference latency and FPS on the current GPU.

    Args:
        model_name:      Model to benchmark
        n_clips:         Number of clips to average over
        checkpoint_path: Path to .pth checkpoint
        frames_per_clip: Frames per synthetic clip
        output_dir:      Where to write results JSON

    Returns:
        Dict with keys: mean_latency_ms, std_latency_ms, throughput_fps, device
    """
    model, device = load_model(model_name, checkpoint_path)

    dummy_batch = torch.randn(frames_per_clip, 3, 224, 224).to(device)

    for _ in range(5):
        with torch.no_grad():
            model.forward_frames(dummy_batch)

    times = []
    for _ in range(n_clips):
        t0 = time.time()
        with torch.no_grad():
            logits = model.forward_frames(dummy_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            _ = float(probs.mean())
        times.append(time.time() - t0)

    mean_ms = float(np.mean(times) * 1000)
    std_ms = float(np.std(times) * 1000)
    fps = frames_per_clip / float(np.mean(times))

    result = {
        "model": model_name,
        "device": device,
        "frames_per_clip": frames_per_clip,
        "n_clips_tested": n_clips,
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "throughput_fps": round(fps, 2),
        "sub_second_pass": mean_ms < 1000,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "speed_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Speed benchmark saved to {out_path}")
    print(f"  Mean latency: {mean_ms:.2f} ms/clip")
    print(f"  Throughput:   {fps:.1f} FPS")
    print(f"  Sub-second:   {'PASS' if result['sub_second_pass'] else 'FAIL'}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=MODELS, default="deeptrace")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--manifest",   type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--deeptrace-root", type=str, default=DEFAULT_DEEPTRACE_ROOT)
    parser.add_argument("--celebdf-root", type=str, default=DEFAULT_CELEBDF_ROOT)
    parser.add_argument("--cross-manipulation", action="store_true",
                        help="Run full cross-manipulation evaluation matrix")
    parser.add_argument("--failure-analysis", action="store_true",
                        help="Run failure analysis on the test set")
    parser.add_argument("--speed-benchmark", action="store_true",
                        help="Run speed benchmark")
    args = parser.parse_args()

    if args.cross_manipulation:
        run_cross_manipulation(
            checkpoint_dir=args.checkpoint_dir,
            manifest_path=args.manifest,
            deeptrace_root=args.deeptrace_root,
            celebdf_root=args.celebdf_root,
        )
    elif args.speed_benchmark:
        run_speed_benchmark(args.model, checkpoint_path=args.checkpoint)
    elif args.failure_analysis:
        run_failure_analysis(args.model, checkpoint_path=args.checkpoint,
                             manifest_path=args.manifest,
                             deeptrace_root=args.deeptrace_root)
    else:
        run_evaluation(args.model, "deeptrace_gv", args.split, args.difficulty,
                       checkpoint_path=args.checkpoint,
                       manifest_path=args.manifest,
                       deeptrace_root=args.deeptrace_root)

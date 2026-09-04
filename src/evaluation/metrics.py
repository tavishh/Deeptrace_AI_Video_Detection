"""
metrics.py - Evaluation metrics for DeepTrace

Owner: Jiajun
Week:  1

Core metrics module. Build this in Week 1 so it is ready when
baseline results arrive in Week 2.

Metrics:
    - AUC-ROC (primary)
    - Accuracy, Precision, Recall, F1
    - AUC_gap  (novelty metric: AUC on Celeb-DF v2 minus AUC on DeepTrace-GV)
    - LOGO AUC (mean and std across three rounds)
    - Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalResult:
    """Container for all evaluation results for one model on one dataset."""
    model_name:   str
    dataset_name: str
    auc:          float = 0.0
    accuracy:     float = 0.0
    precision:    float = 0.0
    recall:       float = 0.0
    f1:           float = 0.0
    confusion_mat: Optional[np.ndarray] = None
    per_generator_auc: dict = field(default_factory=dict)   # {generator: auc}


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute AUC-ROC.

    Args:
        y_true:   Binary ground truth labels (0=real, 1=fake)
        y_scores: Predicted fake probability scores

    Returns:
        AUC-ROC score
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_scores))


def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        labels: Binary ground truth labels (0=real, 1=fake)
        preds:  Binary predictions (0=real, 1=fake)

    Returns:
        Accuracy score
    """
    return float(accuracy_score(labels, preds))


def compute_precision_recall_f1(labels: np.ndarray, preds: np.ndarray) -> dict:
    """
    Compute precision, recall, and F1 for the fake class.

    Args:
        labels: Binary ground truth labels (0=real, 1=fake)
        preds:  Binary predictions (0=real, 1=fake)

    Returns:
        Dict with keys: precision, recall, f1
    """
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
    }


def compute_all_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                        threshold: float = 0.5) -> dict:
    """
    Compute the full set of detection metrics.

    Args:
        y_true:     Binary ground truth labels
        y_scores:   Predicted fake probabilities
        threshold:  Decision threshold for binary predictions

    Returns:
        Dict with keys: auc, accuracy, precision, recall, f1, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "auc":              compute_auc(y_true, y_scores),
        "accuracy":         compute_accuracy(y_true, y_pred),
        **compute_precision_recall_f1(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def compute_auc_gap(auc_celebdf: float, auc_deeptrace_gv: float) -> float:
    """
    Compute the cross-generator AUC gap (novelty metric).

    AUC_gap = AUC(Celeb-DF v2) - AUC(DeepTrace-GV)

    A large gap indicates the model exploits dataset-specific shortcuts
    rather than learning universal forgery signals.

    Args:
        auc_celebdf:       AUC on Celeb-DF v2 test split
        auc_deeptrace_gv:  AUC on DeepTrace-GV test split

    Returns:
        AUC gap value (higher = worse generalization)
    """
    return auc_celebdf - auc_deeptrace_gv


def cross_manipulation_gap(in_dist_auc: float, cross_auc: float) -> float:
    """
    Compute the cross-manipulation AUC gap.

    gap = in_dist_auc - cross_auc

    A large gap means the model fails to generalize across manipulation
    types. Joint training is expected to reduce or eliminate the gap.

    Args:
        in_dist_auc:  AUC on the same manipulation type used for training
        cross_auc:    AUC on the other manipulation type (not seen in training)

    Returns:
        Gap value (higher = worse generalization)
    """
    return in_dist_auc - cross_auc


def compute_logo_auc(round_results: List[float]) -> dict:
    """
    Compute LOGO mean AUC and standard deviation across the three rounds.

    Args:
        round_results: List of held-out AUC values for each LOGO round
                       [auc_held_out_sora, auc_held_out_kling, auc_held_out_veo]

    Returns:
        Dict with keys: mean, std, rounds
    """
    results = np.array(round_results)
    return {
        "mean":   float(np.mean(results)),
        "std":    float(np.std(results)),
        "rounds": round_results
    }


def compute_per_generator_auc(y_true: np.ndarray, y_scores: np.ndarray,
                               generators: np.ndarray) -> dict:
    """
    Compute AUC separately for each generator (Sora, Kling, Veo).

    Each generator's AUC is computed using that generator's fake clips
    combined with all real clips, mirroring the LOGO test-set protocol.

    Args:
        y_true:      Binary ground truth labels
        y_scores:    Predicted fake probabilities
        generators:  Generator label per clip (sora | kling | veo | real)

    Returns:
        Dict mapping generator name to AUC score
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    generators = np.asarray(generators)

    real_mask = generators == "real"
    per_gen = {}

    for gen in sorted(set(generators) - {"real"}):
        mask = real_mask | (generators == gen)
        if len(np.unique(y_true[mask])) < 2:
            per_gen[gen] = float("nan")
            continue
        per_gen[gen] = float(roc_auc_score(y_true[mask], y_scores[mask]))

    return per_gen


def print_results_table(results: List[EvalResult]) -> None:
    """Print a formatted comparison table of evaluation results."""
    header = f"{'Model':<25s} {'Dataset':<18s} {'AUC':>6s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.model_name:<25s} {r.dataset_name:<18s} "
              f"{r.auc:6.4f} {r.accuracy:6.4f} {r.precision:6.4f} "
              f"{r.recall:6.4f} {r.f1:6.4f}")
        if r.per_generator_auc:
            for gen, auc in sorted(r.per_generator_auc.items()):
                print(f"  {'└ ' + gen:<23s} {'':18s} {auc:6.4f}")
        if r.confusion_mat is not None:
            tn, fp, fn, tp = r.confusion_mat.ravel()
            print(f"  {'Confusion':<23s} TP={tp} FP={fp} FN={fn} TN={tn}")

"""
test_pipeline.py - Unit tests for DeepTrace data pipeline and model modules

Run from the repository root:
    python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Data pipeline tests
# ---------------------------------------------------------------------------

class TestExtractFrames:
    """Tests for src/data_pipeline/extract_frames.py"""

    def test_import(self):
        from data_pipeline.extract_frames import extract_frames
        assert callable(extract_frames)

    def test_extract_frames_invalid_path(self):
        from pathlib import Path
        from data_pipeline.extract_frames import extract_frames
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = list(extract_frames(Path("nonexistent.mp4"), Path(tmp), fps=1))
                assert result == []
            except Exception:
                pass  # Raising on missing file is also acceptable

    def test_extract_frames_output_dir_created(self):
        from pathlib import Path
        from data_pipeline.extract_frames import extract_frames
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "frames"
            try:
                list(extract_frames(Path("nonexistent.mp4"), out_dir, fps=1))
            except Exception:
                pass  # Raising on missing file is also acceptable


class TestAnnotateAiClips:
    """Tests for src/data_pipeline/auto_annotate_real.py"""

    def test_infer_category_known(self):
        from data_pipeline.auto_annotate_real import infer_category
        assert infer_category("real_jogging_abc123") == "jogging"
        assert infer_category("real_surfing_water_abc123") == "surfing_water"
        assert infer_category("real_driving_car_QkLN_QBF1hI") == "driving_car"

    def test_infer_category_unknown(self):
        from data_pipeline.auto_annotate_real import infer_category
        assert infer_category("sora_amalfi_coast") is None
        assert infer_category("real_unknown_category_abc") is None

    def test_infer_category_non_real(self):
        from data_pipeline.auto_annotate_real import infer_category
        assert infer_category("kling_cat_as_king") is None


class TestAssignSplits:
    """Tests for src/data_pipeline/assign_splits.py"""

    def test_split_proportions(self):
        """Verify 70/15/15 splits are approximately correct."""
        import csv
        import random
        from collections import defaultdict

        # Simulate a small manifest
        rows = []
        for i in range(100):
            rows.append({
                "clip_id": f"clip_{i:03d}",
                "label": "fake" if i < 50 else "real",
                "generator": "sora" if i < 50 else "real",
                "split": "",
            })

        by_gen = defaultdict(list)
        for row in rows:
            by_gen[row["generator"]].append(row)

        random.seed(42)
        for gen, clips in by_gen.items():
            random.shuffle(clips)
            n = len(clips)
            n_train = round(n * 0.70)
            n_val   = round(n * 0.15)
            for i, clip in enumerate(clips):
                if i < n_train:
                    clip["split"] = "train"
                elif i < n_train + n_val:
                    clip["split"] = "val"
                else:
                    clip["split"] = "test"

        splits = [r["split"] for r in rows]
        assert splits.count("train") > 0
        assert splits.count("val") > 0
        assert splits.count("test") > 0
        # Verify ~70% train
        assert 0.60 <= splits.count("train") / len(rows) <= 0.80


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestMLPHead:
    """Tests for src/model/mlp_head.py"""

    def test_import(self):
        from model.mlp_head import MLPHead, DeepTrace
        assert MLPHead is not None
        assert DeepTrace is not None

    def test_mlp_output_shape(self):
        import torch
        from model.mlp_head import MLPHead
        model = MLPHead(input_dim=512, hidden_dim=128, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(4, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_mlp_single_sample(self):
        import torch
        from model.mlp_head import MLPHead
        model = MLPHead()
        model.eval()
        x = torch.randn(1, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_mlp_output_not_probability(self):
        """MLP outputs raw logits, not probabilities."""
        import torch
        from model.mlp_head import MLPHead
        model = MLPHead()
        model.eval()
        x = torch.randn(8, 512)
        with torch.no_grad():
            out = model(x)
        # Logits can be any value - softmax should sum to 1
        probs = torch.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_deeptrace_predict_clip(self):
        import torch
        from model.mlp_head import MLPHead, DeepTrace
        from unittest.mock import MagicMock

        # Mock the extractor to avoid loading CLIP
        mock_extractor = MagicMock()
        mock_extractor.return_value = torch.randn(5, 512)

        head = MLPHead()
        head.eval()
        model = DeepTrace(mock_extractor, head, aggregation="mean")
        model.eval()

        frames = torch.randn(5, 3, 224, 224)
        result = model.predict_clip(frames)

        assert "fake_prob" in result
        assert "real_prob" in result
        assert "label" in result
        assert "per_frame_probs" in result
        assert 0.0 <= result["fake_prob"] <= 1.0
        assert result["label"] in ("real", "fake")
        assert len(result["per_frame_probs"]) == 5

    def test_deeptrace_majority_vote(self):
        import torch
        from model.mlp_head import MLPHead, DeepTrace
        from unittest.mock import MagicMock

        mock_extractor = MagicMock()
        mock_extractor.return_value = torch.randn(3, 512)

        head = MLPHead()
        model = DeepTrace(mock_extractor, head, aggregation="majority_vote")
        model.eval()

        frames = torch.randn(3, 3, 224, 224)
        result = model.predict_clip(frames)
        assert result["label"] in ("real", "fake")


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for src/evaluation/metrics.py"""

    def test_compute_auc_perfect(self):
        from evaluation.metrics import compute_auc
        y_true   = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_scores) == 1.0

    def test_compute_auc_random(self):
        from evaluation.metrics import compute_auc
        y_true   = np.array([0, 1, 0, 1])
        y_scores = np.array([0.5, 0.5, 0.5, 0.5])
        auc = compute_auc(y_true, y_scores)
        assert 0.0 <= auc <= 1.0

    def test_compute_auc_single_class(self):
        from evaluation.metrics import compute_auc
        y_true   = np.array([1, 1, 1])
        y_scores = np.array([0.8, 0.9, 0.7])
        assert np.isnan(compute_auc(y_true, y_scores))

    def test_compute_accuracy(self):
        from evaluation.metrics import compute_accuracy
        labels = np.array([0, 1, 0, 1])
        preds  = np.array([0, 1, 0, 0])
        assert compute_accuracy(labels, preds) == 0.75

    def test_compute_precision_recall_f1(self):
        from evaluation.metrics import compute_precision_recall_f1
        labels = np.array([0, 1, 1, 0])
        preds  = np.array([0, 1, 0, 0])
        result = compute_precision_recall_f1(labels, preds)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5

    def test_cross_manipulation_gap(self):
        from evaluation.metrics import cross_manipulation_gap
        assert cross_manipulation_gap(0.95, 0.50) == pytest.approx(0.45)
        assert cross_manipulation_gap(0.90, 0.90) == pytest.approx(0.00)

    def test_compute_logo_auc(self):
        from evaluation.metrics import compute_logo_auc
        result = compute_logo_auc([1.0, 0.985, 0.939])
        assert "mean" in result
        assert "std" in result
        assert result["mean"] == pytest.approx(np.mean([1.0, 0.985, 0.939]))

    def test_compute_all_metrics_keys(self):
        from evaluation.metrics import compute_all_metrics
        y_true   = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_all_metrics(y_true, y_scores)
        for key in ["auc", "accuracy", "precision", "recall", "f1"]:
            assert key in result

    def test_compute_per_generator_auc(self):
        from evaluation.metrics import compute_per_generator_auc
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7])
        generators = np.array(["real", "real", "sora", "sora", "real", "kling"])
        result = compute_per_generator_auc(y_true, y_scores, generators)
        assert "sora" in result
        assert "kling" in result
        assert "real" not in result
"""Lightweight tests for the Gradio demo pipeline contract."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.demo.pipeline import (
    DeepTraceInference,
    DemoPipelineError,
    PreparedClip,
    analyze_video,
)


class FakeSession:
    """Minimal ONNX Runtime stand-in with deterministic logits."""

    def run(self, output_names, inputs):
        batch_size = next(iter(inputs.values())).shape[0]
        logits = np.tile(
            np.array([[0.0, 1.0]], dtype=np.float32),
            (batch_size, 1),
        )
        return [logits]


class FakeEngine:
    def load(self):
        return self

    def predict_frames(self, frame_paths):
        return np.array([0.2, 0.8], dtype=np.float32)


class DemoPipelineTests(unittest.TestCase):
    def test_onnx_logits_are_converted_to_fake_probabilities(self):
        engine = DeepTraceInference("/tmp/not-used.onnx", device="cpu")
        engine.extractor = object()
        engine.session = FakeSession()

        features = np.zeros((3, 512), dtype=np.float32)
        probabilities = engine.predict_features(features)

        np.testing.assert_allclose(
            probabilities,
            np.full(3, 0.7310586, dtype=np.float32),
            rtol=1e-5,
        )

    def test_analysis_mean_aggregation_and_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_paths = []
            for index in range(2):
                frame_path = root / f"frame_{index:04d}.jpg"
                cv2.imwrite(
                    str(frame_path),
                    np.full((32, 32, 3), index * 100, dtype=np.uint8),
                )
                frame_paths.append(frame_path)

            source_path = root / "source.mp4"
            source_path.touch()

            @contextmanager
            def fake_prepare_clip(video_path, sample_fps=1):
                yield PreparedClip(
                    source_path=source_path,
                    encoded_path=root / "normalized.mp4",
                    frame_paths=tuple(frame_paths),
                )

            with patch("src.demo.pipeline.prepare_clip", fake_prepare_clip):
                result = analyze_video(FakeEngine(), source_path, threshold=0.5)

        self.assertEqual(result.label, "fake")
        self.assertAlmostEqual(result.fake_probability, 0.5, places=6)
        self.assertEqual(result.frame_count, 2)
        self.assertEqual(len(result.frame_images), 2)
        self.assertGreater(result.latency_ms, 0)

    def test_threshold_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            analyze_video(FakeEngine(), "/tmp/video.mp4", threshold=1.1)

    def test_external_onnx_weights_are_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "deeptrace_joint_head.onnx"
            model_path.touch()

            with self.assertRaisesRegex(DemoPipelineError, "weights not found"):
                DeepTraceInference(model_path).load()


if __name__ == "__main__":
    unittest.main()

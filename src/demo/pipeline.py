"""Reusable preprocessing and inference helpers for the DeepTrace demo.

The demo intentionally mirrors the finalized evaluation path: every uploaded
clip is normalized before frames are sampled.  Temporary artifacts live only
for the duration of one prediction request.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from src.data_pipeline.extract_frames import extract_frames


DEFAULT_SAMPLE_FPS = 1
FFMPEG_TIMEOUT_SECONDS = 300
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_MODEL = REPO_ROOT / "models/onnx/deeptrace_joint_head.onnx"


class DemoPipelineError(RuntimeError):
    """Raised when an uploaded clip cannot be prepared for inference."""


@dataclass(frozen=True)
class PreparedClip:
    """Paths created while preparing a single uploaded video."""

    source_path: Path
    encoded_path: Path
    frame_paths: tuple[Path, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)


@dataclass(frozen=True)
class PredictionResult:
    """Clip-level and frame-level outputs returned to the Gradio layer."""

    label: str
    fake_probability: float
    real_probability: float
    frame_probabilities: tuple[float, ...]
    frame_images: tuple[np.ndarray, ...]
    threshold: float
    elapsed_seconds: float

    @property
    def frame_count(self) -> int:
        return len(self.frame_probabilities)

    @property
    def latency_ms(self) -> float:
        return self.elapsed_seconds * 1_000.0

    @property
    def clips_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return 1.0 / self.elapsed_seconds

    @property
    def frames_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frame_count / self.elapsed_seconds


def _validate_source(video_path: str | Path) -> Path:
    """Return a resolved upload path or raise a user-facing pipeline error."""
    if not video_path:
        raise DemoPipelineError("Please upload a video before running detection.")

    source_path = Path(video_path).expanduser().resolve()
    if not source_path.is_file():
        raise DemoPipelineError(f"Uploaded video was not found: {source_path}")
    if source_path.stat().st_size == 0:
        raise DemoPipelineError("The uploaded video is empty.")
    return source_path


def _require_ffmpeg() -> str:
    """Locate ffmpeg so startup failures are clear and actionable."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise DemoPipelineError(
            "ffmpeg is required for video normalization but was not found on PATH."
        )
    return executable


def reencode_for_inference(source_path: Path, output_path: Path) -> None:
    """Normalize one clip to H.264, CRF 23, 1280x720 with audio removed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _require_ffmpeg(),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vf",
        (
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2"
        ),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        "-an",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DemoPipelineError("Video re-encoding timed out after 5 minutes.") from exc
    except OSError as exc:
        raise DemoPipelineError(f"Unable to start ffmpeg: {exc}") from exc

    if result.returncode != 0 or not output_path.is_file():
        detail = result.stderr.strip() or "ffmpeg did not create an output video."
        raise DemoPipelineError(f"Video re-encoding failed: {detail}")


@contextmanager
def prepare_clip(
    video_path: str | Path,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
) -> Iterator[PreparedClip]:
    """Re-encode and sample an upload, cleaning all temporary files afterward."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")

    source_path = _validate_source(video_path)
    with tempfile.TemporaryDirectory(prefix="deeptrace-") as temp_dir:
        workspace = Path(temp_dir)
        encoded_path = workspace / "normalized.mp4"
        frames_dir = workspace / "frames"

        reencode_for_inference(source_path, encoded_path)
        frame_paths = tuple(extract_frames(encoded_path, frames_dir, fps=sample_fps))
        if not frame_paths:
            raise DemoPipelineError(
                "No frames could be extracted from the uploaded video."
            )

        yield PreparedClip(
            source_path=source_path,
            encoded_path=encoded_path,
            frame_paths=frame_paths,
        )


class DeepTraceInference:
    """Frozen CLIP feature extractor plus ONNX classification head."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ONNX_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.external_data_path = self.model_path.with_name(
            f"{self.model_path.name}.data"
        )
        self.device = device
        self.extractor = None
        self.session = None
        self.input_name = "clip_features"
        self.output_name = "logits"

    def load(self) -> "DeepTraceInference":
        """Load both models once and validate the expected ONNX contract."""
        if self.extractor is not None and self.session is not None:
            return self

        if not self.model_path.is_file():
            raise DemoPipelineError(
                f"ONNX graph not found at {self.model_path}. "
                "Download the final joint model from the shared Drive folder."
            )
        if not self.external_data_path.is_file():
            raise DemoPipelineError(
                f"ONNX weights not found at {self.external_data_path}. "
                "The .onnx and .onnx.data files must stay in the same directory."
            )

        try:
            import onnxruntime as ort
            import torch

            from src.model.clip_extractor import CLIPExtractor
        except ImportError as exc:
            raise DemoPipelineError(
                "Demo dependencies are missing. Install requirements.txt first."
            ) from exc

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        try:
            self.extractor = CLIPExtractor().to(self.device)
            self.extractor.eval()
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            self.extractor = None
            self.session = None
            raise DemoPipelineError(f"Unable to load inference models: {exc}") from exc

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise DemoPipelineError(
                "The ONNX head must expose exactly one input and one output."
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        if self.input_name != "clip_features" or self.output_name != "logits":
            raise DemoPipelineError(
                "Unexpected ONNX interface: expected clip_features -> logits, "
                f"received {self.input_name} -> {self.output_name}."
            )
        return self

    def extract_features(self, frame_paths: tuple[Path, ...]) -> np.ndarray:
        """Convert sampled frames into L2-normalized 512-dim CLIP features."""
        self.load()
        import torch

        tensors = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                tensors.append(self.extractor.preprocess(image.convert("RGB")))

        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            features = self.extractor(batch)

        array = features.detach().cpu().numpy().astype(np.float32, copy=False)
        if array.ndim != 2 or array.shape[1] != 512:
            raise DemoPipelineError(
                f"CLIP returned {array.shape}; expected (batch_size, 512)."
            )
        return array

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Run the ONNX head and return the fake probability for every frame."""
        self.load()
        logits = self.session.run(
            [self.output_name],
            {self.input_name: features.astype(np.float32, copy=False)},
        )[0]
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise DemoPipelineError(
                f"ONNX returned {logits.shape}; expected (batch_size, 2)."
            )

        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 1]

    def predict_frames(self, frame_paths: tuple[Path, ...]) -> np.ndarray:
        """Extract CLIP features and classify every sampled frame."""
        return self.predict_features(self.extract_features(frame_paths))


def analyze_video(
    engine: DeepTraceInference,
    video_path: str | Path,
    threshold: float = 0.5,
) -> PredictionResult:
    """Run the finalized end-to-end demo pipeline for one uploaded video."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    engine.load()
    started_at = time.perf_counter()
    with prepare_clip(video_path, sample_fps=DEFAULT_SAMPLE_FPS) as clip:
        frame_probabilities = engine.predict_frames(clip.frame_paths)
        frame_images = []
        for frame_path in clip.frame_paths:
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                raise DemoPipelineError(f"Unable to read sampled frame: {frame_path}")
            frame_images.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    elapsed_seconds = time.perf_counter() - started_at
    fake_probability = float(np.mean(frame_probabilities))
    return PredictionResult(
        label="fake" if fake_probability >= threshold else "real",
        fake_probability=fake_probability,
        real_probability=1.0 - fake_probability,
        frame_probabilities=tuple(float(value) for value in frame_probabilities),
        frame_images=tuple(frame_images),
        threshold=float(threshold),
        elapsed_seconds=elapsed_seconds,
    )

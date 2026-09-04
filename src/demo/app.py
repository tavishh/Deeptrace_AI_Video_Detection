"""DeepTrace Gradio demo for real-time AI-generated video detection.

Finalized inference path:
    upload -> H.264/CRF23/720p re-encode -> 1 FPS frame sampling
    -> frozen CLIP ViT-B/32 features -> ONNX head -> mean aggregation

Run from the repository root with ``python src/demo/app.py``.
"""

from __future__ import annotations

import html
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.pipeline import (  # noqa: E402
    DEFAULT_ONNX_MODEL,
    DeepTraceInference,
    DemoPipelineError,
    PredictionResult,
    analyze_video,
)


APP_CSS = (Path(__file__).with_name("styles.css")).read_text(encoding="utf-8")
APP_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=["Manrope", "sans-serif"],
    font_mono=["Manrope", "sans-serif"],
).set(
    background_fill_primary="#f7f3eb",
    background_fill_primary_dark="#f7f3eb",
    background_fill_secondary="#fbfaf6",
    background_fill_secondary_dark="#fbfaf6",
    block_background_fill="#fbfaf6",
    block_background_fill_dark="#fbfaf6",
    block_border_color="#d9dde3",
    block_border_color_dark="#d9dde3",
    block_label_background_fill="#f7f3eb",
    block_label_background_fill_dark="#f7f3eb",
    block_label_text_color="#52617a",
    block_label_text_color_dark="#52617a",
    body_text_color="#0b1f3a",
    body_text_color_dark="#0b1f3a",
    body_text_color_subdued="#5f6f86",
    body_text_color_subdued_dark="#5f6f86",
    border_color_primary="#d9dde3",
    border_color_primary_dark="#d9dde3",
    input_background_fill="#fbfaf6",
    input_background_fill_dark="#fbfaf6",
    input_border_color="#d9dde3",
    input_border_color_dark="#d9dde3",
    button_secondary_background_fill="#fbfaf6",
    button_secondary_background_fill_dark="#fbfaf6",
    button_secondary_border_color="#cdd3dc",
    button_secondary_border_color_dark="#cdd3dc",
    button_secondary_text_color="#263853",
    button_secondary_text_color_dark="#263853",
)


EMPTY_VERDICT = """
<div class="dt-verdict">
  <div class="dt-verdict-label">Awaiting analysis</div>
  <div class="dt-verdict-value">No verdict yet</div>
  <div class="dt-verdict-rule"></div>
  <div class="dt-verdict-copy">Upload a clip and run the detector.</div>
</div>
"""

EMPTY_METRICS = """
<div class="dt-performance">
  <div class="dt-performance-title">Performance <span>(awaiting analysis)</span></div>
  <div class="dt-metrics">
    <div class="dt-metric"><strong>--</strong><span>ms / clip</span></div>
    <div class="dt-metric"><strong>--</strong><span>clips / sec</span></div>
    <div class="dt-metric"><strong>--</strong><span>sampled frames</span></div>
    <div class="dt-metric"><strong>--</strong><span>frames / sec</span></div>
  </div>
</div>
"""


@lru_cache(maxsize=4)
def load_model(model_path: str = str(DEFAULT_ONNX_MODEL)) -> DeepTraceInference:
    """Load and cache the frozen CLIP encoder and final ONNX joint head."""
    return DeepTraceInference(model_path=model_path).load()


def predict(
    video_path: str,
    threshold: float = 0.5,
    model_path: str = str(DEFAULT_ONNX_MODEL),
) -> PredictionResult:
    """Run end-to-end inference on one uploaded clip."""
    return analyze_video(load_model(model_path), video_path, threshold)


def _verdict_html(result: PredictionResult) -> str:
    verdict = (
        "Deepfake / AI-Generated Content"
        if result.label == "fake"
        else "Likely real"
    )
    confidence = max(result.fake_probability, result.real_probability)
    summary = (
        f"Mean fake probability {result.fake_probability:.1%} at a "
        f"{result.threshold:.0%} decision threshold."
    )
    return f"""
    <div class="dt-verdict {result.label}">
      <div class="dt-verdict-label">Clip verdict</div>
      <div class="dt-verdict-value">{html.escape(verdict)} · {confidence:.1%}</div>
      <div class="dt-verdict-rule"></div>
      <div class="dt-verdict-copy">{html.escape(summary)}</div>
    </div>
    """


def _metrics_html(result: PredictionResult) -> str:
    return f"""
    <div class="dt-performance">
      <div class="dt-performance-title">Performance</div>
      <div class="dt-metrics">
        <div class="dt-metric"><strong>{result.latency_ms:,.0f}</strong><span>ms / clip</span></div>
        <div class="dt-metric"><strong>{result.clips_per_second:.3f}</strong><span>clips / sec</span></div>
        <div class="dt-metric"><strong>{result.frame_count}</strong><span>sampled frames</span></div>
        <div class="dt-metric"><strong>{result.frames_per_second:.1f}</strong><span>frames / sec</span></div>
      </div>
    </div>
    """


def _gallery_items(result: PredictionResult) -> list[tuple[np.ndarray, str]]:
    items = []
    for index, (image, probability) in enumerate(
        zip(result.frame_images, result.frame_probabilities, strict=True)
    ):
        flagged = probability >= result.threshold
        display_image = image.copy()
        if flagged:
            display_image = cv2.copyMakeBorder(
                display_image,
                7,
                7,
                7,
                7,
                cv2.BORDER_CONSTANT,
                value=(255, 112, 103),
            )
        state = "FLAGGED" if flagged else "clear"
        caption = f"{index:02d}s · {probability:.1%} fake · {state}"
        items.append((display_image, caption))
    return items


def _analysis_note(result: PredictionResult) -> str:
    flagged = sum(
        probability >= result.threshold
        for probability in result.frame_probabilities
    )
    return (
        f"**Frame review:** {flagged} of {result.frame_count} sampled frames "
        f"met the {result.threshold:.0%} threshold. The clip verdict uses the "
        "mean fake probability across every sampled frame."
    )


def run_detection(video_path: str, threshold: float):
    """Gradio event handler that converts pipeline output into UI components."""
    try:
        result = predict(video_path, float(threshold))
    except (DemoPipelineError, ValueError) as exc:
        message = html.escape(str(exc))
        error_card = f"""
        <div class="dt-verdict error">
          <div class="dt-verdict-label">Analysis unavailable</div>
          <div class="dt-verdict-value">Check the setup</div>
          <div class="dt-verdict-copy">{message}</div>
        </div>
        """
        return error_card, None, EMPTY_METRICS, [], f"**Unable to run:** {exc}"

    confidence = {
        "Real": result.real_probability,
        "Deepfake / AI-Generated Content": result.fake_probability,
    }
    return (
        _verdict_html(result),
        confidence,
        _metrics_html(result),
        _gallery_items(result),
        _analysis_note(result),
    )


def _reset_interface():
    return None, 0.5, EMPTY_VERDICT, None, EMPTY_METRICS, [], ""


def build_interface() -> gr.Blocks:
    """Build the interactive DeepTrace forensic workspace."""
    with gr.Blocks(
        title="DeepTrace · Video authenticity detector",
        fill_width=True,
    ) as interface:
        with gr.Column(elem_id="dt-shell"):
            gr.HTML(
                """
                <header id="dt-header">
                  <div class="dt-wordmark">DeepTrace</div>
                  <div class="dt-header-divider"></div>
                  <div class="dt-product-name">Video integrity analysis</div>
                </header>
                """
            )

            with gr.Row(equal_height=True, elem_id="dt-control-strip"):
                with gr.Column(scale=5, elem_classes="dt-control-cell dt-source-cell"):
                    video_input = gr.Video(
                        label="Upload video",
                        sources=["upload"],
                        format=None,
                        include_audio=False,
                        height=240,
                        elem_id="video-input",
                    )
                with gr.Column(scale=5, elem_classes="dt-control-cell dt-threshold-cell"):
                    threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.01,
                        label="Detection threshold",
                        info="Frames at or above this fake probability are flagged.",
                        elem_id="threshold-control",
                    )
                with gr.Column(scale=5, elem_classes="dt-control-cell dt-actions-cell"):
                    with gr.Row(elem_id="dt-actions-row"):
                        analyze_button = gr.Button(
                            "Run analysis",
                            variant="primary",
                            elem_id="analyze-button",
                        )
                        clear_button = gr.Button(
                            "Clear",
                            variant="secondary",
                            elem_id="clear-button",
                        )

            with gr.Row(equal_height=False, elem_id="dt-report"):
                with gr.Column(scale=7, elem_id="dt-verdict-column"):
                    verdict = gr.HTML(EMPTY_VERDICT)
                    gr.HTML(
                        """
                        <p class="dt-process-note">Processing includes H.264/CRF23/720p normalization,<br>one frame sampled per second, CLIP feature extraction,<br>and ONNX inference.</p>
                        """
                    )
                with gr.Column(scale=6, elem_id="dt-summary-column"):
                    confidence = gr.Label(
                        label="Class confidence",
                        num_top_classes=2,
                        elem_id="confidence-panel",
                    )
                    performance = gr.HTML(EMPTY_METRICS)

            with gr.Column(elem_id="dt-frame-section"):
                gr.HTML('<div class="dt-section-label">Frame evidence</div>')
                frame_note = gr.Markdown(
                    "Run an analysis to inspect per-frame evidence."
                )
                gallery = gr.Gallery(
                    label="Sampled frames",
                    columns=5,
                    rows=2,
                    height=360,
                    object_fit="contain",
                    allow_preview=True,
                    type="numpy",
                    elem_id="frame-gallery",
                )

            gr.HTML(
                """
                <p class="dt-footnote">Research prototype. Scores indicate model confidence, not definitive proof of manipulation. Review flagged frames alongside source provenance and other forensic evidence.</p>
                """
            )

        analyze_button.click(
            fn=run_detection,
            inputs=[video_input, threshold],
            outputs=[verdict, confidence, performance, gallery, frame_note],
            show_progress="full",
        )
        clear_button.click(
            fn=_reset_interface,
            outputs=[
                video_input,
                threshold,
                verdict,
                confidence,
                performance,
                gallery,
                frame_note,
            ],
            show_progress="hidden",
        )

    return interface


if __name__ == "__main__":
    build_interface().queue(default_concurrency_limit=1).launch(
        css=APP_CSS,
        theme=APP_THEME,
        footer_links=["api", "gradio"],
    )

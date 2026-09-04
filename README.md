# Deeptrace_AI_Video_Detection
DeepTrace addresses the generalization gap in deepfake detection: existing detectors trained on face-swap datasets fail on modern whole-scene AI video generators (Sora, Kling, Veo).
# DeepTrace

**Deepfake and AI-Generated Video Detection with Cross-Manipulation Generalization Analysis**

CS 5330 Pattern Recognition and Computer Vision · Northeastern University
Team: Tavish Hookoom, Zihao Li, Jiajun Huang, Xijia Zeng

---

## Overview

DeepTrace addresses the generalization gap in deepfake detection: existing detectors
trained on face-swap datasets fail on modern whole-scene AI video generators (Sora,
Kling, Veo). We:

1. Build **DeepTrace-GV** - a curated benchmark dataset combining face-swap and
   whole-scene AI-generated video in one place, with 12 metadata fields per clip.
2. Quantify the **cross-manipulation generalization gap** - training on one
   manipulation type and testing on the other drops AUC to 0.44-0.54 (near random).
3. Show that **joint training recovers the gap** - training on both types restores
   AUC to 0.95-0.98.
4. Train **DeepTrace** - a frozen CLIP ViT-B/32 + MLP head binary classifier
   evaluated under a leave-one-generator-out (LOGO) protocol.
5. Ship a **Gradio demo** with 16.5ms per-clip inference on T4 GPU.

---

## Repository Structure

```
deeptrace/
├── src/
│   ├── data_pipeline/     # Collection, re-encoding, frame extraction, preprocessing
│   ├── model/             # CLIP feature extractor, MLP head, training loop, ONNX export
│   ├── evaluation/        # Metrics, LOGO protocol, cross-manipulation analysis
│   └── demo/              # Gradio interface + inference pipeline
├── data/
│   ├── manifests/         # DeepTrace-GV CSV manifest (12-field metadata, 246 clips)
│   ├── results/           # Evaluation results JSON files
│   ├── scripts/           # Dataset collection and slicing scripts
│   └── difficulty_annotation/ # Post-Phase 3 difficulty annotation tooling
├── models/
│   ├── checkpoints/       # 6 trained model variants (.pth)
│   └── onnx/              # 6 ONNX exports + speed benchmark
├── notebooks/             # DeepTrace_Training.ipynb - full Colab training notebook
├── configs/               # Hyperparameters and paths
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/tavishh/Deeptrace_AI_Video_Detection.git
cd CS5330_SU26_DeepTrace_AI_Detection
pip install -r requirements.txt
```

Requires ffmpeg on PATH:
```bash
brew install ffmpeg   # macOS
```

GPU environment: Google Colab Pro or Kaggle (NVIDIA T4 recommended).

---

## Dataset

### DeepTrace-GV

246 labeled video clips across four categories:

| Generator | Clips | Source |
|---|---|---|
| Sora | 33 | OpenAI official YouTube |
| Veo | 41 | Google DeepMind official YouTube |
| Kling | 42 | Official YouTube + community X posts (@KlingAI) |
| Real | 130 | Kinetics-400 via Kaggle |

All clips re-encoded to H.264/CRF23/720p. 12 metadata fields per clip including
generator label, scene type, motion level, has_faces, and 70/15/15 train/val/test
splits stratified by generator.

**Download: https://www.kaggle.com/datasets/tavishh/deeptrace-gv/**

The manifest CSV is at `data/manifests/staging_manifest.csv`.

### Celeb-DF v2

Required for cross-manipulation evaluation and faceswap/joint training.
Download using gdown:

```python
import gdown, zipfile, os
gdown.download(
    "https://drive.google.com/uc?id=1xCjVCEEPY78SpxlOeqfpDcBpb0lqXger",
    "data/celebdf/Celeb-DF-v2.zip"
)
with zipfile.ZipFile("data/celebdf/Celeb-DF-v2.zip", "r") as z:
    z.extractall("data/celebdf/")
os.remove("data/celebdf/Celeb-DF-v2.zip")
```

The `--cross-manipulation` flag in `evaluate.py` and faceswap/joint
training modes require Celeb-DF v2 at `data/celebdf/`. DeepTrace-GV
evaluation and the Gradio demo work without it.

---

## Data Pipeline

```bash
# 1. Re-encode all clips to H.264/CRF23/720p (run before anything else)
python src/data_pipeline/reencode.py --input data/raw --output data/reencoded

# 2. Extract frames at 1 FPS
python src/data_pipeline/extract_frames.py --input data/reencoded --output data/frames

# 3. Annotate clips (scene_type, motion_level, has_faces)
python src/data_pipeline/annotate_ai_clips.py --manifest data/manifests/staging_manifest.csv

# 4. Assign train/val/test splits
python src/data_pipeline/assign_splits.py --manifest data/manifests/staging_manifest.csv
```

---

## Training

Three variants trained under identical conditions:

```bash
# AI-video only
python src/model/train.py --mode ai_video \
    --manifest data/manifests/staging_manifest.csv \
    --deeptrace-root data/reencoded \
    --celebdf-root data/celebdf \
    --checkpoint-dir models/checkpoints

# Face-swap only
python src/model/train.py --mode faceswap \
    --manifest data/manifests/staging_manifest.csv \
    --deeptrace-root data/reencoded \
    --celebdf-root data/celebdf \
    --checkpoint-dir models/checkpoints

# Joint (primary experiment)
python src/model/train.py --mode joint \
    --manifest data/manifests/staging_manifest.csv \
    --deeptrace-root data/reencoded \
    --celebdf-root data/celebdf \
    --checkpoint-dir models/checkpoints

# LOGO evaluation rounds
python src/model/train.py --mode joint --logo-held-out sora ...
python src/model/train.py --mode joint --logo-held-out kling ...
python src/model/train.py --mode joint --logo-held-out veo ...
```

Full Colab training notebook: `notebooks/DeepTrace_Training.ipynb`

---

## Evaluation

```bash
# Cross-manipulation evaluation (requires Celeb-DF v2)
python src/evaluation/evaluate.py \
    --model deeptrace \
    --cross-manipulation \
    --checkpoint-dir models/checkpoints \
    --manifest data/manifests/staging_manifest.csv \
    --deeptrace-root data/reencoded \
    --celebdf-root data/celebdf

# LOGO evaluation
python src/evaluation/logo_eval.py \
    --model deeptrace \
    --all-rounds \
    --checkpoint-dir models/checkpoints

# Failure analysis
python src/evaluation/evaluate.py \
    --model deeptrace \
    --failure-analysis \
    --checkpoint models/checkpoints/deeptrace_joint_full_best.pth
```

---

## Results

### Training Results

| Variant | Val AUC |
|---|---|
| AI-video only | 0.9941 |
| Face-swap only | 0.8337 |
| Joint | 0.8957 |

### Cross-Manipulation Results (DeepTrace)

| Variant | DeepTrace-GV AUC | Celeb-DF v2 AUC | Gap |
|---|---|---|---|
| AI-video only | 0.9708 | 0.4966 | 0.4741 |
| Face-swap only | 0.7222 | 0.8090 | 0.0868 |
| Joint | 1.0000 | 0.8210 | - |

**Key finding:** Training on AI-generated video only produces near-random
performance on face-swap detection (AUC 0.50). Joint training recovers
performance on both manipulation types simultaneously.

### Cross-Manipulation Results (CLIP Linear Probe Baseline)

| Experiment | AUC |
|---|---|
| In-dist AI-generated | 1.0000 |
| Cross: train AI, test face-swap | 0.4370 |
| Cross: train face-swap, test AI | 0.5409 |
| Joint test DeepTrace-GV | 0.9766 |
| Joint test Celeb-DF v2 | 0.9536 |

### LOGO Evaluation (held-out generator test AUC)

| Held-out Generator | AUC | Accuracy |
|---|---|---|
| Sora | 1.0000 | 100.00% |
| Kling | 0.9850 | 88.46% |
| Veo | 0.9386 | 88.00% |
| **Mean** | **0.9745 ± 0.0261** | |

### Speed Benchmark (T4 GPU)

| Metric | Value |
|---|---|
| Mean latency | 16.5ms per clip |
| Throughput | 303 FPS |
| Sub-second target | PASS |

---

## Testing

```bash
python -m pytest tests/ -v
```

26 tests covering data pipeline, model modules, and evaluation metrics.

---

## Demo

The ONNX model files are already in `models/onnx/`. Run:

```bash
python src/demo/app.py
```

Opens at `http://localhost:7860`. Upload a video to receive:

- Real / AI-generated verdict with confidence score
- Configurable detection threshold (default 0.5)
- Frame-level highlights showing per-frame fake probabilities
- End-to-end latency and FPS counter

Processing pipeline:
```
upload -> H.264/CRF23/720p re-encode -> 1 FPS frame extraction
       -> CLIP ViT-B/32 features -> ONNX MLP head -> mean aggregation
       -> clip verdict
```

Performance includes re-encoding, frame extraction, CLIP, and ONNX inference.
The CLIP and ONNX models are loaded once and reused between requests.

---

## References

[1] Rossler et al., FaceForensics++, ICCV 2019
[2] Ojha et al., Towards Universal Fake Image Detection, CVPR 2023
[3] Li et al., Celeb-DF, CVPR 2020
[4] Khan et al., CLIP-based Deepfake Detection, ECCV 2024
[5] Chandra et al., Deepfake-Eval-2024, arXiv 2025
[6] Ni et al., GenVidBench, AAAI 2026
[7] Kundu et al., UNITE, arXiv 2024

---

## Citation

```
@misc{deeptrace2026,
  title={DeepTrace: Deepfake and AI-Generated Video Detection
         with Cross-Manipulation Generalization Analysis},
  author={Hookoom, Tavish and Li, Zihao and Huang, Jiajun and Zeng, Xijia},
  year={2026}
}
```
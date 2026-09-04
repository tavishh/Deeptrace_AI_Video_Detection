# DeepTrace-GV Dataset Datasheet

Following the format of Gebru et al., "Datasheets for Datasets," CACM 2021.

---

## Motivation

**For what purpose was the dataset created?**
DeepTrace-GV was created to benchmark deepfake detectors on modern AI video
generators (Sora, Kling, Veo). Existing benchmarks (FaceForensics++, DFDC,
Celeb-DF) cover only face-swap manipulation and do not include whole-scene
generation from diffusion-based video models.

**Who created the dataset and on behalf of which entity?**
Tavish Hookoom, Zihao Li, Jiajun Huang, Xijia Zeng — Northeastern University,
CS 5330 Pattern Recognition and Computer Vision, 2026.

---

## Composition

**What do the instances represent?**
Short video clips (5-30 seconds) labeled as real or AI-generated.

**How many instances are there?**
Approximately 246 clips:
- ~116 AI-generated (Sora: 33, Kling: 42, Veo: 41 - all community sourced + demo outputs)
- ~130 real (Kinetics-400 subset)

**What data does each instance consist of?**
Video clip + 12 metadata fields: clip_id, label, generator, scene_type,
motion_level, has_faces, resolution, duration_sec, source_url,
collection_date, split, difficulty.

---

## Collection Process

**How was data collected?**
- Official AI-generated clips: downloaded via yt-dlp from verified official
  generator channels (OpenAI Sora, Kuaishou Kling, Google DeepMind Veo).
- Community clips: manually collected from Reddit, X, and YouTube Shorts.
  Only clips with explicit generator attribution from the original uploader
  or a visible platform watermark are included.
- Real clips: sampled from the Kinetics-400 public dataset.

**Known selection biases:**
- Official demo clips represent curated, high-quality showcase content and
  may not reflect typical real-world generator outputs.
- Community clips mitigate this bias but are limited in quantity (~20-30
  per generator).
- The dataset will be expanded in future work to include more diverse outputs.

---

## Preprocessing

All clips are re-encoded to H.264 CRF 23 at 720p using ffmpeg before any
further processing, to eliminate compression-based leakage between sources.

---

## Annotations

**Binary labels (real/fake):** Derived from known generation source.
No ambiguity exists since all AI-generated clips come from verified sources.

**Difficulty annotation:**
Each of the four team members will independently rate a 60-clip subset as
easy/medium/hard based on whether the clip fools a naive human viewer.
Inter-rater agreement will be reported as Cohen's kappa. Annotation is
scheduled for completion after Phase 3 submission.

---

## Ethical Considerations

- All collected clips are from publicly available sources.
- No personally identifiable information is collected beyond what appears
  in the original public videos.
- The dataset is intended for research purposes only.
- The dataset explicitly acknowledges its limitations and should not be
  treated as a comprehensive coverage of all AI-generated content.

---

## Distribution

Dataset available at: https://www.kaggle.com/datasets/tavishh/deeptrace-gv/
License: MIT. Video files hosted on Kaggle. Manifest CSV and download
scripts are in the GitHub repository.

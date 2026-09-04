# Difficulty Annotation Workflow

This document describes how to run the DeepTrace-GV difficulty annotation
process. This is a dataset quality step (Cohen's kappa) done at the end of
the project, after Phase 3 submission.

**Rubric:** Rate each clip as `easy`, `medium`, or `hard` based on whether
it would fool a naive viewer with no AI-detection training. Rate
independently - do not discuss with teammates before submitting your sheet.

---

## Step 1 - Select the 60-clip subset (Tavish only, run once)

```bash
python3 src/data_pipeline/select_difficulty_subset.py \
    --manifest data/manifests/staging_manifest.csv \
    --output-dir data/difficulty_annotation \
    --n 60
```

This creates one identical blank rating sheet per team member, plus a
reference sheet listing the selected clips:

```
data/difficulty_annotation/
├── difficulty_ratings_tavish.csv
├── difficulty_ratings_zihao.csv
├── difficulty_ratings_jiajun.csv
├── difficulty_ratings_xijia.csv
└── difficulty_subset_reference.csv
```

Upload the `data/difficulty_annotation/` folder to the shared Drive
folder so everyone can download their own sheet.

---

## Step 2 - Each team member rates their assigned clips

Each person:

1. Downloads `data/reencoded/` from Drive (or just the generator
   subfolders they need - `sora/`, `kling/`, `veo/`, `real/`)
2. Downloads their own `difficulty_ratings_<name>.csv` from
   `data/difficulty_annotation/`
3. Places both in the same relative structure as the repo, then runs:

```bash
python3 rate_difficulty.py --name <your_name> --clips-dir data/reencoded
```

Replace `<your_name>` with one of: `tavish`, `zihao`, `jiajun`, `xijia`.

**What the script does:**
- Opens each unrated clip automatically in your default video player
- Prompts: `(e)asy / (m)edium / (h)ard / (s)kip`
- Saves your progress after every single clip
- Can be stopped (Ctrl+C) and resumed anytime - it skips clips you've
  already rated

When finished, upload your completed `difficulty_ratings_<name>.csv`
back to the shared Drive folder, replacing the blank version.

---

## Step 3 - Compute agreement and merge results (Tavish only, run once)

Once all four sheets are filled in and downloaded back into
`data/difficulty_annotation/`:

```bash
python3 src/data_pipeline/compute_kappa.py \
    --ratings-dir data/difficulty_annotation \
    --manifest data/manifests/staging_manifest.csv
```

This will:
- Compute pairwise Cohen's kappa between every pair of raters
- Print a Landis & Koch interpretation (slight / fair / moderate /
  substantial / almost perfect agreement)
- Merge the majority-vote difficulty label into
  `data/manifests/staging_manifest.csv`

Report the resulting kappa value and interpretation in the IEEE report's
Dataset section as the dataset quality metric.

---

## Timeline

| Step | Owner | When |
|---|---|---|
| Select subset | Tavish | July 8 (after Phase 3) |
| Rate clips | All four | July 8-10 |
| Compute kappa | Tavish | July 11 |

This needs to land before report writing begins in earnest (July 8-18)
since the kappa result and its interpretation belong in the Dataset and
Discussion sections of the final paper.
# OmniDocBench Evaluation for GLM-OCR — Design

**Date:** 2026-07-09
**Status:** Approved

## Goal

Benchmark the GLM-OCR model (as used in this repo) against OmniDocBench's
**end-to-end** track on a ~100-page subset, producing scores comparable in
methodology to the public leaderboard (with one documented exception: the CDM
formula metric is skipped).

## Decisions

- **Track:** End-to-end (page image → markdown), not component-level.
- **Subset:** ~100 pages, stratified across OmniDocBench's document types
  (not simply the first 100), sampled with a fixed seed for reproducibility.
- **Inference:** Official GLM-OCR SDK (PP-DocLayoutV3 layout detection +
  GLM-OCR 0.9B recognition) running locally on this Mac. The raw model alone
  cannot produce full-page markdown, so the SDK pipeline is required for a
  fair end-to-end comparison.
- **Eval environment:** Local Python 3.10 env managed by `uv` inside the
  cloned OmniDocBench repo. No Docker. CDM metric disabled, so TeX Live /
  ImageMagick are not needed; display formulas are scored by edit distance.

## Layout

```
benchmark/
  OmniDocBench/        # cloned github.com/opendatalab/OmniDocBench (gitignored)
  data/                # dataset images + OmniDocBench.json from HF (gitignored)
  predictions/         # one <image_name>.md per sampled page (gitignored)
  predict.py           # inference script (committed)
  sample_pages.py      # or inline in predict.py: stratified 100-page sample (committed)
  configs/end2end.yaml # eval config, committed
```

`.gitignore` gains entries for the cloned repo, data, and predictions.

## Components

### 1. Dataset download
Pull the OmniDocBench dataset (images + `OmniDocBench.json` ground truth)
from HuggingFace (`opendatalab/OmniDocBench`) via `huggingface_hub` into
`benchmark/data/`.

### 2. Page sampling
Read `OmniDocBench.json`, group pages by `data_source` (document type), and
take a proportional stratified sample of ~100 pages with a fixed random seed.
Write the sampled image list to `benchmark/data/sample_list.txt` and a
filtered ground-truth JSON `benchmark/data/OmniDocBench_sample.json`
containing only the sampled pages (so the eval doesn't count the other ~1,550
pages as missing predictions).

### 3. Prediction generation (`benchmark/predict.py`)
- Runs in a dedicated env with the GLM-OCR SDK installed (separate from the
  main project env, since the SDK pins its own deps incl. PaddlePaddle).
- For each sampled image without an existing `.md` in
  `benchmark/predictions/`: run the SDK pipeline, write
  `predictions/<image_stem>.md`.
- Resumable: existing non-empty `.md` files are skipped.
- Per-page failures are logged and produce an empty `.md` (counts against the
  score honestly) rather than aborting the run.

### 4. Evaluation
- `benchmark/OmniDocBench/` cloned, with `uv`-managed Python 3.10 venv,
  `pip install -e .`.
- `benchmark/configs/end2end.yaml`:
  - metrics: text_block `[Edit_dist, BLEU, METEOR]`, display_formula
    `[Edit_dist]` (no CDM), table `[TEDS, Edit_dist]`, reading_order
    `[Edit_dist]`
  - ground truth: `benchmark/data/OmniDocBench_sample.json`
  - predictions: `benchmark/predictions/`
  - match_method: `quick_match`
- Run: `python pdf_validation.py --config ../configs/end2end.yaml` from the
  OmniDocBench repo; results land in its `result/` directory.

### 5. Docs
`README.md` gets a "Benchmark" section: the three commands (download/sample,
predict, evaluate) and a note that CDM is disabled and the score is over a
100-page stratified subset.

## Error handling

- SDK install failure on macOS (PaddlePaddle) is the main risk; if it can't
  be installed, fall back is a scope change to discuss with the user (raw
  model whole-page prompt or hosted API), not something to silently swap in.
- Predict script: log + empty `.md` on per-page failure; nonzero exit summary
  lists failed pages.
- Eval harness errors about missing predictions are prevented by the filtered
  GT JSON.

## Testing / verification

- Smoke test predict.py on 2–3 pages before the full 100-page run; visually
  check one generated `.md` against its page image.
- Verify eval end-to-end on the smoke subset (scores produced, no crashes)
  before running the full sample.
- Success criterion: an end-to-end results table (edit distances per category
  + overall) for the 100-page subset.

## Out of scope

- CDM formula metric, Docker, full 1,651-page run, component-level tracks,
  layout/formula detection tracks. All can be added later on top of this
  structure.

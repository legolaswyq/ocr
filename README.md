# glm-ocr-local

Run [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) locally.

```bash
uv run ocr.py <image_or_pdf> [task]   # task: text (default), formula, table, or a custom prompt
```

## OmniDocBench benchmark

Scores GLM-OCR (official `glmocr` SDK pipeline, served locally via mlx-vlm on
Apple Silicon) on a 100-page stratified sample of
[OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s end-to-end track.

Caveats: 100 of 1,651 pages (stratified by document type, seed 42); the CDM
formula metric is disabled (formulas scored by edit distance) and text is
scored by edit distance only (BLEU/METEOR are broken in the current
OmniDocBench dependency set). Numbers are indicative rather than directly
comparable to the public leaderboard.

### One-time setup

```bash
# dataset (~1.6 GB)
uvx --from 'huggingface_hub[cli]' hf download opendatalab/OmniDocBench \
  --repo-type dataset --local-dir benchmark/data/OmniDocBench_dataset
python3 benchmark/sample_pages.py            # pick the 100-page sample

# model server env (Metal GPU)
uv venv benchmark/.venv-mlx --python 3.12
uv pip install --python benchmark/.venv-mlx/bin/python \
  'git+https://github.com/Blaizzy/mlx-vlm.git'

# glmocr SDK env
uv venv benchmark/.venv-sdk --python 3.12
uv pip install --python benchmark/.venv-sdk/bin/python 'glmocr[layout]' \
  'git+https://github.com/huggingface/transformers.git'

# eval harness (Python 3.10)
git clone https://github.com/opendatalab/OmniDocBench.git benchmark/OmniDocBench
uv venv benchmark/OmniDocBench/.venv --python 3.10
uv pip install --python benchmark/OmniDocBench/.venv/bin/python -e benchmark/OmniDocBench
```

### Run

```bash
# terminal 1: model server
benchmark/.venv-mlx/bin/mlx_vlm.server --trust-remote-code --port 8080

# terminal 2: predictions (resumable), then eval
python3 benchmark/predict.py
cd benchmark/OmniDocBench && .venv/bin/python pdf_validation.py --config ../configs/end2end.yaml
```

Results land in `benchmark/OmniDocBench/result/`.

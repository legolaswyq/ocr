# OmniDocBench Evaluation of GLM-OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score GLM-OCR (official SDK pipeline, running locally on Apple Silicon) on a 100-page stratified subset of OmniDocBench's end-to-end track.

**Architecture:** Three isolated Python environments under `benchmark/`: (1) an `mlx-vlm` server that runs the GLM-OCR model on the Metal GPU behind an OpenAI-compatible HTTP API, (2) the `glmocr` SDK env that does layout detection (PP-DocLayoutV3 safetensors via transformers) + region OCR against that server and emits per-page markdown, (3) a Python 3.10 env with the cloned OmniDocBench repo that scores the markdown against ground truth. A sampler script picks a seeded, stratified 100-page subset and writes a filtered ground-truth JSON so the eval only scores sampled pages.

**Tech Stack:** uv (env management), mlx-vlm (git), glmocr SDK, huggingface_hub (dataset download), OmniDocBench (Python 3.10), pytest.

## Global Constraints

- OCR model served by mlx-vlm: `mlx-community/GLM-OCR-bf16` on port `8080`, endpoint path `/chat/completions` (no `/v1` prefix).
- glmocr SDK and mlx-vlm MUST live in separate venvs (conflicting transformers pins): `benchmark/.venv-mlx` and `benchmark/.venv-sdk`, both Python 3.12.
- OmniDocBench eval env: Python 3.10, venv at `benchmark/OmniDocBench/.venv`.
- CDM metric is NOT used anywhere (no TeX Live/ImageMagick); display formulas scored by `Edit_dist` only.
- Sample: 100 pages, stratified by `page_info.page_attribute.data_source`, random seed `42`.
- Prediction files: `benchmark/predictions/<image_stem>.md`, one per sampled page; empty file on per-page failure.
- Committed files only: `benchmark/*.py`, `benchmark/glmocr_config.yaml`, `benchmark/configs/`, `tests/`, docs. Data, venvs, clones, predictions are gitignored.
- All commands below run from the repo root `/Users/walter.wang/git/ocr` unless a `cd` is shown.

---

### Task 1: Scaffolding, .gitignore, dataset download

**Files:**
- Modify: `.gitignore`
- Create: `benchmark/` directory tree
- Create: `benchmark/data/OmniDocBench_dataset/` (downloaded, gitignored)

**Interfaces:**
- Produces: dataset at `benchmark/data/OmniDocBench_dataset/OmniDocBench.json` and `benchmark/data/OmniDocBench_dataset/images/*.jpg` — consumed by Tasks 2 and 5.

- [ ] **Step 1: Add gitignore entries**

Append to `.gitignore` (create the lines only if not already present):

```gitignore
# OmniDocBench benchmark artifacts
benchmark/OmniDocBench/
benchmark/data/
benchmark/predictions/
benchmark/smoke_out/
benchmark/.venv-mlx/
benchmark/.venv-sdk/
```

- [ ] **Step 2: Create directory tree**

```bash
mkdir -p benchmark/configs benchmark/data benchmark/predictions
```

- [ ] **Step 3: Download the dataset (~1.6 GB)**

```bash
uvx --from 'huggingface_hub[cli]' hf download opendatalab/OmniDocBench \
  --repo-type dataset --local-dir benchmark/data/OmniDocBench_dataset
```

Expected: download completes; if `hf` is unavailable, fall back to `huggingface-cli download` with the same arguments.

- [ ] **Step 4: Verify layout**

```bash
python3 -c "
import json, pathlib
root = pathlib.Path('benchmark/data/OmniDocBench_dataset')
gt = json.loads((root / 'OmniDocBench.json').read_text())
print('pages:', len(gt))
print('sample image_path:', gt[0]['page_info']['image_path'])
print('sample data_source:', gt[0]['page_info']['page_attribute']['data_source'])
imgs = list((root / 'images').glob('*'))
print('images:', len(imgs))
"
```

Expected: `pages: 1651` (or close), an image_path string, a data_source string, and ~1651 images. **If key paths differ (e.g., attribute nesting), note the actual structure and adjust Task 2's accessor accordingly — this is the one place the plan allows adaptation.**

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: add benchmark scaffolding and gitignore entries"
```

---

### Task 2: Stratified page sampler

**Files:**
- Create: `benchmark/sample_pages.py`
- Test: `tests/test_sample_pages.py`

**Interfaces:**
- Produces: `stratified_sample(pages: list[dict], n: int, seed: int) -> list[dict]` (pure function); running the script produces `benchmark/data/OmniDocBench_sample.json` (filtered GT list), `benchmark/data/sample_list.txt` (one image filename per line), and `benchmark/data/sample_images/` (copies of the sampled images) — consumed by Tasks 5 and 6.

- [ ] **Step 1: Write the failing test**

`tests/test_sample_pages.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from sample_pages import stratified_sample


def make_page(name: str, source: str) -> dict:
    return {
        "page_info": {
            "image_path": f"images/{name}.jpg",
            "page_attribute": {"data_source": source},
        }
    }


def make_pages() -> list[dict]:
    pages = []
    for i in range(80):
        pages.append(make_page(f"book_{i}", "book"))
    for i in range(20):
        pages.append(make_page(f"note_{i}", "note"))
    for i in range(2):
        pages.append(make_page(f"rare_{i}", "historical_document"))
    return pages


def test_returns_exactly_n_pages():
    assert len(stratified_sample(make_pages(), 50, seed=42)) == 50


def test_every_group_represented():
    sample = stratified_sample(make_pages(), 50, seed=42)
    sources = {p["page_info"]["page_attribute"]["data_source"] for p in sample}
    assert sources == {"book", "note", "historical_document"}


def test_roughly_proportional():
    sample = stratified_sample(make_pages(), 50, seed=42)
    books = [p for p in sample if p["page_info"]["page_attribute"]["data_source"] == "book"]
    assert 35 <= len(books) <= 45  # 80/102 of 50 ≈ 39


def test_deterministic():
    a = stratified_sample(make_pages(), 50, seed=42)
    b = stratified_sample(make_pages(), 50, seed=42)
    assert [p["page_info"]["image_path"] for p in a] == [p["page_info"]["image_path"] for p in b]


def test_n_larger_than_population_returns_all():
    assert len(stratified_sample(make_pages(), 500, seed=42)) == 102
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_sample_pages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sample_pages'`

- [ ] **Step 3: Write the implementation**

`benchmark/sample_pages.py`:

```python
"""Stratified sampling of OmniDocBench pages by document type.

Usage:
    python3 benchmark/sample_pages.py [--n 100] [--seed 42]

Reads benchmark/data/OmniDocBench_dataset/OmniDocBench.json and writes:
    benchmark/data/OmniDocBench_sample.json  (filtered ground truth)
    benchmark/data/sample_list.txt           (sampled image filenames)
    benchmark/data/sample_images/            (copies of sampled images)
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def stratified_sample(pages: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional stratified sample by data_source, >=1 page per group."""
    if n >= len(pages):
        return list(pages)

    groups: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        groups[page["page_info"]["page_attribute"]["data_source"]].append(page)

    rng = random.Random(seed)
    total = len(pages)

    # Largest-remainder allocation with a floor of 1 per group.
    quotas = {src: n * len(members) / total for src, members in groups.items()}
    counts = {src: max(1, int(q)) for src, q in quotas.items()}
    while sum(counts.values()) > n:
        # Shrink the group with the largest overshoot vs quota (never below 1).
        src = max(
            (s for s in counts if counts[s] > 1),
            key=lambda s: counts[s] - quotas[s],
        )
        counts[src] -= 1
    remainders = sorted(quotas, key=lambda s: quotas[s] - counts[s], reverse=True)
    i = 0
    while sum(counts.values()) < n:
        src = remainders[i % len(remainders)]
        if counts[src] < len(groups[src]):
            counts[src] += 1
        i += 1

    sample = []
    for src in sorted(groups):
        sample.extend(rng.sample(groups[src], min(counts[src], len(groups[src]))))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path("benchmark/data")
    dataset_dir = data_dir / "OmniDocBench_dataset"
    pages = json.loads((dataset_dir / "OmniDocBench.json").read_text())

    sample = stratified_sample(pages, args.n, args.seed)

    (data_dir / "OmniDocBench_sample.json").write_text(json.dumps(sample, ensure_ascii=False))

    image_dir = data_dir / "sample_images"
    image_dir.mkdir(exist_ok=True)
    names = []
    for page in sample:
        name = Path(page["page_info"]["image_path"]).name
        names.append(name)
        src = dataset_dir / "images" / name
        dst = image_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
    (data_dir / "sample_list.txt").write_text("\n".join(names) + "\n")

    by_source = defaultdict(int)
    for page in sample:
        by_source[page["page_info"]["page_attribute"]["data_source"]] += 1
    print(f"sampled {len(sample)} pages:")
    for src in sorted(by_source):
        print(f"  {src}: {by_source[src]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest tests/test_sample_pages.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the sampler for real**

```bash
python3 benchmark/sample_pages.py --n 100 --seed 42
ls benchmark/data/sample_images | wc -l
```

Expected: printed per-source breakdown summing to 100; `100` images copied. If Task 1 Step 4 revealed different JSON key paths, fix the accessors here and in the tests to match reality before running.

- [ ] **Step 6: Commit**

```bash
git add benchmark/sample_pages.py tests/test_sample_pages.py
git commit -m "feat: stratified 100-page OmniDocBench sampler"
```

---

### Task 3: mlx-vlm model server

**Files:**
- Create: `benchmark/.venv-mlx/` (gitignored)

**Interfaces:**
- Produces: OpenAI-compatible server at `http://localhost:8080/chat/completions` serving `mlx-community/GLM-OCR-bf16` — consumed by the glmocr SDK (Tasks 4–5, 7). The server must be running whenever predictions are generated.

- [ ] **Step 1: Create the server venv and install mlx-vlm from git**

GLM-OCR support is not in the mlx-vlm PyPI release yet — install from git:

```bash
uv venv benchmark/.venv-mlx --python 3.12
uv pip install --python benchmark/.venv-mlx/bin/python \
  'git+https://github.com/Blaizzy/mlx-vlm.git'
```

Expected: installs without error (pulls mlx, transformers>=5.0.0rc3, etc.).

- [ ] **Step 2: Launch the server in the background**

```bash
benchmark/.venv-mlx/bin/mlx_vlm.server --trust-remote-code --port 8080
```

(If no `mlx_vlm.server` entry point exists, use `benchmark/.venv-mlx/bin/python -m mlx_vlm.server --trust-remote-code --port 8080`.)

Run in the background; first launch downloads `mlx-community/GLM-OCR-bf16` weights (~2 GB).

- [ ] **Step 3: Health check**

```bash
curl -s http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/GLM-OCR-bf16",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    "max_tokens": 10
  }'
```

Expected: JSON response containing `"choices"`. (First request is slow — Metal shader warmup.)

No commit — this task creates only gitignored artifacts.

---

### Task 4: glmocr SDK environment and config

**Files:**
- Create: `benchmark/glmocr_config.yaml`
- Create: `benchmark/.venv-sdk/` (gitignored)

**Interfaces:**
- Consumes: running mlx-vlm server from Task 3.
- Produces: working `benchmark/.venv-sdk/bin/glmocr` CLI and `benchmark/glmocr_config.yaml` — consumed by Task 5's `predict.py` via subprocess.

- [ ] **Step 1: Create the SDK venv and install glmocr**

```bash
uv venv benchmark/.venv-sdk --python 3.12
uv pip install --python benchmark/.venv-sdk/bin/python glmocr
uv pip install --python benchmark/.venv-sdk/bin/python \
  'git+https://github.com/huggingface/transformers.git'
```

Expected: installs cleanly. (transformers-from-git is required by the SDK for PP-DocLayoutV3 safetensors loading.)

- [ ] **Step 2: Write the SDK config**

`benchmark/glmocr_config.yaml`:

```yaml
pipeline:
  maas:
    enabled: false

  ocr_api:
    api_host: localhost
    api_port: 8080
    model: mlx-community/GLM-OCR-bf16  # required for mlx-vlm
    api_path: /chat/completions        # mlx-vlm has no /v1 prefix
    request_timeout: 300

  # Modest parallelism — a single local Metal GPU serves the requests.
  max_workers: 8

  result_formatter:
    output_format: markdown
```

If the SDK rejects a partial config (does not merge with defaults), copy its full default `config.yaml` from the installed package and apply these same edits.

- [ ] **Step 3: Smoke test on the repo's test image**

With the Task 3 server running:

```bash
benchmark/.venv-sdk/bin/glmocr parse test_image.png \
  --config benchmark/glmocr_config.yaml --output benchmark/smoke_out/
find benchmark/smoke_out -name '*.md' -exec cat {} +
```

Expected: first run downloads the PP-DocLayoutV3 layout model, then produces a `.md` file whose text matches the content of `test_image.png`. Read the markdown and sanity-check it by eye against the image.

- [ ] **Step 4: Commit**

```bash
git add benchmark/glmocr_config.yaml
git commit -m "feat: glmocr SDK config for local mlx-vlm serving"
```

---

### Task 5: Prediction script

**Files:**
- Create: `benchmark/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `benchmark/data/sample_images/` (Task 2), `glmocr` CLI + config (Task 4), running server (Task 3).
- Produces: `pending_images(image_dir: Path, pred_dir: Path) -> list[Path]` and `collect_markdown(out_dir: Path) -> dict[str, Path]` (pure helpers); running the script fills `benchmark/predictions/<stem>.md` — consumed by Task 6's eval.

- [ ] **Step 1: Write the failing tests**

`tests/test_predict.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from predict import collect_markdown, pending_images


def test_pending_skips_nonempty_md(tmp_path):
    images = tmp_path / "images"
    preds = tmp_path / "preds"
    images.mkdir()
    preds.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    (images / "b.jpg").write_bytes(b"x")
    (images / "c.png").write_bytes(b"x")
    (preds / "a.md").write_text("done")
    (preds / "b.md").write_text("")  # empty = failed earlier, retry

    names = [p.name for p in pending_images(images, preds)]
    assert names == ["b.jpg", "c.png"]


def test_pending_ignores_non_images(tmp_path):
    images = tmp_path / "images"
    preds = tmp_path / "preds"
    images.mkdir()
    preds.mkdir()
    (images / "notes.txt").write_text("x")

    assert pending_images(images, preds) == []


def test_collect_markdown_finds_nested_md(tmp_path):
    (tmp_path / "page_a").mkdir()
    (tmp_path / "page_a" / "page_a.md").write_text("A")
    (tmp_path / "page_b").mkdir()
    (tmp_path / "page_b" / "page_b.md").write_text("B")

    found = collect_markdown(tmp_path)
    assert set(found) == {"page_a", "page_b"}
    assert found["page_a"].read_text() == "A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict'`

- [ ] **Step 3: Write the implementation**

`benchmark/predict.py`:

```python
"""Generate OmniDocBench end2end predictions using the glmocr CLI.

Usage (requires the mlx-vlm server from Task 3 to be running):
    python3 benchmark/predict.py [--limit N]

Resumable: pages with an existing non-empty .md in benchmark/predictions
are skipped. Failed pages get an empty .md (retried on the next run) and
are listed at the end; exit code is 1 if any page failed.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GLMOCR = Path("benchmark/.venv-sdk/bin/glmocr")
CONFIG = Path("benchmark/glmocr_config.yaml")


def pending_images(image_dir: Path, pred_dir: Path) -> list[Path]:
    done = {p.stem for p in pred_dir.glob("*.md") if p.stat().st_size > 0}
    return [
        p
        for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem not in done
    ]


def collect_markdown(out_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in out_dir.rglob("*.md")}


def run_batch(images: list[Path], pred_dir: Path) -> list[str]:
    """Parse `images` with one glmocr invocation; return stems that failed."""
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        for img in images:
            (in_dir / img.name).symlink_to(img.resolve())

        proc = subprocess.run(
            [str(GLMOCR), "parse", str(in_dir), "--config", str(CONFIG), "--output", str(out_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout[-2000:], file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)

        produced = collect_markdown(out_dir)
        failed = []
        for img in images:
            md = produced.get(img.stem)
            if md is not None and md.stat().st_size > 0:
                shutil.copy2(md, pred_dir / f"{img.stem}.md")
            else:
                (pred_dir / f"{img.stem}.md").write_text("")
                failed.append(img.stem)
        return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("benchmark/data/sample_images"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/predictions"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    todo = pending_images(args.images, args.out)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(todo)} pages to process")

    failed: list[str] = []
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        failed.extend(run_batch(batch, args.out))
        print(f"progress: {min(start + args.batch_size, len(todo))}/{len(todo)}")

    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("all pages done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest tests/test_predict.py -v`
Expected: 3 passed

- [ ] **Step 5: Smoke run on 3 pages**

With the Task 3 server running:

```bash
python3 benchmark/predict.py --limit 3
ls -la benchmark/predictions/
```

Expected: `3 pages to process`, then 3 non-empty `.md` files. Open one and eyeball it against its source image in `benchmark/data/sample_images/`. If the glmocr CLI's directory-output layout differs from `<stem>/<stem>.md`, adjust `collect_markdown` (it already searches recursively, so only a naming mismatch would matter).

- [ ] **Step 6: Commit**

```bash
git add benchmark/predict.py tests/test_predict.py
git commit -m "feat: resumable OmniDocBench prediction script"
```

---

### Task 6: OmniDocBench eval environment and config

**Files:**
- Create: `benchmark/OmniDocBench/` (cloned, gitignored)
- Create: `benchmark/configs/end2end.yaml`

**Interfaces:**
- Consumes: `benchmark/data/OmniDocBench_sample.json` (Task 2), `benchmark/predictions/` (Task 5).
- Produces: runnable eval — `cd benchmark/OmniDocBench && .venv/bin/python pdf_validation.py --config ../configs/end2end.yaml`; results in `benchmark/OmniDocBench/result/`.

- [ ] **Step 1: Clone and install OmniDocBench (Python 3.10)**

```bash
git clone https://github.com/opendatalab/OmniDocBench.git benchmark/OmniDocBench
uv venv benchmark/OmniDocBench/.venv --python 3.10
uv pip install --python benchmark/OmniDocBench/.venv/bin/python -e benchmark/OmniDocBench
```

Expected: installs cleanly. If a dependency fails to build on macOS, report which one before attempting workarounds.

- [ ] **Step 2: Validate the harness on its own demo data (no CDM)**

Create a throwaway config `benchmark/OmniDocBench/configs/demo_nocdm.yaml` by copying `benchmark/OmniDocBench/configs/end2end.yaml` and deleting the `- CDM` line (and its `cdm_workers` line) under `display_formula`. Then:

```bash
cd benchmark/OmniDocBench
.venv/bin/python pdf_validation.py --config configs/demo_nocdm.yaml
```

Expected: runs to completion on `demo_data/`, printing metric tables and writing to `result/`. This proves the env works independent of our predictions.

- [ ] **Step 3: Write our eval config**

`benchmark/configs/end2end.yaml` (paths are relative to `benchmark/OmniDocBench/`, where the eval runs):

```yaml
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
      - BLEU
      - METEOR
    display_formula:
      metric:
      - Edit_dist
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 4
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ../data/OmniDocBench_sample.json
    prediction:
      data_path: ../predictions
    match_method: quick_match
    match_workers: 4
```

If the harness errors on `BLEU`/`METEOR` for `text_block`, drop them and keep `Edit_dist` (note it in the README task).

- [ ] **Step 4: Run the eval on the smoke predictions**

With the 3 smoke `.md` files from Task 5 present:

```bash
cd benchmark/OmniDocBench
.venv/bin/python pdf_validation.py --config ../configs/end2end.yaml
```

Expected: completes and prints metric tables. Unpredicted sampled pages may score as empty/missing — that's fine for the smoke check; we only need "config parses, GT loads, predictions match by filename, metrics print".

- [ ] **Step 5: Commit**

```bash
git add benchmark/configs/end2end.yaml
git commit -m "feat: OmniDocBench end2end eval config (no CDM)"
```

---

### Task 7: Full 100-page run, final eval, README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: final scores in `benchmark/OmniDocBench/result/`, documented workflow in `README.md`.

- [ ] **Step 1: Run the full prediction pass**

With the mlx-vlm server running:

```bash
python3 benchmark/predict.py
```

Expected: processes the remaining ~97 pages. This can take multiple hours on MPS — run it in the background and check progress periodically. If interrupted, re-running resumes. If pages fail, re-run once; report persistently failing pages rather than looping.

```bash
ls benchmark/predictions/*.md | wc -l          # expect 100
find benchmark/predictions -name '*.md' -empty  # expect no output
```

- [ ] **Step 2: Run the final eval**

```bash
cd benchmark/OmniDocBench
.venv/bin/python pdf_validation.py --config ../configs/end2end.yaml
```

Expected: full metric tables (overall + per-category edit distances, TEDS for tables). Save/copy the printed summary — it goes in the final report to the user.

- [ ] **Step 3: Write the README section**

Append to `README.md`:

```markdown
## OmniDocBench benchmark

Scores GLM-OCR (official `glmocr` SDK pipeline, served locally via mlx-vlm on
Apple Silicon) on a 100-page stratified sample of
[OmniDocBench](https://github.com/opendatalab/OmniDocBench)'s end-to-end track.

Caveats: 100 of 1,651 pages (stratified by document type, seed 42), and the
CDM formula metric is disabled (formulas scored by edit distance), so numbers
are indicative rather than directly comparable to the public leaderboard.

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
uv pip install --python benchmark/.venv-sdk/bin/python glmocr \
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
```

Adjust the caveats line if Task 6 dropped BLEU/METEOR.

- [ ] **Step 4: Verify docs match reality**

Re-check every command in the README section against what was actually run (entry point names, paths). Fix drift.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: OmniDocBench benchmark workflow and caveats"
```

- [ ] **Step 6: Report results**

Present the final metric table to the user: overall edit distance, per-category (text / formula / table / reading order) scores, page counts, and any failed pages, alongside the published GLM-OCR leaderboard numbers for context.

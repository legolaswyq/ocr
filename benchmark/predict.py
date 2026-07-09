"""Generate OmniDocBench end2end predictions using the glmocr CLI.

Usage (requires the mlx-vlm server to be running on :8080):
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
        print(f"progress: {min(start + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
    print("all pages done")


if __name__ == "__main__":
    main()

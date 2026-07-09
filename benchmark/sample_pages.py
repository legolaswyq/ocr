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

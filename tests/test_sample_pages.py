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

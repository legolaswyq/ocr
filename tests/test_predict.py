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

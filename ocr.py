"""Run GLM-OCR on an image or PDF.

Usage:
    uv run ocr.py <image_or_pdf> [task]

Tasks: text (default), formula, table — or any custom prompt string.
PDFs are rendered page by page; each page is OCR'd separately.
"""

import sys
import tempfile
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "zai-org/GLM-OCR"

TASK_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
    )
    return processor, model


def ocr_image(processor, model, image_path: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    return processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def pdf_to_page_images(pdf_path: Path, out_dir: Path, scale: float = 2.0) -> list[Path]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pages = []
        for i, page in enumerate(pdf):
            image = page.render(scale=scale).to_pil()
            path = out_dir / f"page_{i + 1}.png"
            image.save(path)
            pages.append(path)
        return pages
    finally:
        pdf.close()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    input_path = Path(sys.argv[1])
    task = sys.argv[2] if len(sys.argv) > 2 else "text"
    prompt = TASK_PROMPTS.get(task, task)

    processor, model = load_model()

    if input_path.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory() as tmp:
            pages = pdf_to_page_images(input_path, Path(tmp))
            for i, page_path in enumerate(pages, start=1):
                if len(pages) > 1:
                    print(f"\n--- Page {i}/{len(pages)} ---\n")
                print(ocr_image(processor, model, str(page_path), prompt))
    else:
        print(ocr_image(processor, model, str(input_path), prompt))


if __name__ == "__main__":
    main()

# utils.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator, Tuple

from PIL import Image
import fitz  # PyMuPDF

# основной конвейер (сохраняет результаты в /static/results)
from error_level_analysis import run_image

# ---- пути ----
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DPI рендера PDF
PDF_DPI = 300


def iter_pdf_pages(pdf_path: Path, dpi: int = PDF_DPI) -> Iterator[Tuple[int, Image.Image]]:
    """Генератор: (1‑based page_index, PIL.Image) по файлу PDF."""
    with fitz.open(str(pdf_path)) as doc:
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            yield i + 1, Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# --- shim для обратной совместимости ---
def process_pil_image(pil: Image.Image, label: str, batch: str) -> dict:
    """
    Старое имя функции — теперь просто вызывает новый конвейер run_image(...).
    Оставлено для совместимости со старыми импортами.
    """
    if pil is None:
        raise ValueError("Empty image")
    return run_image(pil.convert("RGB"), label, batch, RESULTS_DIR)

import io
import uuid
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ---- пути ----
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- настройки ----
JPEG_QUALITIES = (90, 95, 98)   # ансамбль пересохранений
VAR_KERNEL = 9                  # окно дисперсии
MIN_AREA_RATIO = 0.002          # минимальная площадь пятна (~0.2%)
MASK_MORPH = 3                  # морфология очистки
PDF_DPI = 300                   # DPI для рендера PDF (iter_pdf_pages)


# ---------- базовые утилиты ----------
def _local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var


def _ela_map(pil_img: Image.Image) -> np.ndarray:
    """Ансамблевый ELA: максимум разностей по нескольким JPEG‑качествам."""
    pil_rgb = pil_img.convert("RGB")
    maps = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=q)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(pil_rgb, resaved)

        # автоусиление динамики
        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)

        maps.append(np.array(diff.convert("L")))

    ela = np.maximum.reduce(maps)
    return ela.astype(np.uint8)


def _mask_flat_regions(ela_gray: np.ndarray) -> np.ndarray:
    """
    Маска «подозрительных» зон без семантики:
    ищем ровные (низкая вариативность) участки шумовой карты.
    """
    ela_blur = cv2.GaussianBlur(ela_gray, (3, 3), 0)

    # локальная дисперсия
    var = _local_variance(ela_blur, VAR_KERNEL)
    var = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # инверсия: низкая вариативность → высокая маска
    inv = cv2.bitwise_not(var)

    # адаптивный порог по статистике кадра
    med = np.median(inv)
    q1, q3 = np.percentile(inv, [25, 75])
    iqr = max(1.0, q3 - q1)
    thr = int(min(255, med + 1.25 * iqr))
    _, mask = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)

    # морфология
    kernel = np.ones((MASK_MORPH, MASK_MORPH), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # фильтр по площади
    h, w = mask.shape[:2]
    min_area = int(MIN_AREA_RATIO * w * h)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, thickness=-1)
    return out


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Полупрозрачная заливка маски + контур."""
    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill = np.zeros_like(overlay)
    cv2.drawContours(fill, contours, -1, (255, 140, 0), thickness=-1)   # BGR оранжевый
    overlay = cv2.addWeighted(overlay, 1.0, fill, 0.35, 0)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)
    return overlay


def _summarize(mask: np.ndarray) -> tuple[float, int]:
    total = mask.size
    pos = int((mask > 0).sum())
    pct = 100.0 * pos / max(1, total)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return round(pct, 2), len(contours)


def _verdict(p: float) -> str:
    if p >= 10.0:
        return "High (likely edited)"
    if p >= 3.0:
        return "Medium (possible edits)"
    return "Low (no clear edits)"


# ---------- публичные функции ----------
def process_pil_image(pil: Image.Image, label: str, batch: str) -> dict:
    """
    Строим ELA‑карту, выделяем «пластичные» пятна, считаем % и регионы,
    сохраняем оригинал/ELA/overlay и отдаём словарь для UI.
    """
    if pil is None:
        raise ValueError("Empty image")

    pil = pil.convert("RGB")
    src = np.array(pil)

    ela = _ela_map(pil)                                # (H, W) uint8
    ela_vis = cv2.applyColorMap(ela, cv2.COLORMAP_INFERNO)

    mask = _mask_flat_regions(ela)
    pct, regions = _summarize(mask)
    verdict = _verdict(pct)

    ovl = _overlay_mask(src, mask)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_path = RESULTS_DIR / f"{stem}_src.jpg"
    ela_path = RESULTS_DIR / f"{stem}_ela.jpg"
    ovl_path = RESULTS_DIR / f"{stem}_overlay.jpg"

    Image.fromarray(src).save(str(src_path), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis, cv2.COLOR_BGR2RGB)).save(str(ela_path), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ovl, cv2.COLOR_BGR2RGB)).save(str(ovl_path), "JPEG", quality=95)

    summary = f"{verdict} — suspicious area ≈ {pct:.2f}% across {regions} region(s)."

    return {
        "label": label,
        "original": f"/static/results/{src_path.name}",
        "ela":      f"/static/results/{ela_path.name}",
        "overlay":  f"/static/results/{ovl_path.name}",
        "boxed": "",
        "verdict": verdict,
        "regions": regions,
        "crops": [],
        "summary": summary
    }


def iter_pdf_pages(pdf_path: Path, dpi: int = PDF_DPI):
    """Генератор: страницы PDF как PIL.Image (1‑индексация)."""
    import fitz
    with fitz.open(str(pdf_path)) as doc:
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            yield i + 1, Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

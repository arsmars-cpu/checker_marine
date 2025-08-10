# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import fitz  # PyMuPDF

# ----------------- Параметры (подкручиваемые) -----------------
JPEG_QUALITIES = (90, 95, 98)   # ансамблевый ELA
VAR_KERNEL = 9                  # окно локальной дисперсии
MIN_AREA_RATIO = 0.002          # отсев слишком мелких пятен (доля от площади)
MASK_MORPH = 3                  # ядро морфологии
PDF_DPI = 200                   # DPI рендера PDF (стабильнее)
MAX_PDF_PAGES = 5               # максимум страниц на анализ
TEXT_SUPPRESS = 0.5             # сила подавления печатного текста в score-карте
SCORE_W_ELA = 0.65              # вес ELA в score
SCORE_W_NOISE = 0.35            # вес noise в score
SCORE_PERCENTILE = 95           # верхний перцентиль для маски

# Директория результатов (можно импортировать из app.py, если хочешь)
RESULTS_DIR = Path(__file__).parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- Утилиты -----------------
def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}


def local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    """Локальная дисперсия по окну k×k."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var


# ----------------- ELA / Noise / Text -----------------
def ela_map(pil_img: Image.Image) -> np.ndarray:
    """
    Ансамбль ELA (несколько качеств JPEG) -> поканальный максимум -> grayscale (uint8).
    """
    pil_rgb = pil_img.convert("RGB")
    maps = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=q, optimize=True)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(pil_rgb, resaved)

        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
        maps.append(np.array(diff.convert("L")))  # uint8

    ela = np.maximum.reduce(maps)  # uint8 0..255
    return ela


def noise_map_from_gray(gray_u8: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    """
    Карта «непохожести» локального шума (0..1): 1 — подозрительно ровно/непохоже.
    """
    var = local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var


def text_mask(pil_img: Image.Image) -> np.ndarray:
    """
    Маска печатного текста/тонких штрихов (0/255).
    """
    g = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return txt  # 0..255


# ----------------- Score-карта и маска -----------------
def fused_score(pil_img: Image.Image) -> np.ndarray:
    """
    0..1 score-карта: 0 — обычные области, 1 — сильные аномалии (ELA+noise, подавлен текст).
    """
    # ELA -> [0..1]
    ela = ela_map(pil_img).astype(np.float32)
    ela = (ela - ela.min()) / (ela.ptp() + 1e-6)

    # Noise -> [0..1]
    g = np.array(pil_img.convert("L")).astype(np.float32)
    g = (g - g.min()) / (g.ptp() + 1e-6)
    noise = noise_map_from_gray((g * 255).astype(np.uint8), k=VAR_KERNEL)

    # Слияние
    score = SCORE_W_ELA * ela + SCORE_W_NOISE * noise

    # Подавляем печатный текст (частично)
    tmask = text_mask(pil_img).astype(np.float32) / 255.0
    score = score * (1.0 - TEXT_SUPPRESS * tmask)

    # Нормализация
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score.astype(np.float32)  # 0..1


def suspicious_mask_from_score(score: np.ndarray,
                               percentile: int = SCORE_PERCENTILE) -> np.ndarray:
    """
    Бинарная маска подозрительных пикселей по верхнему перцентилю score.
    """
    thr = float(np.percentile(score, percentile))
    mask = (score >= thr).astype(np.uint8) * 255
    return mask


def clean_and_filter_mask(mask: np.ndarray,
                          min_area_ratio: float = MIN_AREA_RATIO,
                          morph: int = MASK_MORPH) -> np.ndarray:
    """
    Морфология + отсев мелких пятен по площади.
    """
    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    H, W = mask.shape[:2]
    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, thickness=-1)
    return out


# ----------------- Визуализация -----------------
def overlay_suspicious(base_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Полупрозрачная заливка по подозрительным зонам + красный контур.
    """
    overlay = base_bgr.copy()
    color = (0, 140, 255)  # BGR заливка
    alpha = 0.35

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fill = np.zeros_like(overlay)
    cv2.drawContours(fill, contours, -1, color, thickness=-1)
    overlay = cv2.addWeighted(overlay, 1.0, fill, alpha, 0)

    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)
    return overlay


def extract_regions(mask: np.ndarray, score_map: np.ndarray, max_regions: int = 6, pad: int = 8) -> List[Dict]:
    """
    Формирует списки топ-зон [{x,y,w,h,score}], отсортированные по среднему score внутри бокса.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs = []
    H, W = mask.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # подушка вокруг бокса
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)

        roi = score_map[y0:y1, x0:x1]
        sc = float(np.mean(roi)) if roi.size else 0.0
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})

    regs.sort(key=lambda r: r["score"], reverse=True)
    return regs[:max_regions]


def draw_boxes_on_overlay(overlay_bgr: np.ndarray, regions: List[Dict]) -> np.ndarray:
    out = overlay_bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(out, f"#{idx} {r['score']:.2f}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def save_crops(src_rgb: np.ndarray, regions: List[Dict], stem: str, out_dir: Path = RESULTS_DIR) -> List[Dict]:
    """
    Сохраняет кропы зон в JPEG. Возвращает список словарей вида:
      {"index": i, "score": float, "url": "/static/results/..", "box": {...}}
    """
    out = []
    H, W, _ = src_rgb.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = src_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        name = f"{stem}_crop_{idx}.jpg"
        Image.fromarray(crop).save(str(out_dir / name), "JPEG", quality=95)
        out.append({
            "index": idx,
            "score": r["score"],
            "url": f"/static/results/{name}",
            "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        })
    return out


# ----------------- PDF: постраничный рендер -----------------
def iter_pdf_pages(pdf_path: Path,
                   dpi: int = PDF_DPI,
                   max_pages: int = MAX_PDF_PAGES):
    """
    Памяти-бережный генератор: отдаёт (page_index_1based, PIL.Image) по одной странице.
    """
    with fitz.open(str(pdf_path)) as doc:
        pages = min(len(doc), max_pages)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i + 1, img


# ----------------- High-level: процессинг одной картинки -----------------
def process_pil_image(pil: Image.Image, label: str, batch: str) -> Dict:
    """
    Полный конвейер для одной страницы/картинки:
      ELA -> score -> mask -> regions -> overlay/boxed/crops -> json-результат
    """
    # ELA и score-map
    ela = ela_map(pil)
    score = fused_score(pil)  # 0..1

    # Маска + очистка
    mask = suspicious_mask_from_score(score, percentile=SCORE_PERCENTILE)
    mask = clean_and_filter_mask(mask, min_area_ratio=MIN_AREA_RATIO, morph=MASK_MORPH)

    # Топ-зоны
    regions = extract_regions(mask, score, max_regions=6, pad=8)

    # Картинки
    src_rgb = np.array(pil.convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    ela_vis_bgr = cv2.applyColorMap(ela.astype(np.uint8), cv2.COLORMAP_INFERNO)

    overlay_bgr = overlay_suspicious(src_bgr, mask)
    boxed_bgr = draw_boxes_on_overlay(overlay_bgr, regions)

    # Имена файлов
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"

    # Сохранение
    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ela_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ovl_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / boxed_name), "JPEG", quality=95)

    # Кропы
    crops_meta = save_crops(src_rgb, regions, stem, RESULTS_DIR)

    # Сводка
    suspicious_pct = round(100.0 * (mask > 0).sum() / max(1, mask.size), 1)
    verdict = "Review recommended" if regions else "No issues found"

    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela":      f"/static/results/{ela_name}",
        "overlay":  f"/static/results/{ovl_name}",
        "boxed":    f"/static/results/{boxed_name}",
        "verdict": verdict,
        "suspicious_percent": suspicious_pct,
        "regions": len(regions),
        "crops": crops_meta,
        "summary": f"Top {len(regions)} region(s) highlighted. Use thumbnails to review. Suspicious area ≈ {suspicious_pct:.1f}%."
    }

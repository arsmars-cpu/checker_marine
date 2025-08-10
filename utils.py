# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import fitz  # PyMuPDF

# ----------------- Константы / параметры -----------------
JPEG_QUALITIES      = (90, 95, 98)   # ансамблевый ELA
VAR_KERNEL          = 9              # окно локальной дисперсии
MIN_AREA_RATIO      = 0.002          # отсев мелких пятен (доля от площади)
MASK_MORPH          = 3              # ядро морфологии
PDF_DPI             = 200            # DPI рендера PDF
MAX_PDF_PAGES       = 5              # максимум страниц на анализ

# Score = 0.65*ELA + 0.35*Noise, минус 50% печатного текста
SCORE_W_ELA         = 0.65
SCORE_W_NOISE       = 0.35
TEXT_SUPPRESS       = 0.5
SCORE_PERCENTILE    = 95             # верхний перцентиль для маски

# Визуал: «кислотные» пятна + приглушённый фон
ELA_USE_CLAHE       = True
ELA_VIS_GAIN        = 1.55           # усиливаем ELA
ELA_VIS_GAMMA       = 0.82
HEAT_COLORMAP       = cv2.COLORMAP_TURBO  # контрастная палитра
OVERLAY_ALPHA       = 0.75           # плотность заливки
CONTOUR_THICK       = 3              # контур зон
GLOW_THICK          = 9              # «свечение» (внешний обвод)

RESULTS_DIR = Path(__file__).parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- Низкоуровневые карты -----------------
def local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var

def noise_map_from_gray(gray_u8: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    var = local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var  # 1 — «странный» шум

def ela_map(pil_img: Image.Image) -> np.ndarray:
    """Ансамбль ELA (0..255, uint8)."""
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
        maps.append(np.array(diff.convert("L")))
    ela = np.maximum.reduce(maps)
    return ela  # uint8 0..255

def text_mask(pil_img: Image.Image) -> np.ndarray:
    """Маска печатного текста (0/255)."""
    g = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    # фикс: сначала изображение, затем ядро
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return txt  # 0..255


# ----------------- Score и маска -----------------
def fused_score_from_maps(ela_u8: np.ndarray, gray_u8: np.ndarray) -> np.ndarray:
    """Score 0..1: 0 — нормально, 1 — сильная аномалия."""
    ela = ela_u8.astype(np.float32) / 255.0
    noise = noise_map_from_gray(gray_u8, k=VAR_KERNEL)
    score = SCORE_W_ELA * ela + SCORE_W_NOISE * noise
    tmask = text_mask(Image.fromarray(gray_u8)).astype(np.float32) / 255.0
    score = score * (1.0 - TEXT_SUPPRESS * tmask)
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score.astype(np.float32)

def suspicious_mask_from_score(score: np.ndarray, percentile: int = SCORE_PERCENTILE) -> np.ndarray:
    thr = float(np.percentile(score, percentile))
    return (score >= thr).astype(np.uint8) * 255

def clean_and_filter_mask(mask: np.ndarray,
                          min_area_ratio: float = MIN_AREA_RATIO,
                          morph: int = MASK_MORPH) -> np.ndarray:
    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    H, W = mask.shape[:2]
    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


# ----------------- Зоны -----------------
def extract_regions(mask: np.ndarray, score_map: np.ndarray,
                    max_regions: int = 6, pad: int = 8) -> List[Dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs: List[Dict] = []
    H, W = mask.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        roi = score_map[y0:y1, x0:x1]
        sc = float(np.mean(roi)) if roi.size else 0.0
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})
    regs.sort(key=lambda r: r["score"], reverse=True)
    return regs[:max_regions]


# ----------------- Визуализация -----------------
def vivid_ela_visual(ela_gray_u8: np.ndarray,
                     use_clahe: bool = ELA_USE_CLAHE,
                     gain: float = ELA_VIS_GAIN,
                     gamma: float = ELA_VIS_GAMMA,
                     cmap: int = HEAT_COLORMAP) -> np.ndarray:
    """Яркий контрастный ELA (BGR)."""
    x = ela_gray_u8.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        x = clahe.apply(x)
    f = np.clip((x.astype(np.float32) / 255.0) * gain, 0, 1)
    f = np.power(f, gamma)
    x = (f * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x, cmap)

def _desaturate_and_blur(bgr: np.ndarray) -> np.ndarray:
    """Делаем фон «гладким»: лёгкая десатурация + блюр."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= 0.25  # уменьшили насыщенность
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    bgr_soft = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr_soft = cv2.GaussianBlur(bgr_soft, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return bgr_soft

def acid_overlay(base_bgr: np.ndarray,
                 score: np.ndarray,
                 mask: np.ndarray,
                 *,
                 colormap: int = HEAT_COLORMAP,
                 alpha: float = OVERLAY_ALPHA,
                 contour_thick: int = CONTOUR_THICK,
                 glow_thick: int = GLOW_THICK) -> np.ndarray:
    """
    «Кислотные» зоны: фон приглушён, сами зоны в TURBO, с контуром и лёгким свечением.
    """
    H, W = mask.shape[:2]
    # Фон
    soft_bgr = _desaturate_and_blur(base_bgr)
    out = soft_bgr.copy()

    # Теплокарта из score
    hm = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm, colormap)

    # Наложение только в маске
    m3 = np.repeat((mask > 0)[:, :, None], 3, axis=2)
    out[m3] = cv2.addWeighted(base_bgr[m3], 1.0 - alpha, hm_bgr[m3], alpha, 0)

    # Свечение (внешний обвод)
    if glow_thick > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glow_thick, glow_thick))
        glow = cv2.dilate(mask, kernel, iterations=1)
        glow_edge = cv2.bitwise_and(glow, cv2.bitwise_not(mask))
        glow_bgr = np.zeros_like(out)
        glow_color = (0, 180, 255)  # яркая «бирюза»
        glow_bgr[glow_edge > 0] = glow_color
        out = cv2.addWeighted(out, 1.0, glow_bgr, 0.25, 0)

    # Контур
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), thickness=contour_thick)

    return out

def draw_boxes(overlay_bgr: np.ndarray, regions: List[Dict]) -> np.ndarray:
    out = overlay_bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(out, f"#{idx}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def save_crops(src_rgb: np.ndarray, regions: List[Dict], stem: str,
               out_dir: Path = RESULTS_DIR) -> List[Dict]:
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
def iter_pdf_pages(pdf_path: Path, dpi: int = PDF_DPI, max_pages: int = MAX_PDF_PAGES):
    with fitz.open(str(pdf_path)) as doc:
        pages = min(len(doc), max_pages)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i + 1, img


# ----------------- Высокоуровневый пайплайн (один режим) -----------------
def process_pil_image(pil: Image.Image,
                      label: str,
                      batch: str,
                      *,
                      score_percentile: int = SCORE_PERCENTILE) -> Dict:
    """
    Один фиксированный режим:
      ELA⊕Noise -> верхний перцентиль -> «кислотный» overlay + боксы + кропы.
    """
    # Карты
    ela_u8 = ela_map(pil)                    # 0..255
    gray_u8 = np.array(pil.convert("L"))     # 0..255
    score = fused_score_from_maps(ela_u8, gray_u8)  # 0..1

    # Маска и зоны
    mask = suspicious_mask_from_score(score, percentile=score_percentile)
    mask = clean_and_filter_mask(mask, min_area_ratio=MIN_AREA_RATIO, morph=MASK_MORPH)
    regions = extract_regions(mask, score, max_regions=6, pad=8)

    # Изображения
    src_rgb = np.array(pil.convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)

    ela_vis_bgr = vivid_ela_visual(ela_u8, use_clahe=ELA_USE_CLAHE,
                                   gain=ELA_VIS_GAIN, gamma=ELA_VIS_GAMMA,
                                   cmap=HEAT_COLORMAP)

    overlay_bgr = acid_overlay(src_bgr, score, mask,
                               colormap=HEAT_COLORMAP,
                               alpha=OVERLAY_ALPHA,
                               contour_thick=CONTOUR_THICK,
                               glow_thick=GLOW_THICK)

    boxed_bgr = draw_boxes(overlay_bgr, regions)

    # Файлы
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name   = f"{stem}_src.jpg"
    ela_name   = f"{stem}_ela.jpg"
    ovl_name   = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"

    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ela_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ovl_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(boxed_bgr,  cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / boxed_name), "JPEG", quality=95)

    crops_meta = save_crops(src_rgb, regions, stem, RESULTS_DIR)

    # Лаконичный вывод
    verdict = "Review recommended" if regions else "No issues found"

    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela":      f"/static/results/{ela_name}",
        "overlay":  f"/static/results/{ovl_name}",
        "boxed":    f"/static/results/{boxed_name}",
        "verdict":  verdict,
        "regions":  len(regions),
        "crops":    crops_meta,
        "summary":  "Top regions highlighted." if regions else "No anomalies detected."
    }

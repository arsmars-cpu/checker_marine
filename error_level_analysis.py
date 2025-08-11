# error_level_analysis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import io

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ===== Параметры ==============================================================

# Ансамбль ELA по нескольким JPEG-качествам
ELA_QUALS: Tuple[int, ...] = (90, 95, 98)

# Сегментация
VAR_KERNEL: int = 9             # окно локальной дисперсии
MIN_AREA_RATIO: float = 0.002   # отсев мелких пятен (доля от площади)
MASK_MORPH: int = 3             # морфология (ядро)
SCORE_W_ELA: float = 0.65       # вес ELA в общей карте
SCORE_W_NOISE: float = 0.35     # вес карты «непохожести» шума
TEXT_SUPPRESS: float = 0.5      # подавление печатного текста в score
DEFAULT_PERCENTILE: int = 95    # чувствительность (порог по перцентилю)

# Визуал
COLORMAP: int = cv2.COLORMAP_TURBO
ELA_GAIN: float = 1.5           # усиление ELA-визуала (1.2..1.8)
ELA_GAMMA: float = 0.85         # гамма (ниже -> светлее)
ELA_USE_CLAHE: bool = True      # локальный контраст для ELA
OVERLAY_ALPHA: float = 0.58     # «кислотность» оверлея
BOX_COLOR = (255, 255, 0)       # BGR: жёлтые рамки
CONTOUR_COLOR = (0, 0, 255)     # BGR: красные контуры
ARROW_COLOR = (255, 255, 255)   # BGR: белые стрелки
BOX_THICK: int = 2
CONTOUR_THICK: int = 2
ARROW_THICK: int = 2

# Старый блочный отчёт (если нужно)
BLOCK: int = 24


# ===== ELA / Noise / Text =====================================================

def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    """Один прогон ELA на JPEG-качестве q -> float32 HxWx3 (0..255)."""
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=q, optimize=True)
    buf.seek(0)
    comp = Image.open(buf).convert('RGB')
    ela = ImageChops.difference(pil_img.convert('RGB'), comp)

    extrema = ela.getextrema()
    maxv = 0
    for e in extrema:
        maxv = max(maxv, e[1] if isinstance(e, tuple) else e)
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max(1, maxv))
    return np.asarray(ela).astype(np.float32)


def ela_ensemble_gray(pil_img: Image.Image) -> np.ndarray:
    """Ансамблевый ELA -> grayscale uint8 (0..255)."""
    pil_img = pil_img.convert('RGB')
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]   # HxWx3
    ela = np.mean(arrs, axis=0)
    # L2 по каналам -> [0..1]
    ela_gray = np.sqrt(np.sum(ela ** 2, axis=2))
    ela_gray = (ela_gray - ela_gray.min()) / (ela_gray.ptp() + 1e-6)
    return (ela_gray * 255.0 + 0.5).astype(np.uint8)


def _local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sqmean = cv2.blur(g * g, (k, k))
    return np.clip(sqmean - mean * mean, 0, None)


def noise_map_from_gray(gray_u8: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    """Карта «непохожести» локального шума (0..1): 1 — подозрительно ровно/иначе."""
    var = _local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var


def text_mask(pil_img: Image.Image) -> np.ndarray:
    """Маска печатного текста (0/1 float32)."""
    g = np.asarray(pil_img.convert('L'))
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    # FIX: cv2.dilate(image, kernel, ...)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return (txt > 0).astype(np.float32)


def fused_score(pil_img: Image.Image) -> np.ndarray:
    """
    Итоговая карта 0..1: ELA ⊕ Noise с приглушением печатного текста.
    0 — обычная область, 1 — «аномально шумно/непохоже».
    """
    # ELA -> [0..1]
    ela_u8 = ela_ensemble_gray(pil_img)
    ela = ela_u8.astype(np.float32) / 255.0

    # Noise -> [0..1]
    g = np.asarray(pil_img.convert('L')).astype(np.float32)
    g = (g - g.min()) / (g.ptp() + 1e-6)
    noise = noise_map_from_gray((g * 255).astype(np.uint8), k=VAR_KERNEL)

    # fuse
    score = SCORE_W_ELA * ela + SCORE_W_NOISE * noise

    # suppress printed text
    tmask = text_mask(pil_img)
    score = score * (1.0 - TEXT_SUPPRESS * tmask)

    # normalize
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score.astype(np.float32)


# ===== Сегментация ============================================================

def regions_from_score(score: np.ndarray,
                       percentile: int = DEFAULT_PERCENTILE,
                       morph: int = MASK_MORPH,
                       min_area_ratio: float = MIN_AREA_RATIO,
                       pad: int = 8,
                       top_k: int = 6) -> Tuple[np.ndarray, List[Dict]]:
    """
    score (0..1) → бинарная маска (uint8 0/255) + топ-регионы [{x,y,w,h,score}].
    """
    H, W = score.shape
    thr = float(np.percentile(score, percentile))
    mask = (score >= thr).astype(np.uint8) * 255

    # морфология
    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 1)

    # фильтр по площади
    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regs: List[Dict] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        roi = score[y0:y1, x0:x1]
        sc = float(np.mean(roi)) if roi.size else 0.0
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})

    regs.sort(key=lambda r: r["score"], reverse=True)
    return mask, regs[:top_k]


# ===== Визуал =================================================================

def make_ela_visual(ela_u8: np.ndarray,
                    gain: float = ELA_GAIN,
                    gamma: float = ELA_GAMMA,
                    use_clahe: bool = ELA_USE_CLAHE,
                    colormap: int = COLORMAP) -> np.ndarray:
    """Яркий «кислотный» ELA-визуал: uint8 → BGR."""
    x = ela_u8.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        x = clahe.apply(x)
    xf = np.clip((x.astype(np.float32) / 255.0) * gain, 0, 1)
    xf = np.power(xf, gamma)
    x8 = (xf * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, colormap)


def _draw_arrows(bgr: np.ndarray, regions: List[Dict]) -> None:
    """Добавляем стрелки, «смотрящие» на зоны."""
    H, W = bgr.shape[:2]
    for r in regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cx, cy = x + w // 2, y + h // 2
        start = (max(0, x - int(0.06 * W)), max(0, y - int(0.06 * H)))
        end = (cx, cy)
        cv2.arrowedLine(bgr, start, end, ARROW_COLOR, ARROW_THICK, tipLength=0.25)


def acid_overlay_on_original(pil_img: Image.Image,
                             score: np.ndarray,
                             regions: List[Dict],
                             alpha: float = OVERLAY_ALPHA,
                             colormap: int = COLORMAP) -> np.ndarray:
    """«Кислотная» теплокарта на оригинале + рамки + стрелки. Возвращает BGR."""
    src_bgr = cv2.cvtColor(np.asarray(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    hm = (score * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm, colormap)

    over = cv2.addWeighted(src_bgr, 1.0, hm_bgr, alpha, 0)

    thr = int(np.percentile(hm, 97))
    _, mask_hi = cv2.threshold(hm, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_hi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, contours, -1, CONTOUR_COLOR, CONTOUR_THICK)

    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(over, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICK)
        cv2.putText(over, f"#{idx}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, BOX_COLOR, 1, cv2.LINE_AA)

    _draw_arrows(over, regions)
    return over


# ===== Сохранение визуалов и кропов ==========================================

def save_visuals_and_crops(pil_img: Image.Image,
                           score: np.ndarray,
                           regions: List[Dict],
                           out_dir: Path,
                           stem: str,
                           *,
                           ela_u8: np.ndarray | None = None,
                           ela_gain: float = ELA_GAIN,
                           ela_gamma: float = ELA_GAMMA,
                           use_clahe: bool = ELA_USE_CLAHE,
                           colormap: int = COLORMAP) -> Tuple[str, str, str, list]:
    """
    Сохраняет:
      - ELA-визуал (кислотный)
      - Overlay на оригинал (теплокарта + рамки + стрелки)
      - Boxed (копия overlay — для совместимости)
      - «лид»-кроп самой сильной зоны
    Возвращает имена файлов: (ela_name, overlay_name, boxed_name, crops_list)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ELA (BGR)
    if ela_u8 is None:
        ela_u8 = ela_ensemble_gray(pil_img)
    ela_bgr = make_ela_visual(ela_u8, gain=ela_gain, gamma=ela_gamma, use_clahe=use_clahe, colormap=colormap)

    # Overlay/Boxed (BGR)
    over_bgr = acid_overlay_on_original(pil_img, score, regions, alpha=OVERLAY_ALPHA, colormap=colormap)
    boxed_bgr = over_bgr.copy()

    # Save BGR напрямую — БЕЗ конверта в RGB!
    ela_name   = f"{stem}_ela.jpg"
    ovl_name   = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"
    cv2.imwrite(str(out_dir / ela_name),   ela_bgr,   [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / ovl_name),   over_bgr,  [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / boxed_name), boxed_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # «лид»-кроп
    crops = []
    if regions:
        src = np.asarray(pil_img.convert('RGB'))
        H, W, _ = src.shape
        r0 = regions[0]
        x0, y0 = max(0, r0["x"]), max(0, r0["y"])
        x1, y1 = min(W, r0["x"] + r0["w"]), min(H, r0["y"] + r0["h"])
        crop = src[y0:y1, x0:x1]
        if crop.size:
            name = f"{stem}_lead_crop.jpg"
            Image.fromarray(crop).save(out_dir / name, "JPEG", quality=95)
            crops.append({
                "index": 1,
                "score": r0["score"],
                "filename": name,
                "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
            })

    return ela_name, ovl_name, boxed_name, crops


# ===== Блочный отчёт (опционально) ===========================================

def block_report(score: np.ndarray, block: int = BLOCK, top_k: int = 8) -> List[Dict]:
    H, W = score.shape
    boxes = []
    for y in range(0, H, block):
        for x in range(0, W, block):
            s = score[y:min(y + block, H), x:min(x + block, W)]
            val = float(np.mean(s))
            boxes.append((val, x, y, min(block, W - x), min(block, H - y)))
    boxes.sort(reverse=True, key=lambda t: t[0])
    return [
        {"score": round(v, 4), "x": x, "y": y, "w": w, "h": h}
        for v, x, y, w, h in boxes[:top_k]
    ]

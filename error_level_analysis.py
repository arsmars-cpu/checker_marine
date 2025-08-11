# error_level_analysis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Union
import io

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ===== Параметры ==============================================================

# Ансамбль ELA по нескольким JPEG-качествам
ELA_QUALS: Tuple[int, ...] = (90, 95, 98)

# Сегментация (усилили чувствительность)
VAR_KERNEL: int = 9               # окно локальной дисперсии
MIN_AREA_RATIO: float = 0.0002    # ловим мельче (~0.02% кадра)
MASK_MORPH: int = 5               # чуть больше склейка мелких разрывов
SCORE_W_ELA: float = 0.65
SCORE_W_NOISE: float = 0.35
TEXT_SUPPRESS: float = 0.15       # меньше глушим текст (часто правят текст)
DEFAULT_PERCENTILE: int = 90      # мягче порог

# Визуал
COLORMAP: int = cv2.COLORMAP_TURBO
ELA_GAIN: float = 1.7             # ELA ярче
ELA_GAMMA: float = 0.85
ELA_USE_CLAHE: bool = True
OVERLAY_ALPHA: float = 0.58
BOX_COLOR = (255, 255, 0)
CONTOUR_COLOR = (0, 0, 255)
ARROW_COLOR = (255, 255, 255)
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


# ===== Сегментация + фолбэки =================================================

def _build_mask_from_percentiles(score: np.ndarray,
                                 percentiles: Union[int, List[int], Tuple[int, ...]],
                                 morph: int) -> np.ndarray:
    """Строим маску по одному или нескольким перцентилям и берём объединение."""
    if isinstance(percentiles, (list, tuple)):
        masks = []
        for p in percentiles:
            thr = float(np.percentile(score, p))
            masks.append((score >= thr).astype(np.uint8) * 255)
        mask = np.maximum.reduce(masks)
    else:
        thr = float(np.percentile(score, int(percentiles)))
        mask = (score >= thr).astype(np.uint8) * 255

    if morph > 0:
        kernel = np.ones((morph, morph), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 1)
    return mask


def _hot_windows(score: np.ndarray, k: int = 3, win: int = 48) -> List[Dict]:
    """Safety‑net: берём k самых «горячих» окон win×win без сильного перекрытия."""
    H, W = score.shape
    heat = cv2.GaussianBlur(score, (0, 0), 2)
    regs: List[Dict] = []
    taken = np.zeros_like(score, dtype=np.uint8)

    for _ in range(k):
        yx = np.unravel_index(np.argmax(heat * (1 - taken)), heat.shape)
        y, x = int(yx[0]), int(yx[1])
        x0 = max(0, x - win // 2); y0 = max(0, y - win // 2)
        x1 = min(W, x0 + win);     y1 = min(H, y0 + win)
        roi = score[y0:y1, x0:x1]
        if roi.size == 0 or float(roi.max()) < 0.2:
            break
        regs.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                     "score": float(roi.mean())})
        taken[y0:y1, x0:x1] = 1
    return regs


def regions_from_score(score: np.ndarray,
                       percentile: Union[int, List[int], Tuple[int, ...]] = (85, 90, 95, 97),
                       morph: int = MASK_MORPH,
                       min_area_ratio: float = MIN_AREA_RATIO,
                       pad: int = 8,
                       top_k: int = 6) -> Tuple[np.ndarray, List[Dict]]:
    """
    score (0..1) → бинарная маска (uint8 0/255) + топ-регионы [{x,y,w,h,score}].

    Используем объединение нескольких перцентилей — ловим и «жирные» вмешательства,
    и тонкие правки. Если пусто — включаем два фолбэка.
    """
    H, W = score.shape
    mask = _build_mask_from_percentiles(score, percentile, morph)

    # контуры + отсев по площади
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
    regs = regs[:top_k]

    # Fallback 1: горячие блоки помельче
    if not regs:
        block = 16
        boxes = []
        for by in range(0, H, block):
            for bx in range(0, W, block):
                s = score[by:min(by+block,H), bx:min(bx+block,W)]
                if s.size == 0:
                    continue
                boxes.append((float(s.mean()), bx, by, min(block, W-bx), min(block, H-by)))
        boxes.sort(reverse=True, key=lambda t: t[0])
        regs = [{"x": x, "y": y, "w": w, "h": h, "score": round(v,4)} for v,x,y,w,h in boxes[:min(3, top_k)]]

    # Fallback 2: окна вокруг глобальных максимумов
    if not regs:
        regs = _hot_windows(score, k=min(3, top_k), win=48)

    return mask, regs


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

    if ela_u8 is None:
        ela_u8 = ela_ensemble_gray(pil_img)
    ela_bgr = make_ela_visual(ela_u8, gain=ela_gain, gamma=ela_gamma, use_clahe=use_clahe, colormap=colormap)

    over_bgr = acid_overlay_on_original(pil_img, score, regions, alpha=OVERLAY_ALPHA, colormap=colormap)
    boxed_bgr = over_bgr.copy()

    ela_name   = f"{stem}_ela.jpg"
    ovl_name   = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"
    cv2.imwrite(str(out_dir / ela_name),   ela_bgr,   [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / ovl_name),   over_bgr,  [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / boxed_name), boxed_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

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

# error_level_analysis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Union
import io
import uuid

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ===== Параметры ==============================================================

ELA_QUALS: Tuple[int, ...] = (90, 95, 98)

# Сегментация (чувствительная, с фолбэками)
VAR_KERNEL: int = 9
MIN_AREA_RATIO: float = 0.00005
MASK_MORPH: int = 5
SCORE_W_ELA: float = 0.65
SCORE_W_NOISE: float = 0.35
TEXT_SUPPRESS: float = 0.15
DEFAULT_PERCENTILE: int = 90

# Визуал
COLORMAP: int = cv2.COLORMAP_TURBO
ELA_GAIN: float = 1.7
ELA_GAMMA: float = 0.85
ELA_USE_CLAHE: bool = True
OVERLAY_ALPHA: float = 0.58
BOX_COLOR = (255, 255, 0)
CONTOUR_COLOR = (0, 0, 255)
ARROW_COLOR = (255, 255, 255)
BOX_THICK: int = 2
CONTOUR_THICK: int = 2
ARROW_THICK: int = 2

BLOCK: int = 24


# ===== ELA / Noise / Text =====================================================

def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
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
    pil_img = pil_img.convert('RGB')
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela = np.mean(arrs, axis=0)
    ela_gray = np.sqrt(np.sum(ela ** 2, axis=2))
    # NumPy 2.0: используем np.ptp
    ela_gray = (ela_gray - ela_gray.min()) / (np.ptp(ela_gray) + 1e-6)
    return (ela_gray * 255.0 + 0.5).astype(np.uint8)


def _local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sqmean = cv2.blur(g * g, (k, k))
    return np.clip(sqmean - mean * mean, 0, None)


def noise_map_from_gray(gray_u8: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    var = _local_variance(gray_u8, k)
    var = (var - var.min()) / (np.ptp(var) + 1e-6)
    return 1.0 - var


def text_mask(pil_img: Image.Image) -> np.ndarray:
    g = np.asarray(pil_img.convert('L'))
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return (txt > 0).astype(np.float32)


def fused_score(pil_img: Image.Image) -> np.ndarray:
    ela_u8 = ela_ensemble_gray(pil_img)
    ela = ela_u8.astype(np.float32) / 255.0

    g = np.asarray(pil_img.convert('L')).astype(np.float32)
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    noise = noise_map_from_gray((g * 255).astype(np.uint8), k=VAR_KERNEL)

    score = SCORE_W_ELA * ela + SCORE_W_NOISE * noise
    score = score * (1.0 - TEXT_SUPPRESS * text_mask(pil_img))
    score = (score - score.min()) / (np.ptp(score) + 1e-6)
    return score.astype(np.float32)


# ===== Сегментация + локальный z‑score и фолбэки =============================

def _build_mask_from_percentiles(score: np.ndarray,
                                 percentiles: Union[int, List[int], Tuple[int, ...]],
                                 morph: int) -> np.ndarray:
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


def _local_zscore(score: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    mu  = cv2.GaussianBlur(score, (0, 0), sigma)
    mu2 = cv2.GaussianBlur(score * score, (0, 0), sigma)
    var = np.maximum(mu2 - mu * mu, 1e-6)
    z = (score - mu) / np.sqrt(var)
    z = np.clip(z, 0, None)
    z = (z - z.min()) / (np.ptp(z) + 1e-6)
    return z.astype(np.float32)


def _hot_windows(score: np.ndarray, k: int = 3, win: int = 48) -> List[Dict]:
    H, W = score.shape
    heat = cv2.GaussianBlur(score, (0, 0), 2)
    regs: List[Dict] = []
    taken = np.zeros_like(score, dtype=np.uint8)
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(heat * (1 - taken)), heat.shape)
        x0 = max(0, int(x - win // 2)); y0 = max(0, int(y - win // 2))
        x1 = min(W, x0 + win);         y1 = min(H, y0 + win)
        roi = score[y0:y1, x0:x1]
        if roi.size == 0 or float(roi.max()) < 0.2:
            break
        regs.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "score": float(roi.mean())})
        taken[y0:y1, x0:x1] = 1
    return regs


def regions_from_score(score: np.ndarray,
                       percentile: Union[int, List[int], Tuple[int, ...]] = (82, 88, 94, 97),
                       morph: int = MASK_MORPH,
                       min_area_ratio: float = MIN_AREA_RATIO,
                       pad: int = 12,
                       top_k: int = 6) -> Tuple[np.ndarray, List[Dict]]:
    used = np.clip(0.6 * score + 0.4 * _local_zscore(score), 0, 1)

    H, W = used.shape
    mask = _build_mask_from_percentiles(used, percentile, morph)

    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regs: List[Dict] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        roi = used[y0:y1, x0:x1]
        sc = float(np.mean(roi)) if roi.size else 0.0
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})

    regs.sort(key=lambda r: r["score"], reverse=True)
    regs = regs[:top_k]

    if not regs:
        block = 16
        boxes = []
        for by in range(0, H, block):
            for bx in range(0, W, block):
                s = used[by:min(by+block, H), bx:min(bx+block, W)]
                if s.size == 0:
                    continue
                boxes.append((float(s.mean()), bx, by, min(block, W-bx), min(block, H-by)))
        boxes.sort(reverse=True, key=lambda t: t[0])
        regs = [{"x": x, "y": y, "w": w, "h": h, "score": round(v, 4)} for v, x, y, w, h in boxes[:min(3, top_k)]]

    if not regs:
        regs = _hot_windows(used, k=min(3, top_k), win=48)

    return mask, regs


# ===== Визуал =================================================================

def make_ela_visual(ela_u8: np.ndarray,
                    gain: float = ELA_GAIN,
                    gamma: float = ELA_GAMMA,
                    use_clahe: bool = ELA_USE_CLAHE,
                    colormap: int = COLORMAP) -> np.ndarray:
    x = ela_u8.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        x = clahe.apply(x)
    xf = np.clip((x.astype(np.float32) / 255.0) * gain, 0, 1)
    xf = np.power(xf, gamma)
    x8 = (xf * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, colormap)


def _draw_arrows(bgr: np.ndarray, regions: List[Dict]) -> None:
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


# ===== Вердикт + раннер =======================================================

def verdict_from_maps(score: np.ndarray,
                      mask: np.ndarray,
                      regions: List[Dict]) -> Tuple[str, str, float]:
    suspicious_pct = 100.0 * float((mask > 0).sum()) / float(mask.size)
    n = len(regions)
    max_sc = float(score.max()) if score.size else 0.0
    mean_top = float(np.mean([r["score"] for r in regions[:3]])) if n else 0.0

    is_red = (n >= 2 and mean_top >= 0.45) or (n >= 1 and max_sc >= 0.60 and suspicious_pct >= 0.05)
    is_yellow = (n >= 1 and mean_top >= 0.30) or (max_sc >= 0.50)

    if is_red:
        return "High (likely edited)", "red", round(suspicious_pct, 2)
    if is_yellow:
        return "Medium (possible edits)", "yellow", round(suspicious_pct, 2)
    return "Low (no clear edits)", "green", round(suspicious_pct, 2)


def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    """
    Единая точка входа: строит score/регионы, сохраняет оригинал/визуалы,
    возвращает dict под index.html (пути — web /static/...).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    scr = fused_score(pil_img)
    mask, regs = regions_from_score(scr, percentile=(82, 88, 94, 97))

    ela_u8 = ela_ensemble_gray(pil_img)
    ela_name, ovl_name, boxed_name, crops = save_visuals_and_crops(
        pil_img, scr, regs, out_dir, stem, ela_u8=ela_u8
    )

    verdict, severity, suspicious_pct = verdict_from_maps(scr, mask, regs)

    to_web = lambda name: f"/static/results/{Path(name).name}"

    return {
        "label":   label,
        "original": to_web(orig_name),
        "ela":      to_web(ela_name),
        "overlay":  to_web(ovl_name),
        "boxed":    to_web(boxed_name),
        "verdict":  verdict,
        "severity": severity,
        "regions":  len(regs),
        "crops": [
            {"index": i + 1, "score": r["score"],
             "url": to_web(crops[0]["filename"]) if (i == 0 and crops) else "",
             "box": {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]}}
            for i, r in enumerate(regs)
        ],
        "summary": f"{verdict} — suspicious area ≈ {suspicious_pct:.2f}% across {len(regs)} region(s).",
        "report":  ""
    }

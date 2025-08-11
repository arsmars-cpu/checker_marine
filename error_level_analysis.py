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

# Визуал: «старый фиолетовый»
COLORMAP: int = cv2.COLORMAP_MAGMA
VIS_P_LO: int = 2
VIS_P_HI: int = 98
ELA_GAIN: float = 1.6
ELA_GAMMA: float = 0.9
ELA_USE_CLAHE: bool = False
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


# ===== Сегментация ============================================================

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


# ===== Визуал =================================================================

def make_ela_visual(ela_u8: np.ndarray,
                    gain: float = ELA_GAIN,
                    gamma: float = ELA_GAMMA,
                    use_clahe: bool = ELA_USE_CLAHE,
                    colormap: int = COLORMAP) -> np.ndarray:
    x = ela_u8.astype(np.float32)
    p_lo = np.percentile(x, VIS_P_LO)
    p_hi = np.percentile(x, VIS_P_HI)
    if p_hi <= p_lo:
        p_lo, p_hi = float(x.min()), float(x.max())
    x = (x - p_lo) / max(1e-6, (p_hi - p_lo))
    x = np.clip(x, 0, 1)
    x = np.clip(x * gain, 0, 1)
    x = np.power(x, gamma)
    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, colormap)


# ===== Сохранение ==============================================================

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

    ela_name   = f"{stem}_ela.jpg"
    cv2.imwrite(str(out_dir / ela_name), ela_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return ela_name, "", "", []


# ===== Раннер =================================================================

def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)
    scr = fused_score(pil_img)
    ela_u8 = ela_ensemble_gray(pil_img)
    ela_name, _, _, _ = save_visuals_and_crops(pil_img, scr, [], out_dir, stem, ela_u8=ela_u8)
    to_web = lambda name: f"/static/results/{Path(name).name}"
    return {
        "label":   label,
        "original": to_web(orig_name),
        "ela":      to_web(ela_name),
        "overlay":  "",
        "boxed":    "",
        "verdict":  "",
        "severity": "",
        "regions":  0,
        "crops":    [],
        "summary":  "",
        "report":   ""
    }

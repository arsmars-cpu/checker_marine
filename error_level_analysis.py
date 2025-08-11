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

# Сегментация (оставил, если дальше пригодится)
VAR_KERNEL: int = 9
MIN_AREA_RATIO: float = 0.00005
MASK_MORPH: int = 5
SCORE_W_ELA: float = 0.65
SCORE_W_NOISE: float = 0.35
TEXT_SUPPRESS: float = 0.15
DEFAULT_PERCENTILE: int = 90

# Визуал
# classic/robust делаем отдельными функциями; LUT общий — MAGMA
COLORMAP_CLASSIC = cv2.COLORMAP_MAGMA
COLORMAP_ROBUST  = cv2.COLORMAP_MAGMA

BLOCK: int = 24


# ===== ELA ====================================================================

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


# ===== Noise/Text (оставлено на будущее, сейчас не используем) ================

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


# ===== ДВА СТИЛЯ ВИЗУАЛИЗАЦИИ =================================================

def _vis_classic(ela_u8: np.ndarray) -> np.ndarray:
    """
    «Как раньше»: нормировка по p99.5, гамма >1 — тёмный фиолетовый фон,
    зерно хорошо видно.
    """
    x = ela_u8.astype(np.float32)
    s = float(np.percentile(x, 99.5))
    x = np.clip(x / max(1.0, s), 0, 1)
    x = np.power(x, 1.35)
    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, COLORMAP_CLASSIC)


def _vis_robust(ela_u8: np.ndarray) -> np.ndarray:
    """
    Робастная растяжка p2..p98 + лёгкий gain и gamma<1 — поднимаем низ.
    """
    x = ela_u8.astype(np.float32)
    p_lo, p_hi = np.percentile(x, 2), np.percentile(x, 98)
    if p_hi <= p_lo:
        p_lo, p_hi = float(x.min()), float(x.max())
    x = (x - p_lo) / max(1e-6, (p_hi - p_lo))
    x = np.clip(x * 1.6, 0, 1)
    x = np.power(x, 0.9)
    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, COLORMAP_ROBUST)


def save_ela_both(ela_u8: np.ndarray, out_dir: Path, stem: str) -> Tuple[str, str]:
    """
    Сохраняем две версии: classic и robust. Возвращаем имена файлов.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr_classic = _vis_classic(ela_u8)
    bgr_robust  = _vis_robust(ela_u8)
    n1 = f"{stem}_ela_classic.jpg"
    n2 = f"{stem}_ela_robust.jpg"
    cv2.imwrite(str(out_dir / n1), bgr_classic, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / n2), bgr_robust,  [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return n1, n2


# ===== Раннер =================================================================

def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    # считаем ELA и сохраняем обе визуалки
    _ = fused_score(pil_img)  # пусть считается, если потом захочешь вернуть метрики
    ela_u8 = ela_ensemble_gray(pil_img)
    ela_classic, ela_robust = save_ela_both(ela_u8, out_dir, stem)

    to_web = lambda name: f"/static/results/{Path(name).name}"
    return {
        "label":   label,
        "original": to_web(orig_name),
        "ela":      to_web(ela_classic),   # classic (старый вид)
        "overlay":  to_web(ela_robust),    # robust (p2–p98)
        "boxed":    "",
        "verdict":  "",
        "severity": "",
        "regions":  0,
        "crops":    [],
        "summary":  "",
        "report":   ""
    }

# error_level_analysis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import io
import uuid

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ===== Параметры ==============================================================

ELA_QUALS: Tuple[int, ...] = (90, 95, 98)

# Визуал (ELA-only, без регионов/оверлеев)
COLORMAP: int = cv2.COLORMAP_INFERNO   # хорошо показывает цветной шум
ELA_GAIN: float = 1.6
ELA_GAMMA: float = 0.90
ELA_USE_CLAHE: bool = True             # локальный контраст перед растяжкой


# ===== Ансамбль ELA ===========================================================

def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    """Разность оригинала и JPEG(q) с авторастяжкой яркости."""
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=q, optimize=True)
    buf.seek(0)
    comp = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(pil_img.convert("RGB"), comp)

    # автоусиление динамики (по максимуму по каналам)
    extrema = ela.getextrema()
    maxv = 0
    for e in extrema:
        maxv = max(maxv, e[1] if isinstance(e, tuple) else e)
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max(1, maxv))
    return np.asarray(ela).astype(np.float32)


def ela_ensemble_gray(pil_img: Image.Image) -> np.ndarray:
    """Ансамблевый ELA → градации серого [0..255] (NumPy 2.0 safe)."""
    pil_img = pil_img.convert("RGB")
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]  # (H,W,3)
    ela = np.mean(arrs, axis=0)
    ela_gray = np.sqrt(np.sum(ela ** 2, axis=2))         # энергия по каналам
    ela_gray = (ela_gray - ela_gray.min()) / (np.ptp(ela_gray) + 1e-6)
    return (ela_gray * 255.0 + 0.5).astype(np.uint8)


# ===== Визуал ELA (робастная растяжка) =======================================

def make_ela_visual(ela_u8: np.ndarray,
                    gain: float = ELA_GAIN,
                    gamma: float = ELA_GAMMA,
                    use_clahe: bool = ELA_USE_CLAHE,
                    colormap: int = COLORMAP) -> np.ndarray:
    """
    Растягиваем по перцентилям (5..99.5) + гамма, чтобы «цветной» шум читался,
    а одиночные выбросы не съедали контраст.
    """
    x = ela_u8.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        x = clahe.apply(x)

    lo = float(np.percentile(x, 5.0))
    hi = float(np.percentile(x, 99.5))
    if hi <= lo:
        hi = lo + 1.0

    xf = (x.astype(np.float32) - lo) / (hi - lo)
    xf = np.clip(xf * gain, 0.0, 1.0)
    xf = np.power(xf, gamma)
    x8 = (xf * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, colormap)  # BGR


# ===== Раннер (ELA-only) ======================================================

def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    """
    ELA-only: сохраняем оригинал и ELA-визуал. Никаких регионов/рамок/оверлеев.
    Возвращаем ключи, совместимые с index.html.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"

    # Оригинал
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    # ELA
    ela_u8 = ela_ensemble_gray(pil_img)
    ela_vis = make_ela_visual(ela_u8)
    ela_name = f"{stem}_ela.jpg"
    Image.fromarray(cv2.cvtColor(ela_vis, cv2.COLOR_BGR2RGB)).save(out_dir / ela_name, "JPEG", quality=95)

    # Простой числовой индикатор: доля «горячих» пикселей (порог p95 по исходной ELA)
    t = float(np.percentile(ela_u8, 95.0))
    hot_pct = 100.0 * float((ela_u8 >= t).sum()) / ela_u8.size
    if hot_pct >= 10.0:
        verdict, severity = "High (likely edited)", "red"
    elif hot_pct >= 3.0:
        verdict, severity = "Medium (possible edits)", "yellow"
    else:
        verdict, severity = "Low (no clear edits)", "green"

    to_web = lambda name: f"/static/results/{Path(name).name}"

    return {
        "label":   label,
        "original": to_web(orig_name),
        "ela":      to_web(ela_name),
        "overlay":  "",   # не используем
        "boxed":    "",   # не используем
        "verdict":  verdict,
        "severity": severity,
        "regions":  0,
        "crops":    [],
        "summary": f"{verdict} — hot ELA pixels ≈ {hot_pct:.2f}%.",
        "report":   ""
    }

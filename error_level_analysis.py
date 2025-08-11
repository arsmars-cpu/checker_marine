# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import io
import uuid

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ===== Параметры =====

ELA_QUALS: Tuple[int, ...] = (90, 95, 98)

# Визуал (классика и «усиленная»)
COLORMAP = cv2.COLORMAP_MAGMA       # тот самый фиолетово-оранжевый LUT

# Классика — делаем темнее, как на твоём эталоне
CLASSIC_DARKEN = 0.78               # 0.75..0.85 — подстрой при желании

# robust: p2..p98 растяжка + лёгкий gain/gamma
VIS_P_LO = 2
VIS_P_HI = 98
ROBUST_GAIN  = 1.15
ROBUST_GAMMA = 1.00

# Вердикт по КЛАССИКЕ: адаптивный порог p99.x
P_HOT   = 99.4                      # повысь до 99.5 если ещё «жарит»
LOW_MAX = 2.0                       # <2% горячих пикселей — Low
MID_MAX = 8.0                       # 2–8% — Medium, иначе High

# ===== Базовые шаги ELA =====

def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=q, optimize=True)
    buf.seek(0)
    comp = Image.open(buf).convert('RGB')
    ela = ImageChops.difference(pil_img.convert('RGB'), comp)

    # автонормализация, как раньше
    extrema = ela.getextrema()
    maxv = 0
    for e in extrema:
        maxv = max(maxv, e[1] if isinstance(e, tuple) else e)
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max(1, maxv))

    return np.asarray(ela).astype(np.float32)

def ela_ensemble_gray(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert('RGB')
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela = np.mean(arrs, axis=0)                   # (H,W,3) float32
    ela_gray = np.sqrt(np.sum(ela ** 2, axis=2))  # L2 по каналам
    ela_gray = (ela_gray - ela_gray.min()) / (np.ptp(ela_gray) + 1e-6)
    return (ela_gray * 255.0 + 0.5).astype(np.uint8)  # (H,W) uint8

# ===== Визуализации =====

def make_ela_visual_classic(ela_u8: np.ndarray) -> np.ndarray:
    """Тёмно-фиолетовая карта «как раньше»: чуть темнее + MAGMA LUT."""
    x = np.clip(ela_u8.astype(np.float32) * CLASSIC_DARKEN, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(x, COLORMAP)

def make_ela_visual_robust(ela_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust-вариант: p2..p98 растяжка + лёгкий gain/gamma.
    Возвращает (визуал BGR, нормированную карту 0..1 без цветовой палитры).
    """
    x = ela_u8.astype(np.float32)
    p_lo = np.percentile(x, VIS_P_LO)
    p_hi = np.percentile(x, VIS_P_HI)
    if p_hi <= p_lo:  # защита на плоских случаях
        p_lo, p_hi = float(x.min()), float(x.max())
    x = (x - p_lo) / max(1e-6, (p_hi - p_lo))
    x = np.clip(x, 0, 1)
    x = np.clip(x * ROBUST_GAIN, 0, 1)
    x = np.power(x, ROBUST_GAMMA)
    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    vis = cv2.applyColorMap(x8, COLORMAP)
    return vis, x  # BGR, float(0..1)

# ===== Вердикт (по классике, адаптивный порог) =====

def _hot_percent_from_classic(ela_u8: np.ndarray) -> Tuple[float, float]:
    thr = float(np.percentile(ela_u8, P_HOT))
    hot_pct = float((ela_u8 >= thr).mean() * 100.0)
    return hot_pct, thr / 255.0

def _verdict_from_pct(hot_pct: float) -> Tuple[str, str]:
    if hot_pct < LOW_MAX:  return "Low (no clear edits)", "green"
    if hot_pct < MID_MAX:  return "Medium (possible edits)", "yellow"
    return "High (likely edited)", "red"

# ===== Раннер =====

def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Сохраняем оригинал
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    # 2) Строим ELA
    ela_u8 = ela_ensemble_gray(pil_img)

    # classic (твоё любимое)
    classic_bgr = make_ela_visual_classic(ela_u8)
    classic_name = f"{stem}_ela.jpg"
    cv2.imwrite(str(out_dir / classic_name), classic_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # robust (альтернативный взгляд)
    robust_bgr, robust_norm = make_ela_visual_robust(ela_u8)
    robust_name = f"{stem}_overlay.jpg"
    cv2.imwrite(str(out_dir / robust_name), robust_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 3) Вердикт — по классике с адаптивным порогом p99.x
    hot_pct, thr_norm = _hot_percent_from_classic(ela_u8)
    verdict, severity = _verdict_from_pct(hot_pct)

    # 4) Ответ для шаблона
    to_web = lambda name: f"/static/results/{Path(name).name}"
    return {
        "label":    label,
        "original": to_web(orig_name),
        "ela":      to_web(classic_name),   # слева — «классика»
        "overlay":  to_web(robust_name),    # по кнопке — robust
        "boxed":    "",                     # без боксов/стрелок
        "verdict":  verdict,
        "severity": severity,
        "regions":  0,
        "crops":    [],
        "summary":  f"{verdict} — hot pixels ≈ {hot_pct:.2f}% (adaptive p{P_HOT}, thr≈{thr_norm:.2f}).",
        "report":   ""
    }

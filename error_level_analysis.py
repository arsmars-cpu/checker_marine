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
VIS_P_LO = 2                        # для robust: p2..p98 растяжка
VIS_P_HI = 98
ROBUST_GAIN  = 1.6
ROBUST_GAMMA = 0.9

# Порог и рубрики для вердикта по robust-карте
HOT_THR   = 0.80         # пиксели ярче этого считаем «горячими»
LOW_MAX   = 2.0          # < 2% — Low
MID_MAX   = 8.0          # 2–8% — Medium, иначе High

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
    """Прямой colormap без усилений — «как раньше» (тёмно-фиолетовая карта)."""
    return cv2.applyColorMap(ela_u8, COLORMAP)

def make_ela_visual_robust(ela_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust-вариант: p2..p98 растяжка + лёгкий gain/gamma.
    Возвращает (визуал BGR, нормированную карту 0..1 без цветовой палитры).
    """
    x = ela_u8.astype(np.float32)

    p_lo = np.percentile(x, VIS_P_LO)
    p_hi = np.percentile(x, VIS_P_HI)
    if p_hi <= p_lo:  # защита от вырожденных случаев
        p_lo, p_hi = float(x.min()), float(x.max())

    x = (x - p_lo) / max(1e-6, (p_hi - p_lo))
    x = np.clip(x, 0, 1)
    x = np.clip(x * ROBUST_GAIN, 0, 1)
    x = np.power(x, ROBUST_GAMMA)

    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    vis = cv2.applyColorMap(x8, COLORMAP)
    return vis, x  # BGR, float(0..1)

# ===== Вердикт по robust-карте =====

def verdict_from_hot_percent(hot_pct: float) -> Tuple[str, str]:
    if hot_pct < LOW_MAX:
        return "Low (no clear edits)", "green"
    if hot_pct < MID_MAX:
        return "Medium (possible edits)", "yellow"
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

    # robust (усиленная для «плоских» сканов)
    robust_bgr, robust_norm = make_ela_visual_robust(ela_u8)
    robust_name = f"{stem}_overlay.jpg"
    cv2.imwrite(str(out_dir / robust_name), robust_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 3) Результат (по robust)
    hot_pct = float((robust_norm >= HOT_THR).mean() * 100.0)
    verdict, severity = verdict_from_hot_percent(hot_pct)

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
        "summary":  f"{verdict} — hot pixels ≈ {hot_pct:.2f}% (threshold {int(HOT_THR*100)}%).",
        "report":   ""
    }

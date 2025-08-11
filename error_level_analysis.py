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

# Цветной ELA (как в «идеале») — LUT не используем, карты цветные сами по себе
COLORMAP = cv2.COLORMAP_MAGMA  # оставил для возможных экспериментов

# Robust-нормализация (для второй картинки)
VIS_P_LO = 2
VIS_P_HI = 98
ROBUST_GAIN  = 1.6
ROBUST_GAMMA = 0.9

# Порог и рубрики вердикта (считаем ТОЛЬКО белые зоны на classic)
LOW_MAX = 2.0        # <2% — Low
MID_MAX = 8.0        # 2–8% — Medium, иначе High

# Белая зона = высокая яркость + низкая насыщенность (на classic карте)
WHITE_V_THR = 0.82   # V >= 0.82 (~>= 210/255) — достаточно светло
WHITE_S_THR = 0.28   # S <= 0.28 (~<= 72/255) — «почти белый/серый»

# ===== Базовые шаги ELA =====

def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    """
    JPEG(q) → разность с оригиналом, авто-растяжка яркости, возвращаем RGB float32 [0..255].
    Даёт «цветное зерно» без LUT.
    """
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=q, optimize=True)
    buf.seek(0)
    comp = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(pil_img.convert("RGB"), comp)

    # авто-нормализация как в классике
    extrema = ela.getextrema()
    maxv = 0
    for e in extrema:
        maxv = max(maxv, e[1] if isinstance(e, tuple) else e)
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max(1, maxv))

    return np.asarray(ela).astype(np.float32)  # (H,W,3) float32 0..255


def ela_ensemble_rgb(pil_img: Image.Image) -> np.ndarray:
    """
    Ансамбль по качествам JPEG → среднее RGB, без LUT.
    """
    pil_img = pil_img.convert("RGB")
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela_rgb = np.mean(arrs, axis=0)  # (H,W,3) float32
    # приводим в 0..255
    mx = float(np.max(ela_rgb)) if ela_rgb.size else 1.0
    if mx <= 0:
        mx = 1.0
    ela_rgb = np.clip(ela_rgb * (255.0 / mx), 0, 255)
    return ela_rgb.astype(np.uint8)

# ===== Визуализации (оба — без colormap, чистый RGB шум) =====

def make_ela_color_classic(ela_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Классика: лёгкая гамма, чтобы зёрна читались на тёмном фоне.
    """
    x = ela_rgb_u8.astype(np.float32) / 255.0
    x = np.power(x, 0.9)
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def make_ela_color_robust(ela_rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust-вариант для «плоских» сканов: по каналам p2..p98 + gain/gamma.
    Возвращает (rgb_u8, norm_gray 0..1) — norm_gray можно использовать для своих метрик.
    """
    x = ela_rgb_u8.astype(np.float32)
    y = np.empty_like(x)
    for c in range(3):
        ch = x[:, :, c]
        p_lo = np.percentile(ch, VIS_P_LO)
        p_hi = np.percentile(ch, VIS_P_HI)
        if p_hi <= p_lo:
            p_lo, p_hi = float(ch.min()), float(ch.max())
        ch = (ch - p_lo) / max(1e-6, (p_hi - p_lo))
        ch = np.clip(ch * ROBUST_GAIN, 0, 1)
        ch = np.power(ch, ROBUST_GAMMA)
        y[:, :, c] = ch
    rgb_robust = (np.clip(y, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    # нормированная «яркость» (0..1) — если понадобится
    g = np.sqrt(np.sum(y ** 2, axis=2)) / np.sqrt(3.0)
    return rgb_robust, g

# ===== Подсчёт «белых» артефактов на classic =====

def hot_pct_from_white(classic_rgb_u8: np.ndarray,
                       v_thr: float = WHITE_V_THR,
                       s_thr: float = WHITE_S_THR) -> Tuple[float, np.ndarray]:
    """
    % «горячих» пикселей как белых зон: V >= v_thr и S <= s_thr.
    Используем HSV от classic карты (RGB → HSV).
    """
    hsv = cv2.cvtColor(classic_rgb_u8, cv2.COLOR_RGB2HSV)
    H, W = hsv.shape[:2]
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    mask = (V >= v_thr) & (S <= s_thr)
    hot_pct = float(mask.mean() * 100.0) if H * W else 0.0
    return hot_pct, mask.astype(np.uint8)

# ===== Вердикт =====

def verdict_from_hot_pct(hot_pct: float) -> Tuple[str, str]:
    if hot_pct < LOW_MAX:
        return "Low (no clear edits)", "green"
    if hot_pct < MID_MAX:
        return "Medium (possible edits)", "yellow"
    return "High (likely edited)", "red"

# ===== Раннер =====

def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    """
    Сохраняет оригинал + две цветные ELA-карты (classic и robust),
    считает % «белых» зон на classic и отдаёт статус.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"

    # 1) Оригинал
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    # 2) ELA RGB
    ela_rgb = ela_ensemble_rgb(pil_img)

    # 2.1) classic (твоя эталонная «цветная» карта)
    classic_rgb = make_ela_color_classic(ela_rgb)
    classic_name = f"{stem}_ela.jpg"
    cv2.imwrite(str(out_dir / classic_name),
                cv2.cvtColor(classic_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 2.2) robust (усиленная карта для сравнения/переключателя)
    robust_rgb, _robust_norm = make_ela_color_robust(ela_rgb)
    robust_name = f"{stem}_overlay.jpg"
    cv2.imwrite(str(out_dir / robust_name),
                cv2.cvtColor(robust_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 3) Метрика/вердикт: считаем ТОЛЬКО белые зоны на classic
    hot_pct, _ = hot_pct_from_white(classic_rgb)
    verdict, severity = verdict_from_hot_pct(hot_pct)

    to_web = lambda name: f"/static/results/{Path(name).name}"
    return {
        "label":    label,
        "original": to_web(orig_name),
        "ela":      to_web(classic_name),   # слева — classic (с «цветным шумом»)
        "overlay":  to_web(robust_name),    # кнопка «Toggle» покажет robust
        "boxed":    "",
        "verdict":  verdict,
        "severity": severity,
        "regions":  0,
        "crops":    [],
        "summary":  f"{verdict} — white-hot ≈ {hot_pct:.2f}% (V≥{int(WHITE_V_THR*100)}%, S≤{int(WHITE_S_THR*100)}%).",
        "report":   ""
    }

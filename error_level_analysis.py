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

# ===== Параметры =====

ELA_QUALS: Tuple[int, ...] = (90, 95, 98)

# Цветной ELA — LUT НЕ используем (оставлен для экспериментов)
COLORMAP = cv2.COLORMAP_MAGMA

# Robust-нормализация (для второй картинки)
VIS_P_LO = 2
VIS_P_HI = 98
ROBUST_GAIN  = 1.6
ROBUST_GAMMA = 0.9

# Порог и рубрики вердикта
LOW_MAX = 1.0        # <1% — Low
MID_MAX = 5.0        # 1–5% — Medium, иначе High

# Белая зона на classic = высокая яркость + низкая насыщенность (строже)
WHITE_V_THR = 0.86   # V >= 0.86 (~>= 219/255)
WHITE_S_THR = 0.32   # S <= 0.32 (~<= 82/255)

# Яркая зона на robust (строже)
ROBUST_HOT_THR = 0.86

# Фильтры для подсчёта (борьба с шумом и бликами по краям)
BORDER_INSET = 0.02           # срез 2% по периметру
MIN_BLOB_AREA_RATIO = 0.0002  # ≥0.02% площади кадра
MIN_BLOB_AREA_ABS   = 96      # и не меньше 96 пикселей

# ===== Базовые шаги ELA =====

def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    """
    JPEG(q) → разность с оригиналом, авто-растяжка яркости.
    Даёт «цветное зерно» без каких-либо colormap.
    Возвращает RGB float32 в диапазоне [0..255].
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
    Ансамбль по качествам JPEG → среднее RGB (без LUT).
    """
    pil_img = pil_img.convert("RGB")
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela_rgb = np.mean(arrs, axis=0)  # (H,W,3) float32

    # нормализуем к 0..255
    mx = float(np.max(ela_rgb)) if ela_rgb.size else 1.0
    if mx <= 0:
        mx = 1.0
    ela_rgb = np.clip(ela_rgb * (255.0 / mx), 0, 255)
    return ela_rgb.astype(np.uint8)

# ===== Визуализации (оба — без colormap, чистый RGB шум) =====

def make_ela_color_classic(ela_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Classic: лёгкая гамма, чтобы зёрна читались на тёмном фоне.
    """
    x = ela_rgb_u8.astype(np.float32) / 255.0
    x = np.power(x, 0.9)
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def make_ela_color_robust(ela_rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust для «плоских» сканов: по каналам p2..p98 + gain/gamma.
    Возвращает (rgb_u8, norm_gray 0..1), где norm_gray — нормированная яркость,
    используемая в метрике.
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

    # нормированная «яркость» 0..1 (для подсчёта ярких зон)
    g = np.sqrt(np.sum(y ** 2, axis=2)) / np.sqrt(3.0)
    return rgb_robust, g

# ===== Подсчёт артефактов (fusion: classic OR robust) =====

def hot_pct_from_fused(classic_rgb_u8: np.ndarray,
                       robust_norm: np.ndarray,
                       v_thr: float = WHITE_V_THR,
                       s_thr: float = WHITE_S_THR,
                       r_thr: float = ROBUST_HOT_THR,
                       inset_ratio: float = BORDER_INSET,
                       min_area_ratio: float = MIN_BLOB_AREA_RATIO,
                       min_area_abs: int = MIN_BLOB_AREA_ABS) -> Tuple[float, np.ndarray]:
    """
    Артефакт = (V>=v_thr & S<=s_thr) на classic  OR  (robust_norm_smooth>=r_thr).
    Шум чистим морфологией и фильтром по минимальной площади. Рамку по периметру отбрасываем.
    Возвращает (процент, бинарную маску).
    """
    H, W = classic_rgb_u8.shape[:2]

    # 1) белые пятна на classic (по HSV)
    hsv = cv2.cvtColor(classic_rgb_u8, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    white_mask = (V >= v_thr) & (S <= s_thr)

    # 2) яркие зоны на robust (слегка сгладим)
    rob_smooth = cv2.GaussianBlur(robust_norm.astype(np.float32), (0, 0), 1.0)
    robust_mask = (rob_smooth >= r_thr)

    fused = (white_mask | robust_mask).astype(np.uint8)

    # 3) срезаем рамку (часто даёт блики/виньетку)
    b = int(round(min(H, W) * inset_ratio))
    if b > 0:
        fused[:b, :] = 0; fused[-b:, :] = 0
        fused[:, :b] = 0; fused[:, -b:] = 0

    # 4) морфология против «соли-перца»
    kernel = np.ones((3, 3), np.uint8)
    fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) фильтр по площади компонент
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
    keep = np.zeros_like(fused)
    min_area = max(int(min_area_ratio * H * W), int(min_area_abs))
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1

    fused = keep
    hot_pct = float(fused.mean() * 100.0) if fused.size else 0.0
    return hot_pct, fused

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
    считает % артефактов по fusion-маске и отдаёт статус.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"

    # 1) Оригинал
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    # 2) ELA RGB
    ela_rgb = ela_ensemble_rgb(pil_img)

    # 2.1) classic (эталонная «цветная» карта)
    classic_rgb = make_ela_color_classic(ela_rgb)
    classic_name = f"{stem}_ela.jpg"
    cv2.imwrite(str(out_dir / classic_name),
                cv2.cvtColor(classic_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 2.2) robust (усиленная карта + норма 0..1)
    robust_rgb, robust_norm = make_ela_color_robust(ela_rgb)
    robust_name = f"{stem}_overlay.jpg"
    cv2.imwrite(str(out_dir / robust_name),
                cv2.cvtColor(robust_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 3) Метрика/вердикт: fusion classic+robust с фильтрами
    hot_pct, _ = hot_pct_from_fused(classic_rgb, robust_norm)
    verdict, severity = verdict_from_hot_pct(hot_pct)

    to_web = lambda name: f"/static/results/{Path(name).name}"
    return {
        "label":    label,
        "original": to_web(orig_name),
        "ela":      to_web(classic_name),   # слева — classic (как на твоём эталоне)
        "overlay":  to_web(robust_name),    # по переключателю — robust
        "boxed":    "",
        "verdict":  verdict,
        "severity": severity,
        "regions":  0,
        "crops":    [],
        "summary":  (
            f"{verdict} — hot≈{hot_pct:.2f}%  "
            f"(classic: V≥{int(WHITE_V_THR*100)}%, S≤{int(WHITE_S_THR*100)}%; "
            f"robust: ≥{int(ROBUST_HOT_THR*100)}%; "
            f"border cut {int(BORDER_INSET*100)}%)"
        ),
        "report":   ""
    }

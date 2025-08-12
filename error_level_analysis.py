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

# Robust-нормализация (только для второй картинки в UI, на вердикт не влияет)
VIS_P_LO = 2
VIS_P_HI = 98
ROBUST_GAIN  = 1.6
ROBUST_GAMMA = 0.9

# SIMPLE: пороги и фильтры «только белый шум»
SIMPLE_WHITE_V_THR   = 0.80   # V >= 0.80  (~>= 204/255) — достаточно светло
SIMPLE_WHITE_S_THR   = 0.45   # S <= 0.45  (~<= 115/255) — низкая насыщенность (почти белый/серый)
SIMPLE_BORDER_INSET  = 0.01   # срез 1% по периметру (боремся с бликами/виньеткой)
SIMPLE_MIN_AREA_RATIO= 0.00005# >= 0.005% площади кадра (режем «соль-перец»)
SIMPLE_MIN_AREA_ABS  = 64     # и не меньше 64 пикселей (страховка)
SIMPLE_YELLOW_THR    = 0.60   # >=0.60% белого шума => Medium (жёлтый), иначе Low (зелёный)

# DEBUG: сохранять ли диагностические маски (white)
DEBUG_SAVE_MASKS = True
MASK_OVERLAY_ALPHA = 0.65  # прозрачность оверлея маски на оригинале

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


def make_ela_color_robust(ela_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Robust-вариант (для сравнения в UI): по каналам p2..p98 + gain/gamma.
    На вердикт НЕ влияет.
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
    return (np.clip(y, 0, 1) * 255.0 + 0.5).astype(np.uint8)

# ===== SIMPLE: подсчёт «белого шума» на classic =====

def simple_white_mask(classic_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Маска белого шума: HSV V>=SIMPLE_WHITE_V_THR и S<=SIMPLE_WHITE_S_THR,
    с обрезкой рамки, морфологией и фильтром по площади.
    """
    H, W = classic_rgb_u8.shape[:2]
    hsv = cv2.cvtColor(classic_rgb_u8, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    m = ((V >= SIMPLE_WHITE_V_THR) & (S <= SIMPLE_WHITE_S_THR)).astype(np.uint8)

    # рамка
    b = int(round(min(H, W) * SIMPLE_BORDER_INSET))
    if b > 0:
        m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0

    # морфология + фильтр площади
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    min_area = max(int(SIMPLE_MIN_AREA_RATIO * H * W), SIMPLE_MIN_AREA_ABS)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1

    return keep

def simple_white_pct(classic_rgb_u8: np.ndarray) -> Tuple[float, np.ndarray]:
    m = simple_white_mask(classic_rgb_u8)
    pct = float(m.mean() * 100.0) if m.size else 0.0
    return pct, m

# ===== Вспомогательное: визуализация маски (DEBUG) =====

def _overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int],
                         alpha: float = MASK_OVERLAY_ALPHA) -> np.ndarray:
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)
    color = np.zeros_like(base); color[:, :] = color_bgr
    m = (mask.astype(np.uint8) * 255)
    m3 = cv2.merge([m, m, m])
    over = np.where(m3 > 0, cv2.addWeighted(base, 1 - alpha, color, alpha, 0), base)
    return over

def _save_debug_white(pil_img: Image.Image,
                      white_mask: np.ndarray,
                      out_dir: Path,
                      stem: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names: Dict[str, str] = {}
    # бинарка
    p = out_dir / f"{stem}_white_bin.jpg"
    cv2.imwrite(str(p), (white_mask.astype(np.uint8) * 255), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    names["white_bin"] = p.name
    # оверлей на оригинал
    orig_rgb = np.asarray(pil_img.convert("RGB"))
    over_white = _overlay_mask_on_rgb(orig_rgb, white_mask, (255, 255, 255))  # белый
    p = out_dir / f"{stem}_white_overlay.jpg"
    cv2.imwrite(str(p), over_white, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    names["white_overlay"] = p.name
    return names

# ===== Вердикт (простая логика) =====

def verdict_simple_from_white_pct(white_pct: float) -> Tuple[str, str]:
    """
    Только два состояния:
    - white_pct >= SIMPLE_YELLOW_THR → Medium (yellow)
    - иначе → Low (green)
    """
    if white_pct >= SIMPLE_YELLOW_THR:
        return "Medium (possible edits)", "yellow"
    return "Low (no clear edits)", "green"

# ===== Раннер =====

def run_image(pil_img: Image.Image, label: str, batch: str, out_dir: Path) -> Dict:
    """
    Простой режим: сохраняет оригинал + 2 цветных ELA (classic и robust для UI),
    считает % белого шума на classic и выдаёт Low/Medium.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"

    # 1) Оригинал
    orig_name = f"{stem}_src.jpg"
    Image.fromarray(np.asarray(pil_img.convert("RGB"))).save(out_dir / orig_name, "JPEG", quality=95)

    # 2) ELA RGB (классика + robust для визуала)
    ela_rgb = ela_ensemble_rgb(pil_img)

    classic_rgb = make_ela_color_classic(ela_rgb)
    classic_name = f"{stem}_ela.jpg"
    cv2.imwrite(str(out_dir / classic_name),
                cv2.cvtColor(classic_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    robust_rgb = make_ela_color_robust(ela_rgb)
    robust_name = f"{stem}_overlay.jpg"
    cv2.imwrite(str(out_dir / robust_name),
                cv2.cvtColor(robust_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 3) Метрика/вердикт: только белый шум
    white_pct, white_mask = simple_white_pct(classic_rgb)
    verdict, severity = verdict_simple_from_white_pct(white_pct)

    # 4) DEBUG-маски (опционально)
    debug = {}
    if DEBUG_SAVE_MASKS:
        dbg = _save_debug_white(pil_img, white_mask, out_dir, stem)
        to_web_dbg = lambda name: f"/static/results/{Path(name).name}"
        debug = {k: to_web_dbg(v) for k, v in dbg.items()}

    to_web = lambda name: f"/static/results/{Path(name).name}"
    return {
        "label":    label,
        "original": to_web(orig_name),
        "ela":      to_web(classic_name),   # слева — classic (эталонный вид)
        "overlay":  to_web(robust_name),    # по переключателю — robust (для наглядности)
        "boxed":    "",
        "verdict":  verdict,                # ТОЛЬКО Low/Medium в простом режиме
        "severity": severity,
        "regions":  0,
        "crops":    [],
        "summary":  (
            f"{verdict} — white-noise ≈ {white_pct:.2f}% "
            f"(V≥{int(SIMPLE_WHITE_V_THR*100)}%, S≤{int(SIMPLE_WHITE_S_THR*100)}%, "
            f"border cut {int(SIMPLE_BORDER_INSET*100)}%, "
            f"area≥max({SIMPLE_MIN_AREA_ABS}px, {SIMPLE_MIN_AREA_RATIO*100:.3f}%))"
        ),
        "debug":    debug,
        "report":   ""
    }

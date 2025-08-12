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

# Порог и рубрики вердикта — чувствительнее
LOW_MAX = 0.6        # <0.6% — Low
MID_MAX = 3.5        # 0.6–3.5% — Medium, иначе High

# Белая зона на classic = высокая яркость + низкая насыщенность (мягче)
WHITE_V_THR = 0.80   # V >= 0.80  (~>= 204/255)
WHITE_S_THR = 0.45   # S <= 0.45  (~<= 115/255)

# Яркая зона на robust (мягче)
ROBUST_HOT_THR = 0.80

# NEW: цветной шум — минимальная «хрома» и адаптивный порог по квантилю
CHROMA_MIN = 0.18    # базовый минимум 0..1 (если документ «плоский»)
CHROMA_Q   = 90      # берём макс(CHROMA_MIN, q-квантиль по карте хромы)

# Фильтры для подсчёта (борьба с шумом и бликами по краям)
BORDER_INSET = 0.01            # срез 1% по периметру
MIN_BLOB_AREA_RATIO = 0.00005  # ≥0.005% площади кадра
MIN_BLOB_AREA_ABS   = 64       # и не меньше 64 пикселей

# DEBUG: сохранять ли диагностические маски
DEBUG_SAVE_MASKS = True
MASK_OVERLAY_ALPHA = 0.65  # прозрачность оверлея масок на оригинале

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

# ===== Карта «цветного шума» (хрома) =====

def chroma_energy(rgb_u8: np.ndarray) -> np.ndarray:
    """
    Энергия цветного шума: Ec = sqrt((R-G)^2 + (R-B)^2 + (G-B)^2) / (sqrt(3)*255).
    Диапазон 0..1. На чистой бумаге низкая, у «цветных» спеклов заметно выше.
    """
    x = rgb_u8.astype(np.float32)
    R, G, B = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    ec = np.sqrt((R - G) ** 2 + (R - B) ** 2 + (G - B) ** 2) / (np.sqrt(3) * 255.0)
    return np.clip(ec, 0, 1)

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

def hot_masks_from_fused(classic_rgb_u8: np.ndarray,
                         robust_norm: np.ndarray,
                         v_thr: float = WHITE_V_THR,
                         s_thr: float = WHITE_S_THR,
                         r_thr: float = ROBUST_HOT_THR,
                         inset_ratio: float = BORDER_INSET,
                         min_area_ratio: float = MIN_BLOB_AREA_RATIO,
                         min_area_abs: int = MIN_BLOB_AREA_ABS
                         ) -> Tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray]:
    """
    Считает три маски: WHITE (classic по HSV), ROBUST (по яркости robust_norm c фильтром по хроме),
    FUSED = OR. Применяет обрезку рамки, морфологию и фильтр по площади.
    Возвращает: (pct_fused, mask_fused, pct_white, mask_white, pct_robust, mask_robust).
    """
    H, W = classic_rgb_u8.shape[:2]

    # 0) карта цветного шума + адаптивный порог по квантилю
    ec = chroma_energy(classic_rgb_u8)                              # 0..1
    ec_thr = max(CHROMA_MIN, float(np.percentile(ec, CHROMA_Q)))    # адаптивно под кадр

    # 1) белые пятна на classic (по HSV)
    hsv = cv2.cvtColor(classic_rgb_u8, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    white_mask = (V >= v_thr) & (S <= s_thr)

    # 2) яркие зоны на robust (слегка сгладим) + фильтр по «цветной» хроме
    rob_smooth = cv2.GaussianBlur(robust_norm.astype(np.float32), (0, 0), 1.0)
    robust_mask = (rob_smooth >= r_thr) & (ec >= ec_thr)

    # 3) OR
    fused = (white_mask | robust_mask).astype(np.uint8)

    # 4) срезаем рамку
    b = int(round(min(H, W) * inset_ratio))
    if b > 0:
        for m in (white_mask, robust_mask, fused):
            m[:b, :] = 0; m[-b:, :] = 0; m[:, :b] = 0; m[:, -b:] = 0

    # 5) морфология против «соли-перца»
    kernel = np.ones((3, 3), np.uint8)
    white_mask  = cv2.morphologyEx(white_mask.astype(np.uint8),  cv2.MORPH_OPEN, kernel, iterations=1)
    robust_mask = cv2.morphologyEx(robust_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    fused       = cv2.morphologyEx(fused.astype(np.uint8),       cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) фильтр по площади компонент (для каждой маски одинаково)
    def area_filter(bin_mask: np.ndarray) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        keep = np.zeros_like(bin_mask)
        min_area = max(int(min_area_ratio * H * W), int(min_area_abs))
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        return keep

    white_mask  = area_filter(white_mask)
    robust_mask = area_filter(robust_mask)
    fused       = area_filter(fused)

    pct_white  = float(white_mask.mean()  * 100.0) if white_mask.size  else 0.0
    pct_robust = float(robust_mask.mean() * 100.0) if robust_mask.size else 0.0
    pct_fused  = float(fused.mean()       * 100.0) if fused.size       else 0.0

    return pct_fused, fused, pct_white, white_mask, pct_robust, robust_mask

# ===== Вспомогательное: визуализация масок (DEBUG) =====

def _overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int],
                         alpha: float = MASK_OVERLAY_ALPHA) -> np.ndarray:
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)
    color = np.zeros_like(base)
    color[:, :] = color_bgr
    m = (mask.astype(np.uint8) * 255)
    m3 = cv2.merge([m, m, m])
    over = np.where(m3 > 0, cv2.addWeighted(base, 1 - alpha, color, alpha, 0), base)
    return over

def _save_debug_masks(pil_img: Image.Image,
                      classic_rgb: np.ndarray,
                      fused: np.ndarray,
                      white_mask: np.ndarray,
                      robust_mask: np.ndarray,
                      out_dir: Path,
                      stem: str,
                      chroma_mask: np.ndarray | None = None) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names: Dict[str, str] = {}
    # бинарки
    for name, m in (("white_bin", white_mask), ("robust_bin", robust_mask), ("fused_bin", fused)):
        p = out_dir / f"{stem}_{name}.jpg"
        cv2.imwrite(str(p), (m.astype(np.uint8) * 255), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        names[name] = p.name
    if chroma_mask is not None:
        p = out_dir / f"{stem}_chroma_bin.jpg"
        cv2.imwrite(str(p), (chroma_mask.astype(np.uint8) * 255), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        names["chroma_bin"] = p.name
    # оверлеи на оригинал
    orig_rgb = np.asarray(pil_img.convert("RGB"))
    over_white  = _overlay_mask_on_rgb(orig_rgb, white_mask,  (255, 255, 255))  # белый
    over_robust = _overlay_mask_on_rgb(orig_rgb, robust_mask, (255, 0, 255))    # фуксия
    over_fused  = _overlay_mask_on_rgb(orig_rgb, fused,       (0, 255, 255))    # циан/жёлт
    for name, img in (("white_overlay", over_white),
                      ("robust_overlay", over_robust),
                      ("fused_overlay", over_fused)):
        p = out_dir / f"{stem}_{name}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        names[name] = p.name
    if chroma_mask is not None:
        over_chroma = _overlay_mask_on_rgb(orig_rgb, chroma_mask, (0, 165, 255))  # оранжевый
        p = out_dir / f"{stem}_chroma_overlay.jpg"
        cv2.imwrite(str(p), over_chroma, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        names["chroma_overlay"] = p.name
    return names

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
    (Если DEBUG_SAVE_MASKS=True — дополнительно сохраняет маски white/robust/fused/chroma.)
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
    pct_fused, fused_mask, pct_white, white_mask, pct_robust, robust_mask = hot_masks_from_fused(
        classic_rgb, robust_norm
    )
    verdict, severity = verdict_from_hot_pct(pct_fused)

    # 4) DEBUG-маски (опционально) — добавим и хрому
    debug = {}
    if DEBUG_SAVE_MASKS:
        ec = chroma_energy(classic_rgb)
        ec_thr = max(CHROMA_MIN, float(np.percentile(ec, CHROMA_Q)))
        chroma_mask = (ec >= ec_thr).astype(np.uint8)
        dbg_names = _save_debug_masks(pil_img, classic_rgb, fused_mask, white_mask, robust_mask, out_dir, stem, chroma_mask)
        to_web_dbg = lambda name: f"/static/results/{Path(name).name}"
        debug = {k: to_web_dbg(v) for k, v in dbg_names.items()}

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
            f"{verdict} — hot≈{pct_fused:.2f}% "
            f"(white≈{pct_white:.2f}%, robust≈{pct_robust:.2f}%; "
            f"classic: V≥{int(WHITE_V_THR*100)}%, S≤{int(WHITE_S_THR*100)}%; "
            f"robust: ≥{int(ROBUST_HOT_THR*100)}%; border cut {int(BORDER_INSET*100)}%; "
            f"chroma≥max({int(CHROMA_MIN*100)}%, Q{CHROMA_Q}))"
        ),
        "debug":    debug,  # ссылки на маски, если включено
        "report":   ""
    }

# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance
import fitz  # PyMuPDF

# ----------------- Константы / параметры -----------------
JPEG_QUALITIES      = (90, 95, 98)   # ансамблевый ELA
VAR_KERNEL          = 7              # поменьше окно -> ловим мелкие правки
MIN_AREA_RATIO      = 0.0015         # чуть ниже порог площади
MASK_MORPH          = 3
PDF_DPI             = 200
MAX_PDF_PAGES       = 5

SCORE_W_ELA         = 0.65
SCORE_W_NOISE       = 0.35
TEXT_SUPPRESS       = 0.5

# авто-порог будет считать сам; это верх/низ «здравого» окна маски
AUTO_TARGET_LO = 0.006   # 0.6% площади
AUTO_TARGET_HI = 0.060   # 6% площади
AUTO_PCTL_MIN  = 90
AUTO_PCTL_MAX  = 97

# Визуал
HEAT_COLORMAP       = cv2.COLORMAP_TURBO
OVERLAY_ALPHA       = 0.65

ELA_USE_CLAHE       = True
ELA_VIS_GAIN        = 1.55
ELA_VIS_GAMMA       = 0.82

RESULTS_DIR = Path(__file__).parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Базовые карты -----------------
def local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sqmean = cv2.blur(g * g, (k, k))
    return np.clip(sqmean - mean * mean, 0, None)

def noise_map_from_gray(gray_u8: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    var = local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var

# >>> НОВОЕ: ELA «как раньше»: ансамбль по качествам + L2 по каналам
def ela_map(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    arrs = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_img.save(buf, "JPEG", quality=q, optimize=True)
        buf.seek(0)
        comp = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(pil_img, comp)
        extrema = diff.getextrema()
        maxv = max((e[1] if isinstance(e, tuple) else e) for e in extrema)
        diff = ImageEnhance.Brightness(diff).enhance(255.0 / max(1, maxv))
        arrs.append(np.asarray(diff).astype(np.float32))   # H×W×3
    ela = np.mean(arrs, axis=0)                            # ансамбль
    ela_gray = np.sqrt(np.sum(ela**2, axis=2))             # L2 по каналам
    ela_gray = (ela_gray - ela_gray.min()) / (ela_gray.ptp() + 1e-6)
    return (ela_gray * 255.0 + 0.5).astype(np.uint8)       # 0..255

def text_mask(pil_img: Image.Image) -> np.ndarray:
    g = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, 1)
    txt = cv2.dilate(cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1, dst=txt)
    return txt

def fused_score_from_maps(ela_u8: np.ndarray, gray_u8: np.ndarray) -> np.ndarray:
    ela = ela_u8.astype(np.float32) / 255.0
    noise = noise_map_from_gray(gray_u8, k=VAR_KERNEL)
    score = SCORE_W_ELA * ela + SCORE_W_NOISE * noise
    tmask = text_mask(Image.fromarray(gray_u8)).astype(np.float32) / 255.0
    score = score * (1.0 - TEXT_SUPPRESS * tmask)
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score.astype(np.float32)

# ----------------- Детекторы удаления/вклейки -----------------
def erase_flatness_map(pil_img: Image.Image, txt_mask_255: np.ndarray) -> np.ndarray:
    g = np.array(pil_img.convert("L"))
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    hf = np.abs(lap)
    hf = (hf - hf.min()) / (hf.ptp() + 1e-6)
    band = cv2.dilate(txt_mask_255, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 2).astype(bool)
    flat = 1.0 - hf
    out = np.zeros_like(flat, dtype=np.float32)
    out[band] = flat[band]
    if np.any(out > 0):
        qhi = float(np.quantile(out[out > 0], 0.98))
        out = np.clip(out / (qhi + 1e-6), 0, 1)
    return out

def seam_map(pil_img: Image.Image, ela_u8: np.ndarray) -> np.ndarray:
    g = np.array(pil_img.convert("L")).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    grad = np.sqrt(gx*gx + gy*gy)
    grad = (grad - grad.min())/(grad.ptp()+1e-6)
    ela = ela_u8.astype(np.float32)/255.0
    s = grad * np.power(ela, 0.8)
    s = (s - s.min())/(s.ptp()+1e-6)
    s = cv2.GaussianBlur(s, (0,0), 0.8)
    return s.astype(np.float32)

def copy_move_regions(pil_img: Image.Image, min_matches: int = 12, bin_size: int = 8) -> Tuple[np.ndarray, List[Dict]]:
    img = np.array(pil_img.convert("L"))
    orb = cv2.ORB_create(nfeatures=4000)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or kp is None or len(kp) < 2:
        return np.zeros(img.shape, np.uint8), []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(des, des)
    good = [x for x in m if x.queryIdx != x.trainIdx and x.distance < 35]
    from collections import defaultdict
    buckets = defaultdict(list)
    for x in good:
        p = np.array(kp[x.queryIdx].pt); q = np.array(kp[x.trainIdx].pt)
        dx, dy = q - p
        key = (int(dx//bin_size), int(dy//bin_size))
        buckets[key].append((p, q))
    mask = np.zeros(img.shape, np.uint8)
    regs: List[Dict] = []
    for _, pairs in sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]:
        if len(pairs) < min_matches:
            continue
        pts = np.vstack([np.vstack([p for p,_ in pairs]), np.vstack([q for _,q in pairs])]).astype(np.int32)
        x,y,w,h = cv2.boundingRect(pts)
        cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
        regs.append({"x":int(x), "y":int(y), "w":int(w), "h":int(h), "score":1.0})
    return mask, regs

# ----------------- Маски/регионы -----------------
def suspicious_mask_from_score(score: np.ndarray, percentile: int) -> np.ndarray:
    thr = float(np.percentile(score, percentile))
    return (score >= thr).astype(np.uint8) * 255

def clean_and_filter_mask(mask: np.ndarray,
                          min_area_ratio: float = MIN_AREA_RATIO,
                          morph: int = MASK_MORPH) -> np.ndarray:
    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 1)
    H, W = mask.shape[:2]
    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out

def extract_regions(mask: np.ndarray, score_map: np.ndarray,
                    max_regions: int = 8, pad: int = 8) -> List[Dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs: List[Dict] = []
    H, W = mask.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        roi = score_map[y0:y1, x0:x1]
        sc = float(np.mean(roi)) if roi.size else 0.0
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})
    regs.sort(key=lambda r: r["score"], reverse=True)
    return regs[:max_regions]

# ----------------- Авто-порог -----------------
def auto_percentile(score: np.ndarray,
                    target_lo: float = AUTO_TARGET_LO,
                    target_hi: float = AUTO_TARGET_HI,
                    pmin: int = AUTO_PCTL_MIN,
                    pmax: int = AUTO_PCTL_MAX) -> int:
    """Бинарный поиск перцентиля так, чтобы площадь маски попала в окно [lo,hi]."""
    H, W = score.shape
    lo, hi = pmin, pmax
    best = pmax
    while lo <= hi:
        mid = (lo + hi) // 2
        m = suspicious_mask_from_score(score, mid)
        frac = (m > 0).sum() / float(H * W)
        if target_lo <= frac <= target_hi:
            best = mid; break
        if frac > target_hi:
            lo = mid + 1
        else:
            best = mid
            hi = mid - 1
    return int(max(pmin, min(pmax, best)))

# ----------------- Визуал -----------------
def vivid_ela_visual(ela_gray_u8: np.ndarray,
                     use_clahe: bool = ELA_USE_CLAHE,
                     gain: float = ELA_VIS_GAIN,
                     gamma: float = ELA_VIS_GAMMA,
                     cmap: int = HEAT_COLORMAP) -> np.ndarray:
    x = ela_gray_u8.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        x = clahe.apply(x)
    f = np.clip((x.astype(np.float32) / 255.0) * gain, 0, 1)
    f = np.power(f, gamma)
    x = (f * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x, cmap)

def _desat(bgr: np.ndarray, sat_scale: float = 0.45) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def compose_multi_overlay(base_bgr: np.ndarray,
                          union_map: np.ndarray,   # вся смесь 0..1
                          erase_map: np.ndarray,   # 0..1
                          seam_map_f: np.ndarray,  # 0..1
                          cm_mask: np.ndarray | None,
                          alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """
    TURBO = любые «горячие» зоны (union_map),
    магента = erase, оранж контур = seam, лайм = copy-move боксы.
    """
    out = _desat(base_bgr, 0.45)

    # общий heat
    hm = cv2.applyColorMap((np.clip(union_map, 0, 1) * 255).astype(np.uint8), HEAT_COLORMAP)
    out = cv2.addWeighted(out, 1.0, hm, alpha, 0)

    # erase (magenta)
    er = (np.clip(erase_map, 0, 1) * 255).astype(np.uint8)
    if er.max() > 0:
        mag = np.zeros_like(out); mag[...,2] = 255; mag[...,0] = 255
        out[er > 200] = cv2.addWeighted(out[er > 200], 0.2, mag[er > 200], 0.8, 0)

    # seam (orange thin)
    s = (np.clip(seam_map_f, 0, 1) * 255).astype(np.uint8)
    _, sbin = cv2.threshold(s, 210, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(sbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 140, 255), 2)

    # copy-move (lime)
    if cm_mask is not None and cm_mask.any():
        cnts, _ = cv2.findContours(cm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 180), 2)
    return out

def draw_boxes(overlay_bgr: np.ndarray, regions: List[Dict]) -> np.ndarray:
    out = overlay_bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(out, f"#{idx}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return out

def save_crops(src_rgb: np.ndarray, regions: List[Dict], stem: str,
               out_dir: Path = RESULTS_DIR) -> List[Dict]:
    out = []
    H, W, _ = src_rgb.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = src_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        name = f"{stem}_crop_{idx}.jpg"
        Image.fromarray(crop).save(str(out_dir / name), "JPEG", quality=95)
        out.append({
            "index": idx,
            "score": r["score"],
            "url": f"/static/results/{name}",
            "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        })
    return out

# ----------------- PDF: постраничный рендер -----------------
def iter_pdf_pages(pdf_path: Path, dpi: int = PDF_DPI, max_pages: int = MAX_PDF_PAGES):
    with fitz.open(str(pdf_path)) as doc:
        pages = min(len(doc), max_pages)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i + 1, img

# ----------------- Один режим (multi-cue) -----------------
def process_pil_image(pil: Image.Image,
                      label: str,
                      batch: str,
                      *,
                      score_percentile: int | None = None) -> Dict:
    """
    add/replace (ELA⊕Noise) + erase + seam + copy–move → объединение → оверлей.
    """
    # Базовые карты
    ela_u8 = ela_map(pil)                    # 0..255
    gray_u8 = np.array(pil.convert("L"))     # 0..255
    score_base = fused_score_from_maps(ela_u8, gray_u8)  # 0..1
    txt255 = text_mask(pil)                  # 0/255

    # Новые детекторы
    erase = erase_flatness_map(pil, txt255)       # 0..1
    seam  = seam_map(pil, ela_u8)                 # 0..1
    cm_mask, _ = copy_move_regions(pil)           # 0/255

    # Объединённая карта
    score_union = np.maximum.reduce([
        score_base,
        0.7 * erase,
        0.8 * seam,
        0.9 * ((cm_mask > 0).astype(np.float32))
    ])

    # Авто-порог (если не передан снаружи)
    pctl_union = auto_percentile(score_union)
    mask_union = suspicious_mask_from_score(score_union, pctl_union)
    mask_union = clean_and_filter_mask(mask_union, MIN_AREA_RATIO, MASK_MORPH)
    regions_union = extract_regions(mask_union, score_union, max_regions=8, pad=8)

    # Визуалы
    src_rgb = np.array(pil.convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)

    ela_vis_bgr = vivid_ela_visual(ela_u8, use_clahe=ELA_USE_CLAHE,
                                   gain=ELA_VIS_GAIN, gamma=ELA_VIS_GAMMA,
                                   cmap=HEAT_COLORMAP)

    overlay_bgr = compose_multi_overlay(
        src_bgr,
        union_map=score_union,
        erase_map=erase,
        seam_map_f=seam,
        cm_mask=cm_mask,
        alpha=OVERLAY_ALPHA
    )
    boxed_bgr = draw_boxes(overlay_bgr, regions_union)

    # Файлы
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name   = f"{stem}_src.jpg"
    ela_name   = f"{stem}_ela.jpg"
    ovl_name   = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"

    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ela_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ovl_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(boxed_bgr,  cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / boxed_name), "JPEG", quality=95)

    crops_meta = save_crops(src_rgb, regions_union, stem, RESULTS_DIR)

    verdict = "Review recommended" if regions_union else "No issues found"

    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela":      f"/static/results/{ela_name}",
        "overlay":  f"/static/results/{ovl_name}",
        "boxed":    f"/static/results/{boxed_name}",
        "verdict":  verdict,
        "regions":  len(regions_union),
        "crops":    crops_meta,
        "summary":  f"Auto pctl={pctl_union}. TURBO=add/replace, Magenta=erase, Orange=seam, Lime=copy-move."
    }

# error_level_analysis.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import io

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# ----- Параметры по умолчанию -----
ELA_QUALS = (90, 95, 98)       # ансамбль ELA
BLOCK = 24                      # размер блока (если пользуешь block_report)
DEFAULT_COLORMAP = cv2.COLORMAP_TURBO

# ---------------- ELA / Noise / Text ----------------
def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=q, optimize=True)
    buf.seek(0)
    comp = Image.open(buf).convert('RGB')
    ela = ImageChops.difference(pil_img.convert('RGB'), comp)
    extrema = ela.getextrema()
    maxv = 0
    for e in extrema:
        if isinstance(e, tuple):
            maxv = max(maxv, e[1])
        else:
            maxv = max(maxv, e)
    maxv = max(maxv, 1)
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / maxv)
    return np.asarray(ela).astype(np.float32)

def ela_ensemble_gray(pil_img: Image.Image) -> np.ndarray:
    """ELA 0..255 (uint8, gray)"""
    pil_img = pil_img.convert('RGB')
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela = np.mean(arrs, axis=0)                         # H×W×3
    ela_gray = np.sqrt(np.sum(ela**2, axis=2))          # L2
    ela_gray = (ela_gray - ela_gray.min()) / (ela_gray.ptp() + 1e-6)
    return (ela_gray * 255.0 + 0.5).astype(np.uint8)

def noise_map(pil_img: Image.Image, ksize: int = 9) -> np.ndarray:
    g = np.asarray(pil_img.convert('L')).astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(ksize, ksize))
    mean2 = cv2.boxFilter(g**2, ddepth=-1, ksize=(ksize, ksize))
    var = np.clip(mean2 - mean**2, 0, None)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var  # 1 — «странный» шум

def text_mask(pil_img: Image.Image) -> np.ndarray:
    g = np.asarray(pil_img.convert('L'))
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return (txt > 0).astype(np.float32)  # 0/1

def fused_score(pil_img: Image.Image,
                w_ela: float = 0.65,
                w_noise: float = 0.35,
                text_suppress: float = 0.5,
                noise_ksize: int = 9) -> np.ndarray:
    """Score 0..1: ELA⊕Noise с приглушением печатного текста."""
    ela_u8 = ela_ensemble_gray(pil_img)
    ela = ela_u8.astype(np.float32) / 255.0
    nmap = noise_map(pil_img, noise_ksize)
    tmask = text_mask(pil_img)
    score = w_ela * ela + w_noise * nmap
    score = score * (1.0 - text_suppress * tmask)
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score

# ---------------- Сегментация зон ----------------
def regions_from_score(score: np.ndarray,
                       percentile: int = 95,
                       morph: int = 3,
                       min_area_ratio: float = 0.002,
                       pad: int = 8,
                       top_k: int = 6) -> Tuple[np.ndarray, List[Dict]]:
    """
    Возвращает бинарную маску (uint8 0/255) и топ-зоны [{x,y,w,h,score}].
    """
    H, W = score.shape
    thr = float(np.percentile(score, percentile))
    mask = (score >= thr).astype(np.uint8) * 255

    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regs: List[Dict] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        roi = score[y0:y1, x0:x1]
        sc = float(np.mean(roi)) if roi.size else 0.0
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})

    regs.sort(key=lambda r: r["score"], reverse=True)
    return mask, regs[:top_k]

# ---------------- Визуал ELA и боксы ----------------
def make_ela_visual(ela_u8: np.ndarray,
                    gain: float = 1.4,
                    gamma: float = 0.85,
                    use_clahe: bool = True,
                    colormap: int = DEFAULT_COLORMAP) -> np.ndarray:
    """Яркий контрастный ELA-визуал (uint8 → BGR)."""
    x = ela_u8.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        x = clahe.apply(x)
    xf = np.clip((x.astype(np.float32) / 255.0) * gain, 0, 1)
    xf = np.power(xf, gamma)
    x8 = (xf * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(x8, colormap)  # BGR

def overlay_boxes_on(bgr: np.ndarray, regions: List[Dict]) -> np.ndarray:
    out = bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(out, f"#{idx}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA)
    return out

def save_visuals_and_crops(pil_img: Image.Image,
                           score: np.ndarray,
                           regions: List[Dict],
                           out_dir: Path,
                           stem: str,
                           *,
                           ela_u8: np.ndarray | None = None,
                           ela_gain: float = 1.4,
                           ela_gamma: float = 0.85,
                           use_clahe: bool = True,
                           colormap: int = DEFAULT_COLORMAP) -> Tuple[str, str, str, list]:
    """
    Сохраняет:
      - яркий ELA-визуал (colormap)
      - overlay heatmap
      - boxed (heatmap + рамки)
      - кроп главной зоны
    Возвращает имена файлов: (ela_name, overlay_name, boxed_name, crops_list)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ELA BGR
    if ela_u8 is None:
        ela_u8 = ela_ensemble_gray(pil_img)
    ela_bgr = make_ela_visual(ela_u8, gain=ela_gain, gamma=ela_gamma, use_clahe=use_clahe, colormap=colormap)

    # overlay heatmap (на оригинал)
    src_bgr = cv2.cvtColor(np.asarray(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    hm = (score * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm, colormap)
    overlay_bgr = cv2.addWeighted(src_bgr, 1.0, hm_bgr, 0.55, 0)

    # boxed поверх overlay
    boxed_bgr = overlay_boxes_on(overlay_bgr, regions)

    # save
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"
    cv2.imwrite(str(out_dir / ela_name), cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / ovl_name), cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out_dir / boxed_name), cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # crops (главная зона)
    crops = []
    if regions:
        x, y, w, h = regions[0]["x"], regions[0]["y"], regions[0]["w"], regions[0]["h"]
        src = np.asarray(pil_img.convert('RGB'))
        H, W, _ = src.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = src[y0:y1, x0:x1]
        if crop.size:
            name = f"{stem}_lead_crop.jpg"
            Image.fromarray(crop).save(out_dir / name, "JPEG", quality=95)
            crops.append({"index": 1, "score": regions[0]["score"], "filename": name,
                          "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}})
    return ela_name, ovl_name, boxed_name, crops

# ---------------- Старый отчёт по блокам (на всякий) ----------------
def block_report(score: np.ndarray, block: int = BLOCK, top_k: int = 8) -> List[Dict]:
    H, W = score.shape
    boxes = []
    for y in range(0, H, block):
        for x in range(0, W, block):
            s = score[y:min(y+block, H), x:min(x+block, W)]
            val = float(np.mean(s))
            boxes.append((val, x, y, min(block, W - x), min(block, H - y)))
    boxes.sort(reverse=True, key=lambda t: t[0])
    return [{"score": round(v, 4), "x": x, "y": y, "w": w, "h": h} for v, x, y, w, h in boxes[:top_k]]

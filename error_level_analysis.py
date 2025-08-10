# error_level_analysis.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import io

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

# Параметры
ELA_QUALS = (90, 95, 98)   # качества JPEG для ансамблевого ELA
BLOCK = 24                 # размер блока в отчёте (было 32 — сделал точнее)

# ---------------- ELA / Noise / Text ----------------
def _ela_single(pil_img: Image.Image, q: int) -> np.ndarray:
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=q, optimize=True)
    buf.seek(0)
    comp = Image.open(buf)
    ela = ImageChops.difference(pil_img, comp)
    # нормализация яркости ELA
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

def ela_ensemble(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert('RGB')
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela = np.mean(arrs, axis=0)
    # сворачиваем в серый (L2 по каналам)
    ela_gray = np.sqrt(np.sum(ela**2, axis=2))
    ela_gray = (ela_gray - ela_gray.min()) / (ela_gray.ptp() + 1e-6)
    return ela_gray  # 0..1

def noise_map(pil_img: Image.Image, ksize: int = 9) -> np.ndarray:
    g = np.asarray(pil_img.convert('L')).astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(ksize, ksize))
    mean2 = cv2.boxFilter(g**2, ddepth=-1, ksize=(ksize, ksize))
    var = np.clip(mean2 - mean**2, 0, None)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var  # инверсия: «непохожие» зоны выше

def text_mask(pil_img: Image.Image) -> np.ndarray:
    """Маска печатного текста для снижения ложных срабатываний. Возвращает 0/1 float32."""
    g = np.asarray(pil_img.convert('L'))
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    # тонкие штрихи шрифта
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return (txt > 0).astype(np.float32)  # 1 — текст

def fuse_scores(ela: np.ndarray, noise: np.ndarray, txt_mask_arr: np.ndarray,
                w_ela: float = 0.65, w_noise: float = 0.35, text_suppress: float = 0.6) -> np.ndarray:
    """
    Совмещённый скор: ELA + карта шума, с приглушением печатного текста.
    Возвращает 0..1
    """
    score = w_ela * ela + w_noise * noise
    score = score * (1.0 - text_suppress * txt_mask_arr)  # приглушаем строки
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score

def fused_score(pil_img: Image.Image,
                w_ela: float = 0.65,
                w_noise: float = 0.35,
                text_suppress: float = 0.5,
                noise_ksize: int = 9) -> np.ndarray:
    """
    Готовая score-карта 0..1: ELA⊕Noise с приглушением текста.
    """
    ela = ela_ensemble(pil_img)            # 0..1
    nmap = noise_map(pil_img, noise_ksize) # 0..1 (1 — «странный» шум)
    tmask = text_mask(pil_img)             # 0 или 1
    return fuse_scores(ela, nmap, tmask, w_ela=w_ela, w_noise=w_noise, text_suppress=text_suppress)

# ---------------- Визуализация ----------------
def heatmap_overlay(pil_img: Image.Image, score: np.ndarray, alpha: float = 0.55) -> Image.Image:
    base = np.asarray(pil_img.convert('RGB')).astype(np.float32) / 255.0
    hm = (score * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0  # BGR->RGB
    out = (1 - alpha) * base + alpha * hm
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)

# ---------------- Зоны / Боксы / Кропы ----------------
def regions_from_score(score: np.ndarray,
                       percentile: int = 95,
                       morph: int = 3,
                       min_area_ratio: float = 0.002,
                       pad: int = 8,
                       top_k: int = 6) -> List[Dict]:
    """
    Бинаризуем по верхнему перцентилю, чистим морфологией,
    находим контуры, считаем боксы с подушкой и средний score в них.
    """
    H, W = score.shape
    thr = float(np.percentile(score, percentile))
    mask = (score >= thr).astype(np.uint8) * 255

    kernel = np.ones((morph, morph), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    min_area = int(min_area_ratio * H * W)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regs = []
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
    return regs[:top_k]

def overlay_boxes(pil_img: Image.Image, score: np.ndarray, regions: List[Dict], alpha: float = 0.55) -> Image.Image:
    """
    Теплокарта + рамки с индексами и баллом.
    """
    # базовая теплокарта
    base = np.asarray(heatmap_overlay(pil_img, score, alpha)).copy()
    bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(bgr, f"#{idx} {r['score']:.2f}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def save_heatmap_and_boxed(pil_img: Image.Image,
                           score: np.ndarray,
                           regions: List[Dict],
                           out_dir: Path,
                           stem: str) -> Tuple[str, str]:
    """
    Сохраняет heatmap и heatmap+boxes. Возвращает относительные имена файлов.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    heat = heatmap_overlay(pil_img, score, alpha=0.55)
    boxed = overlay_boxes(pil_img, score, regions, alpha=0.55)

    heat_name = f"{stem}_heat.jpg"
    boxed_name = f"{stem}_boxed.jpg"
    heat.save(out_dir / heat_name, "JPEG", quality=95)
    boxed.save(out_dir / boxed_name, "JPEG", quality=95)
    return heat_name, boxed_name

def save_crops(pil_img: Image.Image,
               regions: List[Dict],
               out_dir: Path,
               stem: str) -> List[Dict]:
    """
    Сохраняет кропы зон. Возвращает список метаданных: index, score, filename, box.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src = np.asarray(pil_img.convert("RGB"))
    H, W, _ = src.shape
    out = []
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = src[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        name = f"{stem}_crop_{idx}.jpg"
        Image.fromarray(crop).save(out_dir / name, "JPEG", quality=95)
        out.append({
            "index": idx,
            "score": r["score"],
            "filename": name,
            "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
        })
    return out

# ---------------- Старый отчёт по блокам (оставил для совместимости) ----------------
def block_report(score: np.ndarray, block: int = BLOCK, top_k: int = 8) -> List[Dict]:
    H, W = score.shape
    boxes = []
    for y in range(0, H, block):
        for x in range(0, W, block):
            s = score[y:min(y+block, H), x:min(x+block, W)]
            val = float(np.mean(s))
            boxes.append((val, x, y, min(block, W - x), min(block, H - y)))
    boxes.sort(reverse=True, key=lambda t: t[0])
    return [
        {"score": round(v, 4), "x": x, "y": y, "w": w, "h": h}
        for v, x, y, w, h in boxes[:top_k]
    ]

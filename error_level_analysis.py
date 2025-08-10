# error_level_analysis.py
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import io

# Качества JPEG для ансамблевого ELA
ELA_QUALS = (90, 95, 98)
# Размер блока для отчёта
BLOCK = 32

def _ela_single(pil_img, q):
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

def ela_ensemble(pil_img):
    pil_img = pil_img.convert('RGB')
    arrs = [_ela_single(pil_img, q) for q in ELA_QUALS]
    ela = np.mean(arrs, axis=0)
    # сворачиваем в серый (L2 по каналам)
    ela_gray = np.sqrt(np.sum(ela**2, axis=2))
    ela_gray = (ela_gray - ela_gray.min()) / (ela_gray.ptp() + 1e-6)
    return ela_gray  # 0..1

def noise_map(pil_img, ksize=9):
    g = np.asarray(pil_img.convert('L')).astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(ksize, ksize))
    mean2 = cv2.boxFilter(g**2, ddepth=-1, ksize=(ksize, ksize))
    var = np.clip(mean2 - mean**2, 0, None)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var  # инверсия: «непохожие» зоны выше

def text_mask(pil_img):
    """Маска печатного текста для снижения ложных срабатываний."""
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

def fuse_scores(ela, noise, txt_mask_arr, w_ela=0.65, w_noise=0.35):
    score = w_ela * ela + w_noise * noise
    # приглушаем строки печатного текста (не влияет на печати/подписи)
    score = score * (1.0 - 0.6 * txt_mask_arr)
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score

def heatmap_overlay(pil_img, score, alpha=0.55):
    base = np.asarray(pil_img.convert('RGB')).astype(np.float32) / 255.0
    hm = (score * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0  # BGR->RGB
    out = (1 - alpha) * base + alpha * hm
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)

def block_report(score, block=BLOCK, top_k=8):
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

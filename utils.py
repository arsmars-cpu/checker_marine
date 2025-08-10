import io
import uuid
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import fitz

JPEG_QUALITIES = (90, 95, 98)
VAR_KERNEL = 9
MIN_AREA_RATIO = 0.002
MASK_MORPH = 3
PDF_DPI = 200  # Снижен для стабильности

RESULTS_DIR = Path(__file__).parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def local_variance(gray, k=9):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    return np.clip(sqmean - mean * mean, 0, None)


def ela_map(pil_img):
    pil_rgb = pil_img.convert("RGB")
    maps = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=q)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(pil_rgb, resaved)
        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
        maps.append(np.array(diff.convert("L")))
    return np.maximum.reduce(maps)


def suspicious_mask_from_background(ela_gray):
    ela_norm = cv2.GaussianBlur(ela_gray, (3, 3), 0)
    var = local_variance(ela_norm, VAR_KERNEL)
    var = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv = cv2.bitwise_not(var)
    med = np.median(inv)
    q1, q3 = np.percentile(inv, [25, 75])
    thr = int(min(255, med + 1.25 * max(1.0, q3 - q1)))
    _, mask = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)
    kernel = np.ones((MASK_MORPH, MASK_MORPH), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    h, w = mask.shape[:2]
    min_area = int(MIN_AREA_RATIO * w * h)
    out = np.zeros_like(mask)
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def summarize_mask(mask):
    total = mask.size
    pos = int((mask > 0).sum())
    pct = 100.0 * pos / max(1, total)
    regions = len(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    return pct, regions


def verdict_from_percent(p):
    if p >= 10.0:
        return "High (likely edited)"
    if p >= 3.0:
        return "Medium (possible edits)"
    return "Low (no clear edits)"


def overlay_suspicious(base_rgb, mask):
    overlay = base_rgb.copy()
    color = (0, 140, 255)
    alpha = 0.35
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(overlay)
    cv2.drawContours(fill, contours, -1, color, -1)
    overlay = cv2.addWeighted(overlay, 1.0, fill, alpha, 0)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


def render_pdf_to_images(pdf_path, dpi=PDF_DPI):
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
    return pages


def process_pil_image(pil, label, batch):
    ela = ela_map(pil)
    mask = suspicious_mask_from_background(ela)
    pct, regions = summarize_mask(mask)
    verdict = verdict_from_percent(pct)

    src = np.array(pil.convert("RGB"))
    ela_vis = cv2.applyColorMap(ela.astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = overlay_suspicious(src, mask)

    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"

    Image.fromarray(src).save(RESULTS_DIR / src_name, "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis, cv2.COLOR_BGR2RGB)).save(RESULTS_DIR / ela_name, "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(RESULTS_DIR / ovl_name, "JPEG", quality=95)

    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela": f"/static/results/{ela_name}",
        "overlay": f"/static/results/{ovl_name}",
        "verdict": verdict,
        "suspicious_percent": round(pct, 1),
        "regions": regions,
        "summary": f"{verdict} — suspicious area ≈ {pct:.1f}% across {regions} region(s)."
    }

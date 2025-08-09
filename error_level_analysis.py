
import io
import cv2
import numpy as np
from PIL import Image

def perform_ela_pil(pil_img, quality=90):
    """
    ELA-style map: re-save JPEG -> difference -> contrast stretch.
    Returns: (diff_image_PIL, stats_dict, boxes)
    """
    # Convert to JPEG bytes at given quality
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    re_jpeg = Image.open(buf)

    # Compute absolute difference
    diff = Image.fromarray(
        np.abs(np.asarray(pil_img.convert("RGB"), dtype=np.int16) - 
               np.asarray(re_jpeg.convert("RGB"), dtype=np.int16)).astype(np.uint8)
    )

    # Auto-contrast (percentile stretch)
    arr = np.asarray(diff, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    low, high = np.percentile(gray, [90, 99.9])
    if high <= low:
        high = low + 1
    scale = 255.0 / (high - low)
    stretched = np.clip((gray - low) * scale, 0, 255).astype(np.uint8)

    # Threshold to find "hot" areas
    thr_val = max(30, int(np.percentile(stretched, 98)))
    _, thresh = cv2.threshold(stretched, thr_val, 255, cv2.THRESH_BINARY)

    # Morphology to group regions
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and boxes
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = stretched.shape[:2]
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < max(200, (w*h)*0.002):  # filter tiny noise
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)})

    # Make a heat overlay visualization
    heat = cv2.applyColorMap(stretched, cv2.COLORMAP_INFERNO)
    # draw boxes
    for b in boxes:
        cv2.rectangle(heat, (b["x"], b["y"]), (b["x"]+b["w"], b["y"]+b["h"]), (0,255,0), 2)

    heat_pil = Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))

    # Stats
    hot_ratio = float(np.mean(stretched > thr_val))
    stats = {
        "hot_pixel_ratio": round(hot_ratio, 4),
        "threshold": int(thr_val),
        "boxes_count": len(boxes),
        "note": "Higher hot_pixel_ratio or multiple concentrated boxes may indicate local recompression/retouch."
    }
    return heat_pil, stats, boxes

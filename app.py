import os
import io
import uuid
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance, ImageFile
import numpy as np
import cv2
import fitz  # PyMuPDF

# -----------------------------------
# Безопасная загрузка больших картинок
# -----------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# ----------------- Конфиг -----------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB per request

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}

# Анализ
JPEG_QUALITIES = (90, 95, 98)   # ансамблевый ELA
VAR_KERNEL = 9                  # окно локальной дисперсии
MIN_AREA_RATIO = 0.002          # отсев мелких пятен
MASK_MORPH = 3                  # морфология
PDF_DPI = 200                   # рендер PDF
MAX_PDF_PAGES = 5               # максимум страниц
SCORE_PERCENTILE = 95           # «дефолт», но мы еще усилим адаптивом

# Визуал
ELA_USE_CLAHE = True

# Подсветка (рамки)
OVERLAY_CONTOUR_THICK = 3

# Хаускипинг / приватность
MAX_RESULT_AGE_HOURS = 3
MAX_LONG_EDGE = 4800

# ------------------------------------------------
# Helpers / утилиты
# ------------------------------------------------
def allowed_file(name: str) -> bool:
    return bool(name) and Path(name).suffix.lower() in ALLOWED_EXT

def cleanup_results_dir(max_age_hours: int = MAX_RESULT_AGE_HOURS):
    """Удаляем старые JPG/PDF из результатов: гигиена и приватность."""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for p in RESULTS_DIR.glob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
            continue
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass
    if removed:
        app.logger.info(f"Cleanup: removed {removed} old result file(s)")

def downscale_if_huge(pil_img: Image.Image, max_long_edge: int = MAX_LONG_EDGE) -> Image.Image:
    w, h = pil_img.size
    m = max(w, h)
    if m <= max_long_edge:
        return pil_img
    scale = max_long_edge / float(m)
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.LANCZOS)

# ------------------------------------------------
# Низкоуровневые карты: вариация/шум/ELA/текст/резкость
# ------------------------------------------------
def local_variance(gray: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var

def noise_map_from_gray(gray_u8: np.ndarray, k: int = VAR_KERNEL) -> np.ndarray:
    var = local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var  # 1 — «странно/непохоже»

def ela_map(pil_img: Image.Image) -> np.ndarray:
    """Ансамблевый ELA (0..255)."""
    pil_rgb = pil_img.convert("RGB")
    maps = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=q, optimize=True)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(pil_rgb, resaved)
        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
        maps.append(np.array(diff.convert("L")))
    return np.maximum.reduce(maps)

def text_mask(pil_img: Image.Image) -> np.ndarray:
    """Маска печатного текста (уменьшаем ложные срабатывания на шрифтах)."""
    g = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel1, iterations=1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    txt = cv2.dilate(txt, kernel2, 1)
    return txt  # 0/255

def blur_map(gray_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Карта резкости по дисперсии лапласиана, 0..1 (выше = резче)."""
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=ksize)
    v = lap * lap
    v = (v - v.min()) / (v.ptp() + 1e-6)
    return v

# ------------------------------------------------
# Сильнее скоринг: ELA ⊕ noise ⊕ sharp (с приглушением текста)
# ------------------------------------------------
def fused_score_plus(ela_gray_u8: np.ndarray, gray_u8: np.ndarray) -> np.ndarray:
    ela = ela_gray_u8.astype(np.float32) / 255.0
    # noise
    var = local_variance(gray_u8, k=VAR_KERNEL)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    noise = 1.0 - var
    # sharp
    sharp = blur_map(gray_u8, ksize=3)
    # смесь: усиливаем добор мелких правок
    score = 0.50 * ela + 0.25 * noise + 0.25 * sharp
    # приглушение печатного текста
    tmask = text_mask(Image.fromarray(gray_u8)).astype(np.float32) / 255.0
    score *= (1.0 - 0.5 * tmask)
    # нормировка
    score = (score - score.min()) / (score.ptp() + 1e-6)
    return score.astype(np.float32)

def adaptive_threshold(score: np.ndarray, perc: int = SCORE_PERCENTILE, k: float = 1.0) -> float:
    """Порог = max(перцентиль, mean + k*std)."""
    p = float(np.percentile(score, perc))
    m = float(score.mean())
    s = float(score.std())
    return max(p, m + k * s)

def two_stage_mask(score: np.ndarray,
                   min_area_ratio_big: float = 0.0015,
                   min_area_ratio_small: float = 0.0004,
                   morph_big: int = 4,
                   morph_small: int = 3) -> np.ndarray:
    """
    2-проходная сегментация: сначала крупное, затем добор мелочи внутри.
    Возвращает объединённую бинарную маску 0/255.
    """
    H, W = score.shape

    # 1) крупные области
    thr_big = adaptive_threshold(score, perc=95, k=1.0)
    m1 = (score >= thr_big).astype(np.uint8) * 255
    k1 = np.ones((morph_big, morph_big), np.uint8)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k1, 1)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN,  k1, 1)

    min_area_big = int(min_area_ratio_big * H * W)
    cnts1, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m1f = np.zeros_like(m1)
    for c in cnts1:
        if cv2.contourArea(c) >= min_area_big:
            cv2.drawContours(m1f, [c], -1, 255, -1)

    # 2) мелкие горячие точки внутри крупных
    thr_small = adaptive_threshold(score, perc=92, k=0.6)
    m2 = (score >= thr_small).astype(np.uint8) * 255
    k2 = np.ones((morph_small, morph_small), np.uint8)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k2, 1)
    m2 = cv2.bitwise_and(m2, m1f)  # оставляем только вокруг крупных аномалий

    min_area_small = int(min_area_ratio_small * H * W)
    cnts2, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m2f = np.zeros_like(m2)
    for c in cnts2:
        if cv2.contourArea(c) >= min_area_small:
            cv2.drawContours(m2f, [c], -1, 255, -1)

    out = cv2.bitwise_or(m1f, m2f)
    return out

# ------------------------------------------------
# ACID heat (кислотная теплокарта) + наложение
# ------------------------------------------------
def make_acid_heat(score: np.ndarray,
                   clip_low=0.88,   # всё ниже -> 0
                   clip_high=0.995, # верхний хвост растягиваем
                   gamma=0.6,
                   boost=1.8,
                   colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """
    На вход 0..1 score, на выходе BGR uint8 heat 0..255 «кислотный».
    """
    s = score.astype(np.float32)
    lo = float(np.quantile(s, clip_low))
    hi = float(np.quantile(s, clip_high))
    if hi <= lo:
        hi = lo + 1e-6
    s = (s - lo) / (hi - lo)
    s = np.clip(s, 0, 1)
    s = np.power(s, gamma) * boost
    s = np.clip(s, 0, 1)
    heat_u8 = (s * 255.0 + 0.5).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, colormap)

def apply_acid_overlay(base_rgb: np.ndarray,
                       heat_bgr: np.ndarray,
                       alpha_max=0.85,
                       base_desat=0.35) -> np.ndarray:
    """
    Фон приглушаем, hotspots накладываем с альфой, зависящей от «жары».
    """
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
    gray = cv2.cvtColor((base_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray3 = np.dstack([gray, gray, gray])
    base_soft = base_bgr * (1 - base_desat) + gray3 * base_desat

    heat = heat_bgr.astype(np.float32) / 255.0
    heat_v = cv2.cvtColor((heat * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = np.clip((heat_v ** 2) * alpha_max, 0, alpha_max)[..., None]

    out = base_soft * (1 - alpha) + heat * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

# ------------------------------------------------
# Постпроцесс / регионы
# ------------------------------------------------
def extract_regions(mask: np.ndarray, score_map: np.ndarray, max_regions: int = 6, pad: int = 8):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs = []
    H, W = mask.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi = score_map[y:y + h, x:x + w]
        sc = float(np.mean(roi)) if roi.size else 0.0
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "score": round(sc, 4)})
    regs.sort(key=lambda r: r["score"], reverse=True)
    return regs[:max_regions]

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

# ------------------------------------------------
# Светофор + отчёт
# ------------------------------------------------
def classify_severity(regions, score_map) -> tuple[str, str]:
    if not regions:
        return "green", "All good"
    max_region_score = max((r["score"] for r in regions), default=0.0)
    if max_region_score >= 0.60:
        return "red", "Likely edited"
    return "yellow", "Suspicious"

def _severity_color_rgb(sev: str):
    if sev == "red":    return (1, 0.27, 0.27)
    if sev == "yellow": return (1, 0.75, 0.2)
    return (0.2, 0.85, 0.55)

def _fitz_insert_image(page, rect, img_path: Path | None):
    try:
        if img_path:
            page.insert_image(rect, filename=str(img_path))
    except Exception:
        pass

def create_report_pdf(stem: str, label: str, severity: str, verdict_text: str, paths: dict) -> str:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 72dpi

    LM, TM, RM = 36, 36, 36
    y = TM

    title = "SeaDoc Checker — Forensic Report"
    color = _severity_color_rgb(severity)
    page.insert_text((LM, y), title, fontsize=16, color=color, fontname="helv")
    y += 24
    page.insert_text((LM, y), f"File: {label}", fontsize=10, color=(1, 1, 1), fontname="helv")
    y += 16
    page.insert_text((LM, y), f"Verdict: {verdict_text}", fontsize=12, color=color, fontname="helv")
    y += 18

    # Легенда
    chip_w, chip_h = 16, 10
    legend_x = LM
    legends = [("Green — OK", (0.2, 0.85, 0.55)),
               ("Yellow — Suspicious", (1, 0.75, 0.2)),
               ("Red — Likely Edited", (1, 0.27, 0.27))]
    for text, col in legends:
        page.draw_rect(fitz.Rect(legend_x, y, legend_x + chip_w, y + chip_h), color=col, fill=col)
        page.insert_text((legend_x + chip_w + 6, y + chip_h), text, fontsize=9, color=(0.9, 0.9, 0.95), fontname="helv")
        legend_x += 180
    y += 24

    # Сетки картинок
    col_w = (595 - LM - RM - 12) / 2
    row_h = 240
    r1 = fitz.Rect(LM, y, LM + col_w, y + row_h)                 # Original
    r2 = fitz.Rect(LM + col_w + 12, y, LM + 2 * col_w + 12, y + row_h)  # Highlighted
    y += row_h + 12
    r3 = fitz.Rect(LM, y, LM + 2 * col_w + 12, y + row_h)        # Key Region

    page.insert_text((r1.x0, r1.y0 - 4), "Original", fontsize=10, color=(0.8, 0.85, 0.95), fontname="helv")
    page.insert_text((r2.x0, r2.y0 - 4), "Highlighted", fontsize=10, color=(0.8, 0.85, 0.95), fontname="helv")
    page.insert_text((r3.x0, r3.y0 - 4), "Key Region", fontsize=10, color=(0.8, 0.85, 0.95), fontname="helv")

    _fitz_insert_image(page, r1, paths.get("original"))
    _fitz_insert_image(page, r2, paths.get("boxed") or paths.get("overlay"))
    _fitz_insert_image(page, r3, paths.get("lead_crop"))

    pdf_path = RESULTS_DIR / f"{stem}_report.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return f"/static/results/{pdf_path.name}"

# ------------------------------------------------
# Основной конвейер
# ------------------------------------------------
def process_pil_image(pil: Image.Image, label: str, batch: str) -> dict:
    # Даунскейл делаем снаружи (в /analyze), чтобы размеры везде совпали.

    # 1) ELA и карты в градациях серого
    ela = ela_map(pil)  # 0..255
    gray_u8 = np.array(pil.convert("L"))

    # 2) Усиленный скоринг
    score_map = fused_score_plus(ela, gray_u8)  # 0..1

    # 3) Двухпроходная маска (крупные + мелкие горячие точки)
    mask = two_stage_mask(score_map)

    # 4) Зоны
    regions = extract_regions(mask, score_map, max_regions=6, pad=8)

    # 5) Визуалы: кислотный режим
    src_rgb = np.array(pil.convert("RGB"))

    # ELA превью (acid)
    ela_u8 = ela.astype(np.uint8)
    if ELA_USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        ela_u8 = clahe.apply(ela_u8)
    ela_norm = ela_u8.astype(np.float32) / 255.0
    ela_heat_bgr = make_acid_heat(ela_norm, clip_low=0.85, clip_high=0.995, gamma=0.6, boost=1.6)

    # Основной overlay (acid)
    score_heat_bgr = make_acid_heat(score_map, clip_low=0.88, clip_high=0.995, gamma=0.6, boost=1.8)
    overlay_bgr = apply_acid_overlay(src_rgb, score_heat_bgr, alpha_max=0.85, base_desat=0.35)

    # Рамки поверх
    boxed_bgr = overlay_bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(boxed_bgr, (x, y), (x + w, y + h), (255, 255, 255), OVERLAY_CONTOUR_THICK)
        cv2.putText(boxed_bgr, f"#{idx}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # 6) Файлы
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"

    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_heat_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ela_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ovl_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / boxed_name), "JPEG", quality=95)

    # 7) Главный кроп
    crop_paths = []
    lead_crop_path = None
    if regions:
        H, W, _ = src_rgb.shape
        r0 = regions[0]
        x0, y0 = max(0, r0["x"]), max(0, r0["y"])
        x1, y1 = min(W, r0["x"] + r0["w"]), min(H, r0["y"] + r0["h"])
        crop = src_rgb[y0:y1, x0:x1]
        if crop.size > 0:
            crop_name = f"{stem}_lead_crop.jpg"
            Image.fromarray(crop).save(str(RESULTS_DIR / crop_name), "JPEG", quality=95)
            lead_crop_path = RESULTS_DIR / crop_name
            crop_paths.append({
                "index": 1,
                "score": r0["score"],
                "url": f"/static/results/{crop_name}",
                "box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
            })

    # 8) Светофор + отчёт
    severity, verdict_txt = classify_severity(regions, score_map)
    report_url = create_report_pdf(
        stem=stem,
        label=label,
        severity=severity,
        verdict_text=verdict_txt,
        paths={
            "original": RESULTS_DIR / src_name,
            "boxed":    RESULTS_DIR / boxed_name,
            "overlay":  RESULTS_DIR / ovl_name,
            "lead_crop": lead_crop_path
        }
    )

    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela":      f"/static/results/{ela_name}",
        "overlay":  f"/static/results/{ovl_name}",
        "boxed":    f"/static/results/{boxed_name}",
        "severity": severity,
        "verdict":  verdict_txt,
        "regions":  len(regions),
        "crops":    crop_paths,
        "report":   report_url,
        "summary":  "Check highlighted areas." if regions else "No anomalies detected."
    }

# ------------------------------------------------
# Роуты
# ------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # UI уже без слайдеров — один режим «acid + traffic light»
    return render_template("index.html", results=[])

@app.route("/analyze", methods=["POST"])
def analyze():
    cleanup_results_dir()

    files = request.files.getlist("file")
    if not files:
        return render_template("index.html", results=[], message="No files uploaded.")

    batch = uuid.uuid4().hex[:8]
    results = []

    for f in files:
        if not f or not allowed_file(f.filename):
            continue
        fname = secure_filename(f.filename)
        ext = Path(fname).suffix.lower()
        upath = UPLOAD_DIR / f"{batch}_{fname}"
        upath.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(upath))

        try:
            size = upath.stat().st_size
            app.logger.info(f"Upload: {fname} | {size} bytes | ext={ext}")
        except Exception:
            pass

        try:
            if ext == ".pdf":
                app.logger.info(f"PDF render start {fname} @ {PDF_DPI}dpi (max {MAX_PDF_PAGES} pages)")
                for i, pimg in iter_pdf_pages(upath, dpi=PDF_DPI, max_pages=MAX_PDF_PAGES):
                    pimg = downscale_if_huge(pimg)
                    res = process_pil_image(pimg, f"{fname} — page {i}", batch)
                    results.append(res)
                app.logger.info(f"PDF render done {fname}: {len(results)} page(s) processed in batch {batch}")
            else:
                pil = Image.open(str(upath)).convert("RGB")
                pil = downscale_if_huge(pil)
                res = process_pil_image(pil, fname, batch)
                results.append(res)
        except Exception as e:
            app.logger.exception(f"Analyze error: {fname} | {e}")
            results.append({
                "label": fname,
                "original": f"/static/uploads/{batch}_{fname}",
                "ela": "",
                "overlay": "",
                "boxed": "",
                "severity": "yellow",
                "verdict": "Analysis error",
                "regions": 0,
                "crops": [],
                "report": None,
                "summary": f"Error during analysis: {e}"
            })

    return render_template("index.html", results=results, message=f"Batch {batch}: processed {len(results)} item(s).")

@app.post("/api/analyze")
def api_analyze():
    cleanup_results_dir()

    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "error": "no files"}), 400

    batch = uuid.uuid4().hex[:8]
    out = []
    for f in files:
        if not f or not allowed_file(f.filename):
            continue
        fname = secure_filename(f.filename)
        ext = Path(fname).suffix.lower()
        try:
            if ext == ".pdf":
                data = io.BytesIO(f.read()); data.seek(0)
                with fitz.open(stream=data.read(), filetype="pdf") as doc:
                    pages = min(len(doc), MAX_PDF_PAGES)
                    scale = PDF_DPI / 72.0
                    mat = fitz.Matrix(scale, scale)
                    for i in range(pages):
                        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
                        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        pil = downscale_if_huge(pil)
                        res = process_pil_image(pil, f"{fname} — page {i+1}", batch)
                        out.append(res)
            else:
                pil = Image.open(f.stream).convert("RGB")
                pil = downscale_if_huge(pil)
                res = process_pil_image(pil, fname, batch)
                out.append(res)
        except Exception as e:
            app.logger.exception(f"/api analyze error: {fname} | {e}")
            out.append({
                "label": fname,
                "original": None,
                "ela": "",
                "overlay": "",
                "boxed": "",
                "severity": "yellow",
                "verdict": "Analysis error",
                "regions": 0,
                "crops": [],
                "report": None,
                "summary": f"PDF/image render error: {e}"
            })

    return jsonify({"ok": True, "batch": batch, "results": out})

@app.get("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

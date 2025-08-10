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

# Безопасная загрузка больших изображений
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

# Подкручиваемые параметры анализа/визуала
JPEG_QUALITIES = (90, 95, 98)        # ансамблевый ELA
VAR_KERNEL = 9                        # окно локальной дисперсии
MIN_AREA_RATIO = 0.002                # отсев мелких пятен по площади
MASK_MORPH = 3                        # морфология для очистки маски
PDF_DPI = 200                         # рендер PDF-страниц
MAX_PDF_PAGES = 5                     # максимум страниц на анализ
SCORE_PERCENTILE = 95                 # дефолт чувствительность (процентиль)

# Усиление ELA-визуала
ELA_GAIN = 1.4                        # 1.0..1.8
ELA_GAMMA = 0.85                      # 0.75..0.95 (ниже -> светлее)
ELA_USE_CLAHE = True
ELA_COLORMAP = cv2.COLORMAP_TURBO

# Подсветка подозрительных зон (оверлей)
OVERLAY_ALPHA = 0.50
OVERLAY_CONTOUR_THICK = 3

# Хаускипинг / приватность
MAX_RESULT_AGE_HOURS = 3              # авто-удаление результатов (мы не храним данные)
MAX_LONG_EDGE = 4800                  # ограничение по длинной стороне для стабильности

# ----------------- Утилиты/Helpers -----------------
def _clamp_float(val, lo, hi, default):
    try:
        v = float(val)
        return max(lo, min(hi, v))
    except Exception:
        return default

def _clamp_int(val, lo, hi, default):
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except Exception:
        return default

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

# ----------------- Низкоуровневые карты -----------------
def local_variance(gray: np.ndarray, k: int = 9) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var

def noise_map_from_gray(gray_u8: np.ndarray, k: int = 9) -> np.ndarray:
    """Карта «непохожести» локального шума: 1 — странно, 0 — нормально."""
    var = local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var

def ela_map(pil_img: Image.Image) -> np.ndarray:
    """Ансамблевый ELA: максимум по нескольким JPEG-качествам, с автоусилением (0..255)."""
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
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        maps.append(np.array(diff.convert("L")))
    ela = np.maximum.reduce(maps)  # 0..255 (uint8)
    return ela

def text_mask(pil_img: Image.Image) -> np.ndarray:
    """Маска печатного текста (уменьшаем ложные срабатывания на шрифтах)."""
    g = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return txt  # 0/255

# ----------------- Визуал/постпроцесс -----------------
def overlay_suspicious(base_bgr: np.ndarray, mask: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """Оверлей полупрозрачной заливки + красный контур по подозрительным зонам."""
    overlay = base_bgr.copy()
    fill_color = (0, 140, 255)  # BGR
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(overlay)
    cv2.drawContours(fill, contours, -1, fill_color, thickness=-1)
    overlay = cv2.addWeighted(overlay, 1.0, fill, alpha, 0)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=OVERLAY_CONTOUR_THICK)
    return overlay

def extract_regions(mask: np.ndarray, score_map: np.ndarray, max_regions: int = 6, pad: int = 8):
    """Возвращает топ-зоны [{x,y,w,h,score}] по среднему скору."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs = []
    H, W = mask.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi = score_map[y:y+h, x:x+w]
        sc = float(np.mean(roi)) if roi.size else 0.0
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        regs.append({"x": int(x0), "y": int(y0), "w": int(x1-x0), "h": int(y1-y0), "score": round(sc, 4)})
    regs.sort(key=lambda r: r["score"], reverse=True)
    return regs[:max_regions]

def iter_pdf_pages(pdf_path: Path, dpi: int = PDF_DPI, max_pages: int = MAX_PDF_PAGES):
    """Памяти-бережный рендер PDF: отдаём по одной PIL-странице."""
    with fitz.open(str(pdf_path)) as doc:
        pages = min(len(doc), max_pages)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i + 1, img

# ----------------- Светофор и PDF-отчёт -----------------
def classify_severity(regions, score_map) -> tuple[str, str]:
    """
    'green' | 'yellow' | 'red' + лаконичный текст без процентов.
    Логика: нет зон -> green; иначе по силе аномалии в лучшей зоне.
    """
    if not regions:
        return "green", "All good"
    max_region_score = max((r["score"] for r in regions), default=0.0)
    if max_region_score >= 0.60:
        return "red", "Likely edited"
    return "yellow", "Suspicious"

def _severity_color_rgb(sev: str):
    if sev == "red":
        return (1, 0.27, 0.27)
    if sev == "yellow":
        return (1, 0.75, 0.2)
    return (0.2, 0.85, 0.55)  # green

def _fitz_insert_image(page, rect, img_path: Path | None):
    try:
        if img_path:
            page.insert_image(rect, filename=str(img_path))
    except Exception:
        pass

def create_report_pdf(stem: str, label: str, severity: str, verdict_text: str, paths: dict) -> str:
    """
    Собирает PDF-отчёт (A4): Original, Highlighted, Key Region, легенда.
    Возвращает web-URL сохранённого PDF.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 72dpi

    LM, TM, RM, BM = 36, 36, 36, 36
    y = TM

    title = "SeaDoc Checker — Forensic Report"
    color = _severity_color_rgb(severity)
    page.insert_text((LM, y), title, fontsize=16, color=color, fontname="helv")
    y += 24
    page.insert_text((LM, y), f"File: {label}", fontsize=10, color=(1, 1, 1), fontname="helv")
    y += 16
    page.insert_text((LM, y), f"Verdict: {verdict_text}", fontsize=12, color=color, fontname="helv")
    y += 18

    # Легенда светофора
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

# ----------------- Основной конвейер -----------------
def process_pil_image(
    pil: Image.Image,
    label: str,
    batch: str,
    *,
    ela_gain: float = ELA_GAIN,
    overlay_alpha: float = OVERLAY_ALPHA,
    score_percentile: int = SCORE_PERCENTILE
) -> dict:
    # ВАЖНО: даунскейл мы делаем ВНЕ этой функции (в /analyze), чтобы
    # маска и оверлеи всегда считались на одном размере.

    # 1) ELA
    ela = ela_map(pil)

    # 2) score_map = ELA/Noise/Text (0..1)
    ela_f = (ela.astype(np.float32) - ela.min()) / (ela.ptp() + 1e-6)
    gray = np.array(pil.convert("L")).astype(np.float32)
    gray = (gray - gray.min()) / (gray.ptp() + 1e-6)
    noise = noise_map_from_gray((gray * 255).astype(np.uint8), k=VAR_KERNEL)
    score_map = 0.65 * ela_f + 0.35 * noise
    tmask = text_mask(pil)  # 0/255
    score_map = score_map * (1.0 - 0.5 * (tmask.astype(np.float32) / 255.0))
    score_map = (score_map - score_map.min()) / (score_map.ptp() + 1e-6)

    # 3) Бинаризация по перцентилю + морфология + отсев шумов
    thr_dyn = float(np.percentile(score_map, score_percentile))
    mask = (score_map >= thr_dyn).astype(np.uint8) * 255
    kernel = np.ones((MASK_MORPH, MASK_MORPH), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    h, w = mask.shape[:2]
    min_area = int(MIN_AREA_RATIO * w * h)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_f = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask_f, [c], -1, 255, -1)
    mask = mask_f

    # 4) Зоны
    regions = extract_regions(mask, score_map, max_regions=6, pad=8)

    # 5) Визуалы
    src_rgb = np.array(pil.convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)

    # Яркий ELA-визуал
    ela_u8 = ela.astype(np.uint8)
    if ELA_USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        ela_u8 = clahe.apply(ela_u8)
    ela_fv = np.clip((ela_u8.astype(np.float32) / 255.0) * ela_gain, 0, 1)
    ela_fv = np.power(ela_fv, ELA_GAMMA)
    ela_u8 = (ela_fv * 255.0 + 0.5).astype(np.uint8)
    ela_vis_bgr = cv2.applyColorMap(ela_u8, ELA_COLORMAP)

    overlay_bgr = overlay_suspicious(src_bgr, mask, alpha=overlay_alpha)
    boxed_bgr = overlay_bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(boxed_bgr, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(
            boxed_bgr, f"#{idx}", (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1, cv2.LINE_AA
        )

    # 6) Файлы
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"

    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ela_name), "JPEG", quality=95)
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
        "severity": severity,              # 'green'|'yellow'|'red'
        "verdict":  verdict_txt,           # короткий текст без процентов
        "regions":  len(regions),
        "crops":    crop_paths,            # главный кроп
        "report":   report_url,            # ссылка на PDF-отчёт
        "summary":  "Check highlighted areas." if regions else "No anomalies detected."
    }

# ----------------- Роуты -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=[])

@app.route("/analyze", methods=["POST"])
def analyze():
    cleanup_results_dir()  # авто-очистка старых результатов

    # Настройки из формы (ползунки). Если нет — дефолты.
    ela_gain = _clamp_float(request.form.get("ela_gain"), 1.0, 1.8, ELA_GAIN)
    overlay_alpha = _clamp_float(request.form.get("overlay_alpha"), 0.2, 0.8, OVERLAY_ALPHA)
    score_percentile = _clamp_int(request.form.get("score_percentile"), 88, 98, SCORE_PERCENTILE)

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
                    res = process_pil_image(
                        pimg, f"{fname} — page {i}", batch,
                        ela_gain=ela_gain, overlay_alpha=overlay_alpha, score_percentile=score_percentile
                    )
                    results.append(res)
                app.logger.info(f"PDF render done {fname}: {len(results)} page(s) processed in batch {batch}")
            else:
                pil = Image.open(str(upath)).convert("RGB")
                pil = downscale_if_huge(pil)
                res = process_pil_image(
                    pil, fname, batch,
                    ela_gain=ela_gain, overlay_alpha=overlay_alpha, score_percentile=score_percentile
                )
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

    ela_gain = _clamp_float(request.form.get("ela_gain"), 1.0, 1.8, ELA_GAIN)
    overlay_alpha = _clamp_float(request.form.get("overlay_alpha"), 0.2, 0.8, OVERLAY_ALPHA)
    score_percentile = _clamp_int(request.form.get("score_percentile"), 88, 98, SCORE_PERCENTILE)

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
                        res = process_pil_image(
                            pil, f"{fname} — page {i+1}", batch,
                            ela_gain=ela_gain, overlay_alpha=overlay_alpha, score_percentile=score_percentile
                        )
                        out.append(res)
            else:
                pil = Image.open(f.stream).convert("RGB")
                pil = downscale_if_huge(pil)
                res = process_pil_image(
                    pil, fname, batch,
                    ela_gain=ela_gain, overlay_alpha=overlay_alpha, score_percentile=score_percentile
                )
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

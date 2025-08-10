import os
import io
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import fitz  # PyMuPDF

# ----------------- Конфиг -----------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB per request

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}

# Порог/параметры (можно подкручивать)
JPEG_QUALITIES = (90, 95, 98)     # ансамблевый ELA
VAR_KERNEL = 9                    # окно для локальной дисперсии
MIN_AREA_RATIO = 0.002            # минимальная площадь пятна (0.2% от страницы)
MASK_MORPH = 3                    # морфология для очистки маски (тоньше)
PDF_DPI = 200                     # рендер PDF (понизили, чтобы не падала память)
MAX_PDF_PAGES = 5                 # анализируем максимум N страниц из PDF

# ----------------- Утилиты -----------------
def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT

def local_variance(gray: np.ndarray, k: int = 9) -> np.ndarray:
    """Локальная дисперсия по окну k×k."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var

def noise_map_from_gray(gray_u8: np.ndarray, k: int = 9) -> np.ndarray:
    """
    Карта «непохожести» локального шума: 1 - странно, 0 - нормально.
    Основана на инверсии нормализованной локальной дисперсии.
    """
    var = local_variance(gray_u8, k)
    var = (var - var.min()) / (var.ptp() + 1e-6)
    return 1.0 - var

def ela_map(pil_img: Image.Image) -> np.ndarray:
    """
    Ансамблевый ELA: карты для нескольких JPEG-качеств, берём покомпонентный максимум.
    Возвращает ELA-грей ndarray (0..255).
    """
    pil_rgb = pil_img.convert("RGB")
    maps = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=q, optimize=True)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(pil_rgb, resaved)

        # автоусиление
        extrema = diff.getextrema()
        max_diff = max(e[1] for e in extrema) or 1
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        maps.append(np.array(diff.convert("L")))

    ela = np.maximum.reduce(maps)  # покомпонентный максимум
    return ela

def text_mask(pil_img: Image.Image) -> np.ndarray:
    """
    Маска печатного текста/тонких штрихов, чтобы уменьшить ложные срабатывания.
    Возвращает uint8 (0/255).
    """
    g = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    # подчистим тонкие штрихи шрифта и немного расширим
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    txt = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
    return txt

def suspicious_mask_from_background(ela_gray: np.ndarray, pil_for_text: Image.Image) -> np.ndarray:
    """
    Комбинированный скоринг:
      1) score = 0.65*ELA_norm + 0.35*noise_map
      2) приглушаем печатный текст
      3) берём верхний перцентиль (95-й) -> бинарная маска
      4) морфология + отсев по площади
    """
    # ELA -> [0..1]
    ela_f = ela_gray.astype(np.float32)
    ela_f = (ela_f - ela_f.min()) / (ela_f.ptp() + 1e-6)

    # шум с исходного грая
    gray = np.array(pil_for_text.convert("L")).astype(np.float32)
    gray = (gray - gray.min()) / (gray.ptp() + 1e-6)
    noise = noise_map_from_gray((gray * 255).astype(np.uint8), k=VAR_KERNEL)

    # скор
    score = 0.65 * ela_f + 0.35 * noise

    # приглушаем печатный текст (но не штампы/подписи)
    tmask = text_mask(pil_for_text)  # 0/255
    score = score * (1.0 - 0.5 * (tmask.astype(np.float32) / 255.0))

    # адаптивный порог по перцентилю
    p95 = float(np.percentile(score, 95))
    mask = (score >= p95).astype(np.uint8) * 255

    # морфология
    kernel = np.ones((MASK_MORPH, MASK_MORPH), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # отсев по площади
    h, w = mask.shape[:2]
    min_area = int(MIN_AREA_RATIO * w * h)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out, [c], -1, 255, thickness=-1)
    return out

def summarize_mask(mask: np.ndarray) -> tuple[float, int]:
    total = mask.size
    pos = int((mask > 0).sum())
    pct = 100.0 * pos / max(1, total)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return pct, len(contours)

def verdict_from_percent(p: float) -> str:
    if p >= 10.0:
        return "High (likely edited)"
    if p >= 3.0:
        return "Medium (possible edits)"
    return "Low (no clear edits)"

def overlay_suspicious(base_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Рисуем полупрозрачную заливку по подозрительным зонам + контур.
    base_bgr — изображение в цветовом пространстве BGR.
    """
    overlay = base_bgr.copy()
    color = (0, 140, 255)  # BGR
    alpha = 0.35

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # заливка
    fill = np.zeros_like(overlay)
    cv2.drawContours(fill, contours, -1, color, thickness=-1)
    overlay = cv2.addWeighted(overlay, 1.0, fill, alpha, 0)

    # обводка
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)
    return overlay

def iter_pdf_pages(pdf_path: Path, dpi: int = PDF_DPI, max_pages: int = MAX_PDF_PAGES):
    """
    Памяти-бережный рендер PDF: отдаём по одной PIL-странице.
    """
    with fitz.open(str(pdf_path)) as doc:
        pages = min(len(doc), max_pages)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield i + 1, img

def process_pil_image(pil: Image.Image, label: str, batch: str) -> dict:
    """
    Основной конвейер:
      PIL -> ELA (ансамбль) -> комбинированная маска (ELA+noise, -text) ->
      проценты -> вердикт -> картинки
    """
    # 1) ELA
    ela = ela_map(pil)

    # 2) подозрительная маска по комбинированному скору
    mask = suspicious_mask_from_background(ela, pil)

    # 3) проценты + аномалии
    pct, regions = summarize_mask(mask)
    verdict = verdict_from_percent(pct)

    # 4) готовим картинки для UI
    src_rgb = np.array(pil.convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    ela_vis_bgr = cv2.applyColorMap(ela.astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay_bgr = overlay_suspicious(src_bgr, mask)

    # 5) имена файлов
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"

    # 6) сохраняем на диск
    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)).save(
        str(RESULTS_DIR / ela_name), "JPEG", quality=95
    )
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(
        str(RESULTS_DIR / ovl_name), "JPEG", quality=95
    )

    # 7) web-пути + плоский summary
    summary_text = f"{verdict} — suspicious area ≈ {pct:.1f}% across {regions} region(s)."

    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela":      f"/static/results/{ela_name}",
        "overlay":  f"/static/results/{ovl_name}",
        "verdict": verdict,
        "suspicious_percent": round(pct, 1),
        "regions": regions,
        "summary": summary_text,
    }

# ----------------- Роуты -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=[])

@app.route("/analyze", methods=["POST"])
def analyze():
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

        # лог размера файла
        try:
            size = upath.stat().st_size
            app.logger.info(f"Upload: {fname} | {size} bytes | ext={ext}")
        except Exception:
            pass

        try:
            if ext == ".pdf":
                app.logger.info(f"PDF render start {fname} @ {PDF_DPI}dpi (max {MAX_PDF_PAGES} pages)")
                for i, pimg in iter_pdf_pages(upath, dpi=PDF_DPI, max_pages=MAX_PDF_PAGES):
                    res = process_pil_image(pimg, f"{fname} — page {i}", batch)
                    results.append(res)
                app.logger.info(f"PDF render done {fname}: {len(results)} page(s) processed in batch {batch}")
            else:
                pil = Image.open(str(upath)).convert("RGB")
                res = process_pil_image(pil, fname, batch)
                results.append(res)
        except Exception as e:
            app.logger.exception(f"Analyze error: {fname} | {e}")
            results.append({
                "label": fname,
                "original": f"/static/uploads/{batch}_{fname}",
                "ela": "",
                "overlay": "",
                "verdict": "Error",
                "suspicious_percent": 0.0,
                "regions": 0,
                "summary": f"Error during analysis: {e}"
            })

    return render_template("index.html", results=results, message=f"Batch {batch}: processed {len(results)} item(s).")

# API для бота/интеграций
@app.post("/api/analyze")
def api_analyze():
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

        # читаем в память и в PIL
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
                        res = process_pil_image(pil, f"{fname} — page {i+1}", batch)
                        out.append(res)
            else:
                pil = Image.open(f.stream).convert("RGB")
                res = process_pil_image(pil, fname, batch)
                out.append(res)
        except Exception as e:
            app.logger.exception(f"/api analyze error: {fname} | {e}")
            out.append({
                "label": fname,
                "original": None,
                "ela": "",
                "overlay": "",
                "verdict": "Error",
                "suspicious_percent": 0.0,
                "regions": 0,
                "summary": f"PDF/image render error: {e}"
            })

    return jsonify({"ok": True, "batch": batch, "results": out})

@app.get("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

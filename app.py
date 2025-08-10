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

def verdict_text(has_regions: bool) -> str:
    return "Review recommended" if has_regions else "No issues found"

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

def extract_regions(mask: np.ndarray, score_map: np.ndarray, max_regions: int = 6, pad: int = 8):
    """
    Возвращает топ-зоны [{x,y,w,h,score}], отсортированные по среднему скору.
    """
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
      PIL -> ELA -> комбинированная маска (ELA+noise, -text) ->
      извлечение зон -> вердикт -> изображения (overlay, boxed, crops)
    """
    # 1) ELA
    ela = ela_map(pil)

    # 2) маска по комбинированному скору
    mask = suspicious_mask_from_background(ela, pil)

    # 2b) score_map (нужен для ранжирования зон)
    ela_f = (ela.astype(np.float32) - ela.min()) / (ela.ptp() + 1e-6)
    gray = np.array(pil.convert("L")).astype(np.float32)
    gray = (gray - gray.min()) / (gray.ptp() + 1e-6)
    noise = noise_map_from_gray((gray * 255).astype(np.uint8), k=VAR_KERNEL)
    score_map = 0.65 * ela_f + 0.35 * noise
    tmask = text_mask(pil)
    score_map = score_map * (1.0 - 0.5 * (tmask.astype(np.float32) / 255.0))

    # 3) извлекаем топ-зоны
    regions = extract_regions(mask, score_map, max_regions=6, pad=8)

    # 4) готовим картинки
    src_rgb = np.array(pil.convert("RGB"))
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    ela_vis_bgr = cv2.applyColorMap(ela.astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay_bgr = overlay_suspicious(src_bgr, mask)

    # overlay + рамки + индексы
    boxed_bgr = overlay_bgr.copy()
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(boxed_bgr, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(boxed_bgr, f"#{idx} {r['score']:.2f}", (x, max(15, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)

    # 5) имена файлов
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"
    boxed_name = f"{stem}_boxed.jpg"

    # 6) сохраняем на диск
    Image.fromarray(src_rgb).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis_bgr, cv2.COLOR_BGR2RGB)).save(
        str(RESULTS_DIR / ela_name), "JPEG", quality=95
    )
    Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)).save(
        str(RESULTS_DIR / ovl_name), "JPEG", quality=95
    )
    Image.fromarray(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB)).save(
        str(RESULTS_DIR / boxed_name), "JPEG", quality=95
    )

    # 7) сохраняем кропы зон
    crop_paths = []
    H, W, _ = src_rgb.shape
    for idx, r in enumerate(regions, start=1):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = src_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop_name = f"{stem}_crop_{idx}.jpg"
        Image.fromarray(crop).save(str(RESULTS_DIR / crop_name), "JPEG", quality=95)
        crop_paths.append({
            "index": idx,
            "score": r["score"],
            "url": f"/static/results/{crop_name}",
            "box": {"x": x0, "y": y0, "w": x1-x0, "h": y1-y0}
        })

    # 8) итоговый вердикт (консервативный)
    has_regions = len(regions) > 0
    verdict = verdict_text(has_regions)

    # 9) проценты по площади (как вспомогательная метрика)
    pct, _ = summarize_mask(mask)

    # 10) ответ
    return {
        "label": label,
        "original": f"/static/results/{src_name}",
        "ela":      f"/static/results/{ela_name}",
        "overlay":  f"/static/results/{ovl_name}",
        "boxed":    f"/static/results/{boxed_name}",
        "verdict": verdict,
        "suspicious_percent": round(pct, 1),
        "regions": len(regions),
        "crops": crop_paths,
        "summary": f"Top {len(regions)} region(s) highlighted. Use thumbnails to review. Suspicious area ≈ {pct:.1f}%."
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
                "boxed": "",
                "verdict": "Error",
                "suspicious_percent": 0.0,
                "regions": 0,
                "crops": [],
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
                "boxed": "",
                "verdict": "Error",
                "suspicious_percent": 0.0,
                "regions": 0,
                "crops": [],
                "summary": f"PDF/image render error: {e}"
            })

    return jsonify({"ok": True, "batch": batch, "results": out})

@app.get("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

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

# Порог/параметры (подкручиваемые)
JPEG_QUALITIES = (90, 95, 98)     # ансамблевый ELA
VAR_KERNEL = 9                    # окно для локальной дисперсии
MIN_AREA_RATIO = 0.002            # минимальная площадь пятна (0.2% от страницы)
MASK_MORPH = 3                    # морфология для очистки маски
PDF_DPI = 300                     # рендер PDF (выше — точнее, но дольше)


# ----------------- Утилиты -----------------
def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT


def local_variance(gray: np.ndarray, k: int = 9) -> np.ndarray:
    # дисперсия по окну k×k
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    sqmean = cv2.blur(gray_f * gray_f, (k, k))
    var = np.clip(sqmean - mean * mean, 0, None)
    return var


def ela_map(pil_img: Image.Image) -> np.ndarray:
    """
    Ансамблевый ELA: строим карты для нескольких JPEG-качеств и берём покомпонентный максимум.
    Возвращает ELA-грей ndarray (0..255).
    """
    pil_rgb = pil_img.convert("RGB")
    maps = []
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        pil_rgb.save(buf, "JPEG", quality=q)
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


def suspicious_mask_from_background(ela_gray: np.ndarray) -> np.ndarray:
    """
    Выделяем именно однородные/плоские пятна фона.
    1) Строим карту локальной дисперсии (на исходном ELA).
    2) Адаптивный порог: берём клетки с низкой вариативностью.
    3) Чистим морфологией + фильтруем по площади.
    """
    ela_norm = cv2.GaussianBlur(ela_gray, (3, 3), 0)

    # локальная дисперсия — «пластилиновые» патчи будут с низкой дисперсией
    var = local_variance(ela_norm, VAR_KERNEL)
    var = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # инвертируем: низкая вариативность -> высокая маска
    inv = cv2.bitwise_not(var)

    # порог адаптивный по статистике кадра
    med = np.median(inv)
    q1, q3 = np.percentile(inv, [25, 75])
    iqr = max(1.0, q3 - q1)
    thr = int(min(255, med + 1.25 * iqr))  # можно подкрутить 1.25..1.75
    _, mask = cv2.threshold(inv, thr, 255, cv2.THRESH_BINARY)

    # морф. очистка
    kernel = np.ones((MASK_MORPH, MASK_MORPH), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # фильтр по минимальной площади
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


def overlay_suspicious(base_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Рисуем полупрозрачную заливку по подозрительным зонам + контур.
    """
    overlay = base_rgb.copy()
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


def render_pdf_to_images(pdf_path: Path, dpi: int = PDF_DPI) -> list[Image.Image]:
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
    return pages


def process_pil_image(pil: Image.Image, label: str, batch: str) -> dict:
    """
    Основной конвейер:
      PIL -> ELA (ансамбль) -> маска ровных пятен -> проценты -> verdict -> картинки
    """
    # 1) ELA
    ela = ela_map(pil)

    # 2) подозрительная маска (только по фону)
    mask = suspicious_mask_from_background(ela)

    # 3) проценты + аномалии
    pct, regions = summarize_mask(mask)
    verdict = verdict_from_percent(pct)

    # 4) готовим картинки для UI
    src = np.array(pil.convert("RGB"))
    ela_vis = cv2.applyColorMap(ela.astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = overlay_suspicious(src, mask)

    # 5) имена файлов
    stem = f"{batch}_{uuid.uuid4().hex[:6]}"
    src_name = f"{stem}_src.jpg"
    ela_name = f"{stem}_ela.jpg"
    ovl_name = f"{stem}_overlay.jpg"

    # 6) сохраняем на диск
    Image.fromarray(src).save(str(RESULTS_DIR / src_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(ela_vis, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ela_name), "JPEG", quality=95)
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(str(RESULTS_DIR / ovl_name), "JPEG", quality=95)

    # 7) web‑пути + плоский summary
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

        try:
            if ext == ".pdf":
                pages = render_pdf_to_images(upath, dpi=PDF_DPI)
                for i, pimg in enumerate(pages, start=1):
                    res = process_pil_image(pimg, f"{fname} — page {i}", batch)
                    results.append(res)
            else:
                pil = Image.open(str(upath)).convert("RGB")
                res = process_pil_image(pil, fname, batch)
                results.append(res)
        except Exception as e:
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
                tmp = io.BytesIO(f.read())
                tmp.seek(0)
                doc = fitz.open(stream=tmp.read(), filetype="pdf")
                for i, page in enumerate(doc, start=1):
                    pix = page.get_pixmap(dpi=PDF_DPI, alpha=False)
                    pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    res = process_pil_image(pil, f"{fname} — page {i}", batch)
                    out.append(res)
            else:
                pil = Image.open(f.stream).convert("RGB")
                res = process_pil_image(pil, fname, batch)
                out.append(res)
        except Exception as e:
            out.append({
                "label": fname,
                "original": None,
                "ela": "",
                "overlay": "",
                "verdict": "Error",
                "suspicious_percent": 0.0,
                "regions": 0,
                "summary": f"Error during analysis: {e}"
            })

    return jsonify({"ok": True, "batch": batch, "results": out})


@app.get("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

import os
import io
import uuid
import time
import traceback
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
import fitz  # PyMuPDF

# Новый пайплайн
from error_level_analysis import run_image
# Только генератор страниц PDF (если используешь)
from utils import iter_pdf_pages

# --------- конфиг ---------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}

# безопасная загрузка больших изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# housekeeping
MAX_RESULT_AGE_HOURS = 3
MAX_LONG_EDGE = 4800


def allowed_file(name: str) -> bool:
    return bool(name) and Path(name).suffix.lower() in ALLOWED_EXT


def cleanup_results_dir(max_age_hours: int = MAX_RESULT_AGE_HOURS):
    cutoff = time.time() - max_age_hours * 3600
    for p in RESULTS_DIR.glob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception:
            pass


def downscale_if_huge(pil_img: Image.Image, max_long_edge: int = MAX_LONG_EDGE) -> Image.Image:
    w, h = pil_img.size
    m = max(w, h)
    if m <= max_long_edge:
        return pil_img
    scale = max_long_edge / float(m)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def make_error_result(label: str, uploaded_path_web: str | None, err: Exception) -> dict:
    app.logger.error("Analysis error for %s: %s\n%s",
                     label, err, traceback.format_exc())
    return {
        "label": label,
        "original": uploaded_path_web,
        "ela": "",
        "overlay": "",
        "boxed": "",
        "verdict": "Analysis error",
        "severity": "red",
        "regions": 0,
        "crops": [],
        "summary": f"Error during analysis: {err}"
    }


# --------- роуты ---------
@app.get("/")
def index():
    return render_template("index.html", results=[])


@app.post("/analyze")
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
        upath = UPLOAD_DIR / f"{batch}_{fname}"
        upath.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(upath))
        uploaded_web = f"/static/uploads/{batch}_{fname}"

        try:
            ext = Path(fname).suffix.lower()
            if ext == ".pdf":
                # если используешь utils.iter_pdf_pages — он удобнее и быстрее
                page_count = 0
                for i, pimg in iter_pdf_pages(upath):
                    page_count += 1
                    pimg = downscale_if_huge(pimg.convert("RGB"))
                    results.append(
                        run_image(pimg, f"{fname} — page {i}", batch, RESULTS_DIR)
                    )
                if page_count == 0:
                    raise RuntimeError("Empty PDF (no pages)")
            else:
                pil = Image.open(str(upath)).convert("RGB")
                pil = downscale_if_huge(pil)
                results.append(run_image(pil, fname, batch, RESULTS_DIR))
        except Exception as e:
            results.append(make_error_result(fname, uploaded_web, e))

    return render_template(
        "index.html",
        results=results,
        message=f"Batch {batch}: processed {len(results)} item(s)."
    )


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
                buf = io.BytesIO(f.read())
                with fitz.open(stream=buf.getvalue(), filetype="pdf") as doc:
                    if len(doc) == 0:
                        raise RuntimeError("Empty PDF (no pages)")
                    for i in range(len(doc)):
                        pix = doc[i].get_pixmap(alpha=False)
                        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        pil = downscale_if_huge(pil)
                        out.append(run_image(pil, f"{fname} — page {i+1}", batch, RESULTS_DIR))
            else:
                pil = Image.open(f.stream).convert("RGB")
                pil = downscale_if_huge(pil)
                out.append(run_image(pil, fname, batch, RESULTS_DIR))
        except Exception as e:
            out.append(make_error_result(fname, None, e))

    return jsonify({"ok": True, "batch": batch, "results": out})


@app.get("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    # оставляю debug=False для продакшна; логи ошибок будут в stdout
    app.run(host="0.0.0.0", port=port, debug=False)

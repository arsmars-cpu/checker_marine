import os
import io
import uuid
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
import fitz  # PyMuPDF

from utils import process_pil_image, iter_pdf_pages  # наш пайплайн

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

        try:
            ext = Path(fname).suffix.lower()
            if ext == ".pdf":
                for i, pimg in iter_pdf_pages(upath):
                    pimg = downscale_if_huge(pimg)
                    res = process_pil_image(pimg, f"{fname} — page {i}", batch)
                    results.append(res)
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
                "verdict": "Analysis error",
                "regions": 0,
                "crops": [],
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
                    for i in range(len(doc)):
                        pix = doc[i].get_pixmap(alpha=False)
                        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        pil = downscale_if_huge(pil)
                        out.append(process_pil_image(pil, f"{fname} — page {i+1}", batch))
            else:
                pil = Image.open(f.stream).convert("RGB")
                pil = downscale_if_huge(pil)
                out.append(process_pil_image(pil, fname, batch))
        except Exception as e:
            app.logger.exception(f"/api analyze error: {fname} | {e}")
            out.append({
                "label": fname,
                "original": None,
                "ela": "",
                "overlay": "",
                "boxed": "",
                "verdict": "Analysis error",
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

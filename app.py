
import os
import uuid
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image
import numpy as np

from error_level_analysis import perform_ela_pil

# Optional PDF support via PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

def save_result_image(pil_img):
    uid = str(uuid.uuid4())
    out_path = os.path.join(RESULTS_DIR, f"{uid}.jpg")
    pil_img.save(out_path, format="JPEG", quality=95)
    return f"static/results/{uid}.jpg"

def analyze_pil(pil_img, label):
    vis, stats, boxes = perform_ela_pil(pil_img)
    rel_path = save_result_image(vis)
    # simple text summary
    severity = "low"
    if stats["hot_pixel_ratio"] > 0.02 or stats["boxes_count"] >= 3:
        severity = "moderate"
    if stats["hot_pixel_ratio"] > 0.05 or stats["boxes_count"] >= 6:
        severity = "high"
    summary = (
        f"File: {label} — ELA anomaly ratio: {stats['hot_pixel_ratio']} "
        f"(thr={stats['threshold']}). Regions: {stats['boxes_count']}. "
        f"Risk: {severity}."
    )
    return {
        "image_url": rel_path,
        "summary": summary,
        "boxes": boxes,
        "metrics": stats
    }

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("files")
    results = []
    for f in files:
        if not f or f.filename == "":
            continue
        fname = f.filename
        ext = fname.lower().split(".")[-1]
        if ext in ["jpg","jpeg","png"]:
            pil = Image.open(f.stream).convert("RGB")
            results.append(analyze_pil(pil, fname))
        elif ext == "pdf" and HAS_PDF:
            # Render each page and analyze
            data = f.read()
            doc = fitz.open(stream=data, filetype="pdf")
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=180, alpha=False)
                pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                results.append(analyze_pil(pil, f"{fname}#p{i+1}"))
        else:
            results.append({"error": f"Unsupported file type or PDF engine missing: {fname}"})
    return jsonify({"results": results})

# health
@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

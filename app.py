import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import fitz  # PyMuPDF
import numpy as np

from error_level_analysis import ela_image

UPLOAD_DIR = Path("static/uploads")
RESULTS_DIR = Path("static/results")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".pdf"}

app = Flask(__name__, static_folder="static", template_folder="templates")

def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXT

def save_pil(pil, path, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(str(path), quality=quality)

def pdf_to_images(pdf_path: Path):
    doc = fitz.open(str(pdf_path))
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield img, page_index

def describe_artifacts(ela_pil):
    arr = np.asarray(ela_pil.convert("L"))
    mean = float(arr.mean())
    strong = (arr > 220).sum()
    total = arr.size
    pct = 100.0 * strong / max(1,total)
    hints = []
    if pct > 0.6:
        hints.append("large bright regions")
    elif pct > 0.25:
        hints.append("moderate highlight regions")
    if mean > 25:
        hints.append("overall elevated error level")
    if not hints:
        return "No obvious manipulated zones; ELA looks consistent."
    return "ELA highlights " + ", ".join(hints) + "."

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=[])

@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("file")
    if not files:
        return render_template("index.html", results=[], message="No files uploaded.")
    batch_id = uuid.uuid4().hex[:8]
    results = []
    for f in files:
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            ext = Path(fname).suffix.lower()
            base = Path(fname).stem
            upload_path = UPLOAD_DIR / f"{batch_id}_{fname}"
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            f.save(str(upload_path))

            if ext == ".pdf":
                for pil_img, page_index in pdf_to_images(upload_path):
                    ela = ela_image(pil_img)
                    out_name = f"{batch_id}_{base}_p{page_index+1}_ela.jpg"
                    out_path = RESULTS_DIR / out_name
                    save_pil(ela, out_path, quality=95)
                    page_name = f"{batch_id}_{base}_p{page_index+1}.jpg"
                    page_path = RESULTS_DIR / page_name
                    save_pil(pil_img, page_path, quality=90)
                    results.append({
                        "original": str(page_path).replace("\\","/"),
                        "ela": str(out_path).replace("\\","/"),
                        "summary": describe_artifacts(ela),
                        "label": f"{fname} — page {page_index+1}"
                    })
            else:
                pil = Image.open(str(upload_path))
                ela = ela_image(pil)
                out_name = f"{batch_id}_{base}_ela.jpg"
                out_path = RESULTS_DIR / out_name
                save_pil(ela, out_path, quality=95)
                results.append({
                    "original": str(upload_path).replace("\\","/"),
                    "ela": str(out_path).replace("\\","/"),
                    "summary": describe_artifacts(ela),
                    "label": fname
                })
    return render_template("index.html", results=results, message=f"Processed {len(results)} page(s)/file(s). Batch {batch_id}.")

@app.route('/health')
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)

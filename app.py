
import os
import uuid
import io
from typing import List, Tuple

from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import cv2
import fitz  # PyMuPDF

# --- Config ---
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB per request
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(img: Image.Image, prefix: str = "img") -> str:
    name = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(RESULT_FOLDER, name)
    img.convert("RGB").save(path, "JPEG", quality=95)
    return path


def perform_ela(pil_image: Image.Image, resave_quality: int = 90) -> Image.Image:
    # ELA core: resave to JPEG, take difference, scale brightness to 0..255
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, "JPEG", quality=resave_quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(pil_image.convert("RGB"), resaved)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(diff).enhance(scale)
    return ela


def detect_artifact_boxes(ela_img: Image.Image, min_area_ratio: float = 0.002) -> List[Tuple[int,int,int,int]]:
    # Convert ELA image to grayscale numpy array
    ela_np = np.array(ela_img.convert("L"))
    # Normalize & threshold highlights
    ela_blur = cv2.GaussianBlur(ela_np, (5, 5), 0)
    # Adaptive/dual threshold
    _, th = cv2.threshold(ela_blur, 200, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = ela_np.shape[:2]
    min_area = int(min_area_ratio * w * h)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            boxes.append((x, y, x + bw, y + bh))
    # Sort by area desc
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes


def region_label(box, w, h):
    # Map bbox center to coarse grid label
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hor = "left" if cx < w/3 else ("center" if cx < 2*w/3 else "right")
    ver = "top" if cy < h/3 else ("middle" if cy < 2*h/3 else "bottom")
    return f"{ver}-{hor}"


def annotate_image(base_img: Image.Image, boxes: List[Tuple[int,int,int,int]]) -> Image.Image:
    out = base_img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0+4, y0+4), f"#{idx}", fill=(255,0,0))
    return out


def analyze_pil(pil_img: Image.Image):
    # Run ELA
    ela = perform_ela(pil_img, resave_quality=90)
    # Detect boxes
    boxes = detect_artifact_boxes(ela)
    # Build descriptions
    w, h = pil_img.size
    regions = [region_label(b, w, h) for b in boxes]
    if boxes:
        desc = f"Detected {len(boxes)} potential manipulated region(s): " + \               ", ".join(sorted(set(regions))) + "."
    else:
        desc = "No significant ELA anomalies detected."
    # Annotate
    overlay = annotate_image(pil_img, boxes)
    return ela, overlay, boxes, desc


def images_from_pdf_bytes(pdf_bytes: bytes, dpi: int = 180) -> List[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi)
        pages.append(Image.open(io.BytesIO(pix.tobytes("jpg"))).convert("RGB"))
    return pages


@app.route("/", methods=["GET", "POST"])
def index():
    results = []  # list of dicts
    error = None
    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or all(f.filename == "" for f in files):
                abort(400, "No files uploaded")

            for f in files:
                if f.filename == "":
                    continue
                filename = secure_filename(f.filename)
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext not in ALLOWED_EXTENSIONS:
                    results.append({
                        "name": filename,
                        "status": "error",
                        "message": f"Unsupported file type: {ext}"
                    })
                    continue

                if ext == "pdf":
                    pdf_bytes = f.read()
                    pages = images_from_pdf_bytes(pdf_bytes, dpi=180)
                    for idx, page_img in enumerate(pages, start=1):
                        ela, overlay, boxes, desc = analyze_pil(page_img)
                        ela_path = save_image(ela, prefix="ela")
                        overlay_path = save_image(overlay, prefix="overlay")
                        results.append({
                            "name": f"{filename} (page {idx})",
                            "status": "ok",
                            "ela_path": ela_path.replace("static/", ""),
                            "overlay_path": overlay_path.replace("static/", ""),
                            "boxes": boxes,
                            "description": desc
                        })
                else:
                    # Image path
                    img = Image.open(f.stream).convert("RGB")
                    ela, overlay, boxes, desc = analyze_pil(img)
                    ela_path = save_image(ela, prefix="ela")
                    overlay_path = save_image(overlay, prefix="overlay")
                    results.append({
                        "name": filename,
                        "status": "ok",
                        "ela_path": ela_path.replace("static/", ""),
                        "overlay_path": overlay_path.replace("static/", ""),
                        "boxes": boxes,
                        "description": desc
                    })

        except Exception as e:
            error = str(e)

    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

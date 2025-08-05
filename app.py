
import os
import uuid
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def perform_ela(image_path):
    original = Image.open(image_path).convert("RGB")
    temp_path = image_path + ".resaved.jpg"
    original.save(temp_path, "JPEG", quality=90)
    resaved = Image.open(temp_path)

    ela_image = Image.new("RGB", original.size)
    for x in range(original.width):
        for y in range(original.height):
            orig_pixel = original.getpixel((x, y))
            resaved_pixel = resaved.getpixel((x, y))
            diff = tuple([min(255, abs(o - r) * 10) for o, r in zip(orig_pixel, resaved_pixel)])
            ela_image.putpixel((x, y), diff)

    ela_output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"ela_{uuid.uuid4().hex}.jpg")
    ela_image.save(ela_output_path)
    return ela_output_path

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        files = request.files.getlist("file")
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                ext = filename.rsplit(".", 1)[-1].lower()
                uid = uuid.uuid4().hex
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uid}_{filename}")

                if ext == "pdf":
                    doc = fitz.open(stream=file.read(), filetype="pdf")
                    for i, page in enumerate(doc):
                        pix = page.get_pixmap(dpi=150)
                        img_path = save_path.replace(".pdf", f"_page{i}.jpg")
                        pix.save(img_path)
                        ela_path = perform_ela(img_path)
                        results.append((img_path, ela_path))
                elif ext in ["jpg", "jpeg", "png"]:
                    file.save(save_path)
                    ela_path = perform_ela(save_path)
                    results.append((save_path, ela_path))
    return render_template("index.html", results=results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

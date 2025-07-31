from flask import Flask, request, render_template, send_file, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import fitz  # PyMuPDF
from error_level_analysis import analyze_image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        uploaded_files = request.files.getlist("file")
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            ext = os.path.splitext(filename)[1].lower()
            images = []
            if ext == ".pdf":
                doc = fitz.open(filepath)
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=200)
                    image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}_page_{i}.jpg")
                    pix.save(image_path)
                    images.append(image_path)
            else:
                images.append(filepath)

            for image_path in images:
                ela_image_path, description = analyze_image(image_path)
                results.append((os.path.basename(image_path), ela_image_path, description))

        return render_template("index.html", results=results)
    return render_template("index.html", results=[])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)

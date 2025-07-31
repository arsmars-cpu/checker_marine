from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from error_level_analysis import analyze_image
from pdf2image import convert_from_path

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        uploaded_files = request.files.getlist("file")
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)

                if filename.lower().endswith(".pdf"):
                    images = convert_from_path(file_path)
                    for i, image in enumerate(images):
                        image_path = file_path + f"_page_{i}.jpg"
                        image.save(image_path, "JPEG")
                        result_path, description = analyze_image(image_path)
                        results.append((result_path, description))
                else:
                    result_path, description = analyze_image(file_path)
                    results.append((result_path, description))
    return render_template("index.html", results=results)
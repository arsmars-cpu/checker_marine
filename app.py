from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import fitz
import cv2
import numpy as np
from error_level_analysis import perform_ela

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        files = request.files.getlist('files')
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Convert PDF to image if needed
            if filename.lower().endswith('.pdf'):
                doc = fitz.open(filepath)
                for page_num in range(len(doc)):
                    pix = doc[page_num].get_pixmap(dpi=300)
                    image_path = os.path.join(UPLOAD_FOLDER, f"{filename}_page{page_num}.jpg")
                    pix.save(image_path)
                    output_path, description = perform_ela(image_path, RESULT_FOLDER)
                    results[f"{filename}_page{page_num}"] = {
                        "output_image": os.path.relpath(output_path, "static"),
                        "description": description
                    }
                os.remove(filepath)
            else:
                output_path, description = perform_ela(filepath, RESULT_FOLDER)
                results[filename] = {
                    "output_image": os.path.relpath(output_path, "static"),
                    "description": description
                }
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

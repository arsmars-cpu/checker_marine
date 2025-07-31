from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            filepath = os.path.join("static", image.filename)
            os.makedirs("static", exist_ok=True)
            image.save(filepath)
            return render_template("index.html", result="Uploaded", filename=image.filename)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from ela import perform_ela
from analysis import analyze_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist("files[]")
        for file in uploaded_files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                ela_path = perform_ela(filepath)
                result_text = analyze_image(ela_path)
                results.append((filename, ela_path, result_text))
        return render_template('index.html', results=results)
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

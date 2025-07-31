
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from error_level_analysis import perform_ela_analysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                result_path, summary = perform_ela_analysis(filepath, app.config['RESULT_FOLDER'])
                results.append({'image': result_path, 'summary': summary})
        return render_template('index.html', results=results)
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


import os
import uuid
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from pdf2image import convert_from_path
from ela import perform_ela_analysis, describe_artifacts

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.' + ext)
                file.save(filepath)

                images = []
                if ext == 'pdf':
                    images = convert_from_path(filepath)
                else:
                    images = [Image.open(filepath)]

                for img in images:
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + '.jpg')
                    img.save(img_path)
                    ela_path, artifact_desc = perform_ela_analysis(img_path)
                    results.append({
                        'filename': filename,
                        'image_path': img_path,
                        'ela_path': ela_path,
                        'description': artifact_desc
                    })
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run()

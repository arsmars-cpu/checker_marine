from flask import Flask, request, render_template
from PIL import Image, ImageChops, ImageEnhance
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def perform_ela(image_path):
    ela_path = os.path.join(UPLOAD_FOLDER, f"ela_{os.path.basename(image_path)}")
    image = Image.open(image_path).convert('RGB')
    temp_path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
    image.save(temp_path, 'JPEG', quality=90)

    ela_image = ImageChops.difference(image, Image.open(temp_path))
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(ela_path)
    return ela_path, "Артефакты обнаружены" if max_diff > 15 else "Артефактов не найдено"

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('documents')
        for file in files:
            if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(path)
                ela_path, message = perform_ela(path)
                results.append({
                    'original': path,
                    'ela': ela_path,
                    'message': message
                })
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)


from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela_analysis(image_path, result_folder):
    ela_path = os.path.join(result_folder, 'ela_' + os.path.basename(image_path))
    original = Image.open(image_path).convert('RGB')
    compressed_path = image_path + '.resaved.jpg'
    original.save(compressed_path, 'JPEG', quality=95)
    compressed = Image.open(compressed_path)
    diff = ImageChops.difference(original, compressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    diff.save(ela_path)
    summary = f"ELA Analysis completed. Max pixel difference: {max_diff}."
    return ela_path, summary

from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela(image_path, quality=90):
    basename = os.path.basename(image_path)
    resaved_path = f"uploads/resaved_{basename}"
    ela_path = f"uploads/ela_{basename}"
    image = Image.open(image_path).convert('RGB')
    image.save(resaved_path, 'JPEG', quality=quality)
    resaved_image = Image.open(resaved_path)
    ela_image = ImageChops.difference(image, resaved_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(ela_path)
    return ela_path

from PIL import Image, ImageChops, ImageEnhance
import os

def analyze_image(image_path):
    original = Image.open(image_path).convert("RGB")
    temp_path = image_path + "_ela.jpg"
    original.save("temp.jpg", "JPEG", quality=90)

    compressed = Image.open("temp.jpg")
    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    ela_image.save(temp_path)

    # Simple analysis
    description = "Signs of digital modification detected." if max_diff > 20 else "No significant artifacts found."
    return temp_path, description
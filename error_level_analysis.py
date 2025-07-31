from PIL import Image, ImageChops, ImageEnhance
import os

def analyze_image(image_path):
    im = Image.open(image_path).convert("RGB")
    ela_path = os.path.join("static", os.path.basename(image_path).replace(".", "_ela."))
    im.save("temp.jpg", "JPEG", quality=90)
    temp = Image.open("temp.jpg")
    ela_image = ImageChops.difference(im, temp)
    enhancer = ImageEnhance.Brightness(ela_image)
    ela_image = enhancer.enhance(20)
    ela_image.save(ela_path)

    description = "Brighter areas in ELA image may indicate manipulated regions (compression differences)."
    return ela_path, description

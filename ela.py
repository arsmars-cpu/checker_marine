
from PIL import Image, ImageChops, ImageEnhance
import os
import uuid

def perform_ela_analysis(image_path, quality=90):
    temp_filename = image_path.replace(".jpg", "_resaved.jpg")
    ela_filename = image_path.replace(".jpg", "_ela.jpg")

    image = Image.open(image_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    resaved_image = Image.open(temp_filename)

    diff = ImageChops.difference(image, resaved_image)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    diff = ImageEnhance.Brightness(diff).enhance(scale)
    diff.save(ela_filename)

    description = describe_artifacts(diff)
    return ela_filename, description

def describe_artifacts(diff_image):
    histogram = diff_image.histogram()
    high_diff_pixels = sum(histogram[200:256])
    total_pixels = sum(histogram)

    if total_pixels == 0:
        return "Ошибка анализа изображения."

    ratio = high_diff_pixels / total_pixels
    if ratio > 0.05:
        return "Обнаружены признаки редактирования (высокая степень артефактов)."
    elif ratio > 0.01:
        return "Умеренные артефакты, возможное редактирование."
    else:
        return "Редактирование не обнаружено, изображение выглядит аутентичным."

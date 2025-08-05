from PIL import Image, ImageChops, ImageEnhance
import os
import cv2
import numpy as np

def perform_ela(image_path, result_folder, quality=90):
    filename = os.path.basename(image_path)
    temp_path = os.path.join(result_folder, "temp.jpg")

    image = Image.open(image_path).convert("RGB")
    image.save(temp_path, 'JPEG', quality=quality)

    ela_image = ImageChops.difference(image, Image.open(temp_path))
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_result_path = os.path.join(result_folder, f"ela_{filename}")
    ela_image.save(ela_result_path)

    img = cv2.imread(ela_result_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(thresh)

    artifact_zones = []
    for label in range(1, num_labels):
        mask = labels_im == label
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        artifact_zones.append((x0, y0, x1, y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv2.imwrite(ela_result_path, img)
    description = f"Detected {len(artifact_zones)} possible manipulated regions."

    return ela_result_path, description

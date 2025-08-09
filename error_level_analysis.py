from PIL import Image, ImageChops, ImageEnhance
import io

def ela_image(pil_img, quality=90, scale=12.0):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    comp = Image.open(buf)
    diff = ImageChops.difference(pil_img, comp)
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema])
    factor = scale if max_diff == 0 else max(1.0, scale * 255.0 / max_diff)
    diff = ImageEnhance.Brightness(diff).enhance(factor)
    return diff

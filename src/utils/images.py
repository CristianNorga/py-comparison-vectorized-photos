import base64
import io

from PIL import Image


def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def pil_to_png_base64(img: Image.Image) -> str:
    with io.BytesIO() as buffer:
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

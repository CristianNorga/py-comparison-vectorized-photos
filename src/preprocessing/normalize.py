from PIL import Image, ImageOps


def normalize(img: Image.Image, size: int = 320) -> Image.Image:
    image = ImageOps.exif_transpose(img)
    return image.resize((size, size))

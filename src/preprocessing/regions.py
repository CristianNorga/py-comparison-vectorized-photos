from collections.abc import Iterable

from PIL import Image

from models.dto import LandmarkDTO


_REGION_KEYS = (
    "eyeLeft",
    "eyeRight",
    "nose",
    "mouthLeft",
    "mouthRight",
    "leftEyeBrowLeft",
    "leftEyeBrowRight",
    "rightEyeBrowLeft",
    "rightEyeBrowRight",
    "leftEarTragus",
    "rightEarTragus",
)


def crop_central_region(full_img: Image.Image, landmarks: Iterable[LandmarkDTO]) -> Image.Image:
    landmark_map = {landmark.type: (landmark.x, landmark.y) for landmark in landmarks}
    width, height = full_img.size

    xs: list[float] = []
    ys: list[float] = []

    for key in _REGION_KEYS:
        if key in landmark_map:
            x, y = landmark_map[key]
            xs.append(x * width)
            ys.append(y * height)

    if not xs:
        x0, y0, x1, y1 = int(0.2 * width), int(0.2 * height), int(0.8 * width), int(0.7 * height)
    else:
        margin = 0.1
        x0 = max(int(min(xs) - margin * width), 0)
        y0 = max(int(min(ys) - margin * height), 0)
        x1 = min(int(max(xs) + margin * width), width)
        y1 = min(int(max(ys) + margin * height), height)

    return full_img.crop((x0, y0, x1, y1))

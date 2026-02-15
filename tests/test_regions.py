from PIL import Image

from models.dto import LandmarkDTO
from preprocessing.regions import crop_central_region


def test_crop_central_region_fallback() -> None:
    img = Image.new("RGB", (100, 100), color="white")
    region = crop_central_region(img, landmarks=[])
    assert region.size[0] > 0
    assert region.size[1] > 0


def test_crop_central_region_with_landmarks() -> None:
    img = Image.new("RGB", (200, 200), color="white")
    landmarks = [
        LandmarkDTO(Type="eyeLeft", X=0.35, Y=0.35),
        LandmarkDTO(Type="eyeRight", X=0.65, Y=0.35),
        LandmarkDTO(Type="nose", X=0.5, Y=0.5),
    ]
    region = crop_central_region(img, landmarks=landmarks)
    assert region.size[0] > 0
    assert region.size[1] > 0

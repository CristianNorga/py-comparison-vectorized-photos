from typing import Any

from botocore.client import BaseClient
from PIL import Image

from models.dto import BoundingBoxDTO, LandmarkDTO
from utils.images import load_image


def detect_face_and_bbox(
    image_bytes: bytes, rekognition_client: BaseClient
) -> tuple[BoundingBoxDTO, list[LandmarkDTO]]:
    response: dict[str, Any] = rekognition_client.detect_faces(
        Image={"Bytes": image_bytes}, Attributes=["ALL"]
    )
    faces: list[dict[str, Any]] = response.get("FaceDetails", [])
    if not faces:
        raise ValueError("No face detected")

    face = max(faces, key=lambda item: float(item.get("Confidence", 0.0)))
    bbox = BoundingBoxDTO.model_validate(face["BoundingBox"])
    landmarks = [LandmarkDTO.model_validate(landmark) for landmark in face.get("Landmarks", [])]
    return bbox, landmarks


def crop_bbox(image_bytes: bytes, bbox: BoundingBoxDTO) -> Image.Image:
    img = load_image(image_bytes)
    width, height = img.size

    left = max(int(bbox.left * width), 0)
    top = max(int(bbox.top * height), 0)
    right = min(int((bbox.left + bbox.width) * width), width)
    bottom = min(int((bbox.top + bbox.height) * height), height)

    if left >= right or top >= bottom:
        raise ValueError("Invalid bbox after projection to pixel coordinates")

    return img.crop((left, top, right, bottom))

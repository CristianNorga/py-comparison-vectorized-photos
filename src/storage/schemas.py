from datetime import UTC, datetime
from typing import Any

from models.dto import BoundingBoxDTO, FaceDocumentDTO, LandmarkDTO


def build_face_doc(
    image_id: str,
    original_filename: str,
    bbox: BoundingBoxDTO,
    landmarks: list[LandmarkDTO],
    embedding: list[float],
    level: str = "daily",
    user_ref: str | None = None,
    face_id: str | None = None, # Deprecated or used for legacy rekognition id? Keeping optional.
    face_thumbnail_b64: str | None = None,
) -> dict[str, Any]:
    # Retornamos dict directamente para insertar en Mongo, o DTO si se prefiere.
    doc = {
        "image_id": image_id,
        "original_filename": original_filename,
        "face_id": face_id,
        "bbox": [bbox.left, bbox.top, bbox.width, bbox.height],
        "landmarks": [landmark.model_dump(by_alias=True) for landmark in landmarks],
        "embedding": embedding,
        "level": level,
        "created_at": datetime.now(UTC),
        "user_ref": user_ref,
        "face_thumbnail_b64": face_thumbnail_b64,
    }
    return doc


def build_cluster_doc(
    level: str,
    cluster_id: str,
    centroid: list[float],
    member_face_ids: list[str],
    period_key: str,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "doc_type": "cluster",
        "level": level,
        "cluster_id": cluster_id,
        "centroid": centroid,
        "member_face_ids": member_face_ids,
        "member_count": len(member_face_ids),
        "period_key": period_key,
        "created_at": now,
        "updated_at": now,
    }

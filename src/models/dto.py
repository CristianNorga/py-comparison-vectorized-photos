from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BoundingBoxDTO(BaseModel):
    left: float = Field(alias="Left")
    top: float = Field(alias="Top")
    width: float = Field(alias="Width")
    height: float = Field(alias="Height")


class LandmarkDTO(BaseModel):
    type: str = Field(alias="Type")
    x: float = Field(alias="X")
    y: float = Field(alias="Y")


class MatchDTO(BaseModel):
    face_id: str
    similarity: float
    original_filename: str | None = None


class FaceDocumentDTO(BaseModel):
    image_id: str
    original_filename: str
    face_id: str | None
    bbox: list[float]
    landmarks: list[dict[str, Any]]
    level: str
    created_at: datetime
    user_ref: str | None = None
    cluster_id: str | None = None
    embedding: list[float] | None = None
    face_thumbnail_b64: str | None = None


class IngestResultDTO(BaseModel):
    suspicious: bool
    matches: list[MatchDTO]
    doc_id: str


class ClusterEventDTO(BaseModel):
    event_type: str
    level: str
    cluster_id: str
    member_count: int
    centroid: list[float]
    period_key: str
    created_at: datetime


class PromotionReportDTO(BaseModel):
    source_level: str
    target_level: str
    merged_count: int
    created_count: int

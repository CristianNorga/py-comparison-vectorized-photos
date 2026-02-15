from typing import Any

from botocore.client import BaseClient
from PIL import Image

from models.dto import MatchDTO
from vectorization.backend import VectorBackend


class AwsRekognitionBackend(VectorBackend):
    def __init__(self, client: BaseClient) -> None:
        self.client = client

    def index_and_search(
        self, image_bytes: bytes, collection_name: str, top_k: int = 5
    ) -> tuple[str | None, list[MatchDTO]]:
        search_response: dict[str, Any] = self.client.search_faces_by_image(
            CollectionId=collection_name,
            Image={"Bytes": image_bytes},
            MaxFaces=top_k,
        )

        matches = [
            MatchDTO(
                face_id=match["Face"]["FaceId"],
                similarity=float(match.get("Similarity", 0.0)),
            )
            for match in search_response.get("FaceMatches", [])
        ]

        index_response: dict[str, Any] = self.client.index_faces(
            CollectionId=collection_name,
            Image={"Bytes": image_bytes},
            DetectionAttributes=["DEFAULT"],
        )

        face_records: list[dict[str, Any]] = index_response.get("FaceRecords", [])
        face_id: str | None = (
            str(face_records[0]["Face"]["FaceId"]) if face_records and face_records[0].get("Face") else None
        )

        return face_id, matches

    def embed(self, pil_img: Image.Image) -> list[float] | None:
        return None

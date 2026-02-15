from datetime import UTC, datetime

from PIL import Image

import pipeline.ingest_sync as ingest_module
from models.dto import BoundingBoxDTO, LandmarkDTO, MatchDTO
from utils.images import pil_to_jpeg_bytes


class FakeStorage:
    def __init__(self) -> None:
        self.inserted: list[tuple[str, dict]] = []

    def find_original_filenames_by_face_ids(self, level: str, face_ids: list[str]) -> dict[str, str]:
        assert level == "daily"
        return {"face-1": "other_image.jpg"} if "face-1" in face_ids else {}

    def insert_face(self, level: str, document: dict) -> None:
        self.inserted.append((level, document))


class FakeBackend:
    def index_and_search(self, image_bytes: bytes, collection_name: str, top_k: int = 5):
        assert collection_name
        return "face-new", [MatchDTO(face_id="face-1", similarity=0.95)]


def test_ingest_sync_marks_suspicious_and_persists(monkeypatch) -> None:
    image = Image.new("RGB", (100, 100), color="white")
    image_bytes = pil_to_jpeg_bytes(image)

    settings = ingest_module.get_settings()
    fake_storage = FakeStorage()

    monkeypatch.setattr(ingest_module, "get_rekognition_client", lambda _settings: object())
    monkeypatch.setattr(ingest_module, "AwsRekognitionBackend", lambda _client: FakeBackend())
    monkeypatch.setattr(
        ingest_module,
        "detect_face_and_bbox",
        lambda _bytes, _client: (
            BoundingBoxDTO(Left=0.0, Top=0.0, Width=1.0, Height=1.0),
            [LandmarkDTO(Type="nose", X=0.5, Y=0.5)],
        ),
    )

    result = ingest_module.ingest_sync(
        image_bytes=image_bytes,
        original_filename="input.jpg",
        user_ref="user-01",
        settings=settings,
        storage=fake_storage,
    )

    assert result.suspicious is True
    assert result.matches[0].original_filename == "other_image.jpg"
    assert len(fake_storage.inserted) == 1
    assert fake_storage.inserted[0][0] == "daily"
    assert isinstance(fake_storage.inserted[0][1]["created_at"], datetime)
    assert fake_storage.inserted[0][1]["created_at"].tzinfo == UTC

from __future__ import annotations

from datetime import UTC, datetime
import io
import uuid

from aws.rekognition_client import get_rekognition_client
from config import Settings, get_settings
from models.dto import IngestResultDTO, MatchDTO
from preprocessing.crop import crop_bbox, detect_face_and_bbox
from preprocessing.normalize import normalize
from preprocessing.regions import crop_central_region
from storage.mongo import MongoStorage, get_db
from storage.schemas import build_face_doc
from utils.images import pil_to_jpeg_bytes, pil_to_png_base64
from vectorization.local_embeddings import LocalEmbeddingBackend


async def ingest_sync(
    image_bytes: bytes,
    original_filename: str,
    user_ref: str | None = None,
    *,
    settings: Settings | None = None,
    storage: MongoStorage | None = None,
) -> IngestResultDTO:
    app_settings = settings or get_settings()
    active_storage = storage or MongoStorage(get_db(app_settings), app_settings)

    rekognition_client = get_rekognition_client(app_settings)
    # backend = AwsRekognitionBackend(rekognition_client) OLD

    bbox, landmarks = detect_face_and_bbox(image_bytes, rekognition_client)
    face_img = normalize(crop_bbox(image_bytes, bbox), size=app_settings.img_size)
    
    # NUEVO: Recorte de región central (ojos/nariz) para vectorización local
    central_face = crop_central_region(face_img, landmarks)
    
    # Convertir a bytes para el backend local
    with io.BytesIO() as buffer:
        central_face.save(buffer, format="JPEG")
        region_bytes = buffer.getvalue()

    # NUEVO: Vectorización Local
    embedding = LocalEmbeddingBackend().embed_region(region_bytes)

    # NUEVO: Búsqueda Vectorial en Mongo
    # Nota: vector_search retorna documentos con score, ya no FaceId de rekognition
    matches_data = active_storage.vector_search(embedding, top_k=app_settings.top_k_matches)
    
    matches: list[MatchDTO] = []
    suspicious = False
    
    for m in matches_data:
        # Calcular similitud o usar score directo (depende de métrica Atlas Search)
        # Asumimos score retornado es similitud normalizada [0,1] o similar
        sim = m.get('score', 0.0)
        matches.append(MatchDTO(face_id="local_vector_match", similarity=sim, original_filename=m.get('original_filename')))
        if sim >= app_settings.thresh_daily:
            suspicious = True

    # Persistencia
    image_id = str(uuid.uuid4())
    doc = build_face_doc(
        image_id=image_id,
        original_filename=original_filename,
        user_ref=user_ref,
        bbox=bbox,
        landmarks=landmarks,
        embedding=embedding,
    )
    # Usar el método general o específico de daily
    active_storage.index_face(doc)

    return IngestResultDTO(
        suspicious=suspicious,
        doc_id=image_id,
        matches=matches,
    )
    enriched_matches.append(
        MatchDTO(
            face_id=match.face_id,
            similarity=match.similarity,
            original_filename=original_lookup.get(match.face_id),
        )
    )

    image_id = str(uuid.uuid4())
    document = build_face_doc(
        image_id=image_id,
        original_filename=original_filename,
        face_id=face_id,
        bbox=bbox,
        landmarks=landmarks,
        level="daily",
        user_ref=user_ref,
        face_thumbnail_b64=pil_to_png_base64(central_face),
    ).model_dump(mode="python")

    document["ingested_at"] = datetime.now(UTC)
    active_storage.insert_face("daily", document)

    return IngestResultDTO(suspicious=suspicious, matches=enriched_matches[:5], doc_id=image_id)

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from clustering.hdbscan_cluster import cluster_embeddings, compute_cluster_centroids
from config import Settings, get_settings
from storage.mongo import MongoStorage, get_db
from utils.images import base64_to_pil
from vectorization.local_embeddings_facenet import LocalEmbeddingBackend


def enrich_daily_faces(
    *,
    settings: Settings | None = None,
    storage: MongoStorage | None = None,
) -> dict[str, int]:
    app_settings = settings or get_settings()
    active_storage = storage or MongoStorage(get_db(app_settings), app_settings)

    if not app_settings.local_embedding_enabled:
        return {"embedded": 0, "clustered": 0}

    now = datetime.now(UTC)
    start_of_day = datetime(now.year, now.month, now.day, tzinfo=UTC)
    end_of_day = start_of_day + timedelta(days=1)

    embedder = LocalEmbeddingBackend()
    pending_docs = active_storage.list_faces_without_embeddings("daily", start_of_day, end_of_day)

    embedded_count = 0
    for doc in pending_docs:
        face_thumbnail_b64 = doc.get("face_thumbnail_b64")
        image_id = doc.get("image_id")
        if not isinstance(face_thumbnail_b64, str) or not isinstance(image_id, str):
            continue

        embedding = embedder.embed(base64_to_pil(face_thumbnail_b64))
        active_storage.update_face_embedding("daily", image_id, embedding)
        embedded_count += 1

    docs_with_embeddings = active_storage.list_faces_with_embeddings("daily", start_of_day, end_of_day)
    assignments = cluster_embeddings(
        docs_with_embeddings,
        min_cluster_size=app_settings.hdbscan_min_cluster_size,
        min_samples=app_settings.hdbscan_min_samples,
    )

    for image_id, cluster_id in assignments.items():
        active_storage.update_face_cluster("daily", image_id, cluster_id)

    centroids = compute_cluster_centroids(docs_with_embeddings, assignments)
    for cluster_id, centroid in centroids.items():
        member_ids = [
            str(doc["face_id"])
            for doc in docs_with_embeddings
            if assignments.get(str(doc.get("image_id"))) == cluster_id and doc.get("face_id")
        ]
        active_storage.insert_event(
            {
                "event_type": "daily_cluster",
                "level": "daily",
                "cluster_id": cluster_id,
                "centroid": centroid,
                "member_face_ids": member_ids,
                "member_count": len(member_ids),
                "period_key": start_of_day.date().isoformat(),
                "created_at": datetime.now(UTC),
            }
        )

    return {"embedded": embedded_count, "clustered": len(assignments)}

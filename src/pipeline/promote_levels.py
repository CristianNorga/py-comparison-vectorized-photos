from __future__ import annotations

from datetime import UTC, datetime, timedelta
import uuid

from clustering.merge_agglomerative import find_best_cluster_match
from config import Settings, get_settings
from models.dto import PromotionReportDTO
from storage.mongo import MongoStorage, get_db
from storage.schemas import build_cluster_doc


def _group_daily_clusters(docs: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for doc in docs:
        cluster_id = doc.get("cluster_id")
        embedding = doc.get("embedding")
        face_id = doc.get("face_id")
        if not isinstance(cluster_id, str) or not isinstance(embedding, list):
            continue

        entry = grouped.setdefault(cluster_id, {"cluster_id": cluster_id, "vectors": [], "members": []})
        entry["vectors"].append(embedding)
        if isinstance(face_id, str):
            entry["members"].append(face_id)

    clusters: list[dict] = []
    for cluster_id, payload in grouped.items():
        vectors = payload["vectors"]
        if not vectors:
            continue
        dimensions = len(vectors[0])
        centroid = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimensions)]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "centroid": centroid,
                "member_face_ids": payload["members"],
                "member_count": len(payload["members"]),
            }
        )
    return clusters


def _promote(
    source_clusters: list[dict],
    target_level: str,
    threshold: float,
    period_key: str,
    storage: MongoStorage,
) -> PromotionReportDTO:
    existing_target_clusters = storage.get_cluster_docs(target_level)

    merged_count = 0
    created_count = 0

    for source in source_clusters:
        centroid = source.get("centroid")
        member_face_ids = source.get("member_face_ids", [])
        if not isinstance(centroid, list):
            continue

        match = find_best_cluster_match(centroid, existing_target_clusters, threshold)

        if match is not None:
            existing_members = set(match.get("member_face_ids", []))
            incoming_members = set(member_face_ids) if isinstance(member_face_ids, list) else set()
            all_members = sorted(existing_members | incoming_members)

            current_count = int(match.get("member_count", 0))
            incoming_count = len(incoming_members)
            total = max(current_count + incoming_count, 1)

            current_centroid = match.get("centroid", centroid)
            if isinstance(current_centroid, list):
                blended = [
                    ((current_centroid[i] * current_count) + (centroid[i] * incoming_count)) / total
                    for i in range(len(centroid))
                ]
            else:
                blended = centroid

            storage.upsert_cluster_doc(
                target_level,
                str(match["cluster_id"]),
                {
                    "period_key": period_key,
                    "centroid": blended,
                    "member_face_ids": all_members,
                    "member_count": len(all_members),
                },
            )
            merged_count += 1
        else:
            new_cluster_id = f"{target_level}-{uuid.uuid4()}"
            cluster_doc = build_cluster_doc(
                level=target_level,
                cluster_id=new_cluster_id,
                centroid=centroid,
                member_face_ids=member_face_ids if isinstance(member_face_ids, list) else [],
                period_key=period_key,
            )
            storage.create_cluster_doc(target_level, cluster_doc)
            existing_target_clusters.append(cluster_doc)
            created_count += 1

    return PromotionReportDTO(
        source_level="unknown", target_level=target_level, merged_count=merged_count, created_count=created_count
    )


def promote_daily_to_weekly(
    *,
    settings: Settings | None = None,
    storage: MongoStorage | None = None,
) -> PromotionReportDTO:
    app_settings = settings or get_settings()
    active_storage = storage or MongoStorage(get_db(app_settings), app_settings)

    now = datetime.now(UTC)
    start_of_day = datetime(now.year, now.month, now.day, tzinfo=UTC)
    end_of_day = start_of_day + timedelta(days=1)

    daily_docs = active_storage.list_faces_with_embeddings("daily", start_of_day, end_of_day)
    daily_clusters = _group_daily_clusters(daily_docs)
    report = _promote(
        source_clusters=daily_clusters,
        target_level="weekly",
        threshold=app_settings.thresh_weekly,
        period_key=start_of_day.date().isoformat(),
        storage=active_storage,
    )
    return report.model_copy(update={"source_level": "daily", "target_level": "weekly"})


def promote_weekly_to_monthly(
    *,
    settings: Settings | None = None,
    storage: MongoStorage | None = None,
) -> PromotionReportDTO:
    app_settings = settings or get_settings()
    active_storage = storage or MongoStorage(get_db(app_settings), app_settings)

    source_clusters = active_storage.get_cluster_docs("weekly")
    now = datetime.now(UTC)
    period_key = f"{now.year}-{now.month:02d}"
    report = _promote(
        source_clusters=source_clusters,
        target_level="monthly",
        threshold=app_settings.thresh_monthly,
        period_key=period_key,
        storage=active_storage,
    )
    return report.model_copy(update={"source_level": "weekly", "target_level": "monthly"})


def promote_monthly_to_yearly(
    *,
    settings: Settings | None = None,
    storage: MongoStorage | None = None,
) -> PromotionReportDTO:
    app_settings = settings or get_settings()
    active_storage = storage or MongoStorage(get_db(app_settings), app_settings)

    source_clusters = active_storage.get_cluster_docs("monthly")
    now = datetime.now(UTC)
    period_key = f"{now.year}"

    report = _promote(
        source_clusters=source_clusters,
        target_level="yearly",
        threshold=app_settings.thresh_yearly,
        period_key=period_key,
        storage=active_storage,
    )
    return report.model_copy(update={"source_level": "monthly", "target_level": "yearly"})

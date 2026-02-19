from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config import Settings


_COLLECTIONS = ("faces_daily", "faces_weekly", "faces_monthly", "faces_yearly", "events")


def get_client(settings: Settings) -> MongoClient[Any]:
    return MongoClient(
        settings.mongo_uri,
        connectTimeoutMS=settings.mongo_connect_timeout_ms,
        serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
    )


def get_db(settings: Settings, client: MongoClient[Any] | None = None) -> Database[Any]:
    local_client = client or get_client(settings)
    return local_client[settings.mongo_db]


class MongoStorage:
    def __init__(self, db: Database[Any], settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def ensure_indexes(self) -> None:
        for collection_name in _COLLECTIONS:
            collection = self.db[collection_name]
            collection.create_index([("created_at", ASCENDING)])
            collection.create_index([("face_id", ASCENDING)])
            collection.create_index([("cluster_id", ASCENDING)])
            collection.create_index([("user_ref", ASCENDING)])

        self.db["faces_daily"].create_index(
            "created_at", expireAfterSeconds=self.settings.faces_daily_ttl_seconds
        )

    def index_face(self, doc: dict[str, Any]) -> None:
        self.db["faces_daily"].insert_one(doc)

    def vector_search(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """
        Realiza una búsqueda vectorial aproximada (ANN) en Mongo.
        Requiere un índice de búsqueda vectorial configurado en Atlas.
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "path": "embedding", 
                    "queryVector": embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k
            }
            },
            {
                "$project": {
                    "_id": 0,
                    "original_filename": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        # Nota: Si no existe el índice, esto fallará o no retornará nada en una instancia local sin Atlas Search.
        # Para desarrollo local sin Atlas, podríamos simular o simplemente retornar vacío.

        try:
           # Intento de búsqueda vectorial (Atlas)
           return list(self.db["faces_daily"].aggregate(pipeline))
        except Exception:
           # Fallback: Búsqueda lineal en memoria (solo para dev/poc)
           # No recomendado para producción con muchos datos
           import numpy as np
           candidates = list(self.db["faces_daily"].find({"embedding": {"$exists": True}}))
           if not candidates:
               return []
           
           query_vec = np.array(embedding)
           query_norm = np.linalg.norm(query_vec)
           
           scores = []
           for doc in candidates:
               cand_vec = np.array(doc["embedding"])
               cand_norm = np.linalg.norm(cand_vec)
               if query_norm > 0 and cand_norm > 0:
                   # Cosine Similarity
                   sim = np.dot(query_vec, cand_vec) / (query_norm * cand_norm)
                   if sim > 0.4: # Filtro mínimo
                       doc["score"] = float(sim)
                       scores.append(doc)
            
           # Ordenar y cortar
           scores.sort(key=lambda x: x["score"], reverse=True)
           return scores[:top_k]



        cursor = self.db[f"faces_{level}"].find(
            {"face_id": {"$in": face_ids}}, {"face_id": 1, "original_filename": 1}
        )
        return {
            item["face_id"]: item.get("original_filename", "")
            for item in cursor
            if item.get("face_id") and item.get("original_filename")
        }

    def list_faces_without_embeddings(
        self, level: str, start: datetime, end: datetime, limit: int = 2000
    ) -> list[dict[str, Any]]:
        query = {
            "created_at": {"$gte": start, "$lt": end},
            "embedding": {"$exists": False},
            "face_thumbnail_b64": {"$exists": True},
        }
        return list(self.db[f"faces_{level}"].find(query).limit(limit))

    def list_faces_with_embeddings(self, level: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        query = {
            "created_at": {"$gte": start, "$lt": end},
            "embedding": {"$exists": True},
        }
        return list(self.db[f"faces_{level}"].find(query))

    def update_face_embedding(self, level: str, image_id: str, embedding: list[float]) -> None:
        self.db[f"faces_{level}"].update_one({"image_id": image_id}, {"$set": {"embedding": embedding}})

    def update_face_cluster(self, level: str, image_id: str, cluster_id: str) -> None:
        self.db[f"faces_{level}"].update_one({"image_id": image_id}, {"$set": {"cluster_id": cluster_id}})

    def insert_event(self, event: dict[str, Any]) -> None:
        self.db["events"].insert_one(event)

    def get_cluster_docs(self, level: str) -> list[dict[str, Any]]:
        return list(self.db[f"faces_{level}"].find({"doc_type": "cluster"}))

    def upsert_cluster_doc(self, level: str, cluster_id: str, updates: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        payload = dict(updates)
        payload["updated_at"] = now
        self.db[f"faces_{level}"].update_one(
            {"doc_type": "cluster", "cluster_id": cluster_id},
            {"$set": payload, "$setOnInsert": {"created_at": now, "doc_type": "cluster", "level": level}},
            upsert=True,
        )

    def create_cluster_doc(self, level: str, document: dict[str, Any]) -> None:
        self.db[f"faces_{level}"].insert_one(document)

    def get_face_collection(self, level: str) -> Collection[Any]:
        return self.db[f"faces_{level}"]

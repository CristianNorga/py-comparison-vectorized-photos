from collections import defaultdict

import hdbscan
import numpy as np


def cluster_embeddings(
    docs: list[dict], min_cluster_size: int, min_samples: int
) -> dict[str, str]:
    if len(docs) < max(2, min_cluster_size):
        return {}

    embeddings = [doc.get("embedding") for doc in docs]
    if not embeddings or any(embedding is None for embedding in embeddings):
        return {}

    matrix = np.array(embeddings, dtype=np.float32)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(matrix)

    assignments: dict[str, str] = {}
    for doc, label in zip(docs, labels, strict=True):
        if int(label) >= 0 and "image_id" in doc:
            assignments[str(doc["image_id"])] = f"daily-{int(label)}"
    return assignments


def compute_cluster_centroids(
    docs: list[dict], assignments: dict[str, str]
) -> dict[str, list[float]]:
    grouped_vectors: dict[str, list[np.ndarray]] = defaultdict(list)

    for doc in docs:
        image_id = str(doc.get("image_id", ""))
        cluster_id = assignments.get(image_id)
        embedding = doc.get("embedding")

        if cluster_id and embedding is not None:
            grouped_vectors[cluster_id].append(np.array(embedding, dtype=np.float32))

    result: dict[str, list[float]] = {}
    for cluster_id, vectors in grouped_vectors.items():
        centroid = np.mean(np.stack(vectors, axis=0), axis=0)
        result[cluster_id] = centroid.astype(float).tolist()
    return result

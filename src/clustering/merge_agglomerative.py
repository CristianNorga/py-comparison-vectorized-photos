import numpy as np


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    arr_a = np.array(vector_a, dtype=np.float32)
    arr_b = np.array(vector_b, dtype=np.float32)

    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / denom)


def find_best_cluster_match(
    centroid: list[float],
    target_clusters: list[dict],
    threshold: float,
) -> dict | None:
    best_doc: dict | None = None
    best_score = -1.0

    for cluster_doc in target_clusters:
        target_centroid = cluster_doc.get("centroid")
        if not isinstance(target_centroid, list):
            continue

        score = cosine_similarity(centroid, target_centroid)
        if score >= threshold and score > best_score:
            best_score = score
            best_doc = cluster_doc

    return best_doc

from clustering.merge_agglomerative import cosine_similarity, find_best_cluster_match


def test_cosine_similarity_identical_vectors() -> None:
    value = cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert value > 0.99


def test_find_best_cluster_match_returns_expected_cluster() -> None:
    clusters = [
        {"cluster_id": "a", "centroid": [1.0, 0.0]},
        {"cluster_id": "b", "centroid": [0.0, 1.0]},
    ]

    match = find_best_cluster_match([0.99, 0.01], clusters, threshold=0.7)
    assert match is not None
    assert match["cluster_id"] == "a"

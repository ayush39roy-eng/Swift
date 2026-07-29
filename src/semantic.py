"""Semantic distance sources for the hybrid confusion-zone distance.

Two independently-derived sources are provided, matching what was used
for CIFAR-10 and CIFAR-100 respectively in the paper:

  - word_embedding_distance: cosine distance between class-name word
    vectors (spaCy en_core_web_md). Falls back to a hardcoded coarse
    grouping if spaCy / model download is unavailable.
  - cifar100_superclass_distance: uses CIFAR-100's native 20-superclass
    grouping (5 classes each) as a semantic prior, requiring no external
    model or internet access.
"""

import numpy as np

# CIFAR-100 official superclass groupings (class indices, torchvision order)
CIFAR100_SUPERCLASSES = {
    0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99], 14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89],
}


def cifar100_superclass_distance(num_classes=100):
    """Semantic distance from CIFAR-100's native superclass structure.
    No external model or internet access required."""
    class_to_sc = {c: sc for sc, members in CIFAR100_SUPERCLASSES.items() for c in members}
    dist = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                dist[i, j] = 0.0
            elif class_to_sc[i] == class_to_sc[j]:
                dist[i, j] = 0.1
            else:
                dist[i, j] = 0.9
    return dist


def word_embedding_distance(class_names, spacy_model="en_core_web_md"):
    """Cosine distance between class-name word embeddings.
    Requires: pip install spacy && python -m spacy download en_core_web_md
    Falls back to a coarse hardcoded grouping if unavailable."""
    n = len(class_names)
    dist = np.zeros((n, n))
    try:
        import spacy
        nlp = spacy.load(spacy_model)
        vecs = [nlp(name.replace("_", " ")).vector for name in class_names]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vi, vj = vecs[i], vecs[j]
                cos = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8))
                dist[i, j] = 1 - cos
    except Exception:
        # Fallback: coarse two-group split (used for CIFAR-10 vehicle/animal
        # split when spaCy / model download is unavailable, e.g. offline).
        raise RuntimeError(
            "spaCy word vectors unavailable; supply a dataset-specific "
            "fallback (see semantic_fallback_cifar10 for an example)."
        )
    return dist


def semantic_fallback_cifar10():
    """Hardcoded CIFAR-10 semantic groups (vehicles vs. animals, with
    domestic-pair and wild-quadruped-pair refinements), for use when
    word embeddings are unavailable (e.g. no internet access)."""
    n = 10
    dist = np.zeros((n, n))
    vehicles = {0, 1, 8, 9}       # plane, car, ship, truck
    domestic = {3, 5}             # cat, dog
    wild_quadruped = {4, 7}       # deer, horse
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0.0
            elif (i in vehicles) == (j in vehicles):
                dist[i, j] = 0.2
            else:
                dist[i, j] = 0.8
    for group in (domestic, wild_quadruped):
        for i in group:
            for j in group:
                if i != j:
                    dist[i, j] = 0.1
    return dist

"""Confusion-zone geometry: Sliced Wasserstein + semantic hybrid distance.

This module implements Steps 1-4 of the BAIT-W methodology:
  1. Extract per-class penultimate-layer feature distributions from a
     reference (clean) classifier.
  2. Compute pairwise Sliced Wasserstein distance between class feature
     distributions (captures full distributional geometry, not just
     centroid proximity).
  3. Compute a semantic distance between class labels (word-embedding
     cosine distance, or dataset-native grouping e.g. CIFAR-100 superclasses).
  4. Combine into a hybrid distance and select each class's K-nearest
     confusion zone.
"""

import numpy as np
import torch
import ot  # Python Optimal Transport (pip install pot)


@torch.no_grad()
def extract_class_features(model, loader, device, num_classes, feat_dim,
                            samples_per_class=500):
    """Run the reference model over the training set and collect per-class
    penultimate-layer features."""
    model.eval()
    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        all_feats.append(model.features(x).cpu())
        all_labels.append(y)
    feats = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    barycenters = torch.zeros(num_classes, feat_dim)
    class_features = {}
    for k in range(num_classes):
        fk = feats[labels == k]
        barycenters[k] = fk.mean(0)
        idx = torch.randperm(len(fk))[:samples_per_class]
        class_features[k] = fk[idx].numpy()
    return barycenters, class_features


def sliced_wasserstein_matrix(class_features, num_classes, n_projections=200, seed=42):
    """Pairwise Sliced Wasserstein distance between class feature distributions."""
    sw = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i < j:
                d = ot.sliced_wasserstein_distance(
                    class_features[i], class_features[j],
                    n_projections=n_projections, seed=seed,
                )
                sw[i, j] = sw[j, i] = float(d)
    return sw


def hybrid_distance(sw_matrix, sem_matrix, alpha):
    """d(i,j) = alpha * SW_norm(i,j) + (1 - alpha) * Sem_norm(i,j)."""
    sw_norm = sw_matrix / (sw_matrix.max() + 1e-8)
    sem_norm = sem_matrix / (sem_matrix.max() + 1e-8)
    hybrid = alpha * sw_norm + (1 - alpha) * sem_norm
    np.fill_diagonal(hybrid, 0.0)
    return torch.tensor(hybrid)


def select_confusion_zones(hybrid_tensor, num_classes, k):
    """For each class, the K classes with smallest hybrid distance (excluding self)."""
    nearest_classes = {}
    for c in range(num_classes):
        d = hybrid_tensor[c].clone()
        d[c] = float("inf")
        nearest_classes[c] = torch.topk(d, k, largest=False).indices.tolist()
    return nearest_classes


def soft_labels_from_zones(nearest_classes, num_classes, k, device):
    """Uniform soft label over each class's confusion zone."""
    soft_labels = torch.zeros(num_classes, num_classes, device=device)
    for c, zone in nearest_classes.items():
        for target in zone:
            soft_labels[c, target] = 1.0 / k
    return soft_labels

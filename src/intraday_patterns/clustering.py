"""K-Shape label discovery fitted exclusively on the training interval."""
from itertools import permutations
import numpy as np
from tslearn.clustering import KShape
from .data import zscore


def prototypes():
    t = np.linspace(0., 1., 240)
    return np.stack([zscore(t), zscore(-t), zscore(np.sin(8 * np.pi * t))])[:, :, None]


def assign_clusters(shapes, centers):
    """Nearest SBD centroid with fresh norms for every unseen sample.

    Do not use KShape.predict in tslearn 0.7: its _cross_dists reuses
    training norms even when the prediction batch has a different length.
    Full linear cross-correlation implements 1 - max(NCC), no square root.
    """
    x = zscore(np.asarray(shapes)[:, :, 0], axis=1)
    c = zscore(np.asarray(centers)[:, :, 0], axis=1)
    distances = np.empty((len(x), len(c)))
    for i, series in enumerate(x):
        for j, center in enumerate(c):
            denom = np.linalg.norm(series) * np.linalg.norm(center)
            if denom < 1e-12:
                raise ValueError("Cannot assign a constant shape or centroid")
            distances[i, j] = 1 - np.correlate(series, center, mode="full").max() / denom
    return distances.argmin(axis=1)


def fit_labels(shapes, train_mask, seed=42, max_iter=100):
    model = KShape(n_clusters=3, init=prototypes(), n_init=1,
                   random_state=seed, max_iter=max_iter)
    model.fit(shapes[train_mask])
    if not np.isfinite(model.inertia_) or len(np.unique(model.labels_)) != 3:
        raise ValueError("K-Shape failed to form 3 nonempty training clusters.")
    centers = model.cluster_centers_
    # Deterministic one-to-one display mapping by zero-lag prototype correlation.
    # Initialization alone is not a guarantee of final economic semantics.
    corr = zscore(centers[:, :, 0], axis=1) @ prototypes()[:, :, 0].T / 240
    assignment = max(permutations(range(3)), key=lambda p: sum(corr[i, p[i]] for i in range(3)))
    mapping = np.asarray(assignment)
    raw = assign_clusters(shapes, centers)
    labels = mapping[raw]
    semantic_centers = np.empty_like(centers)
    for i, mapped in enumerate(mapping):
        semantic_centers[mapped] = centers[i]
    return labels, semantic_centers, {"raw_to_display": mapping.tolist(),
                                     "prototype_correlation": corr.tolist(),
                                     "inertia": float(model.inertia_)}

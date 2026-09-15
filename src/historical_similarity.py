"""
src/historical_similarity.py
-----------------------------
Fits a k-NN model on scaled training features so that, for each new alert,
we can retrieve the single most similar past record with its known label and
similarity score.

This "nearest-neighbour context" is fed into the LLM summary prompt to
ground the narrative in an actual past incident rather than generic language.

No external vector-DB needed -- scikit-learn's NearestNeighbors at NSL-KDD
scale (up to ~125k rows × 41 features) is fast enough for demo purposes.

Usage:
    # During training:
    fit_and_save(X_train_scaled, y_labels, feature_names)

    # During inference:
    result = find_nearest(X_query_scaled, k=1)
    # -> {"label": "neptune", "similarity": 0.94, "distance": 0.23}
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def fit_and_save(
    X_train_scaled: np.ndarray,
    y_labels: np.ndarray,   # fine-grained labels (e.g. "neptune", "normal")
    feature_names: list[str],
    subsample_size: Optional[int] = 20_000,
) -> None:
    """
    Fit a NearestNeighbors index on the training data and persist it.

    We subsample to `subsample_size` rows to keep the index small and
    retrieval fast at demo time.  The sample is stratified so all attack
    types are represented.

    Args:
        X_train_scaled  : Scaled feature matrix (n, d)
        y_labels        : Fine-grained attack labels (n,)
        feature_names   : Column names matching X_train_scaled
        subsample_size  : Max rows to index (None = use all)
    """
    from sklearn.neighbors import NearestNeighbors

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    n = len(X_train_scaled)
    if subsample_size and n > subsample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=subsample_size, replace=False)
        X_idx = X_train_scaled[idx]
        y_idx = y_labels[idx]
        log.info(
            "Subsampled training set to %d rows for NearestNeighbors index "
            "(from %d total).",
            subsample_size, n,
        )
    else:
        X_idx = X_train_scaled
        y_idx = y_labels

    log.info("Fitting NearestNeighbors index on %d rows …", len(X_idx))
    nn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", metric="euclidean", n_jobs=-1)
    nn.fit(X_idx)

    joblib.dump(nn, MODELS_DIR / "nn_index.pkl")
    joblib.dump(y_idx, MODELS_DIR / "nn_labels.pkl")
    joblib.dump(X_idx, MODELS_DIR / "nn_X.pkl")
    log.info("NearestNeighbors index saved (%d rows).", len(X_idx))


def find_nearest(
    X_query_scaled: np.ndarray,   # shape (1, n_features)
    k: int = 1,
) -> dict:
    """
    Find the k nearest historical records for a single query vector.

    Returns a dict with the top-1 result:
        {
          "label": "neptune",
          "category": "dos",
          "similarity": 0.94,   # [0, 1], higher = more similar
          "distance": 0.32,     # raw Euclidean distance in scaled space
          "found": True
        }

    If the index hasn't been built yet, returns a safe "not available" dict.
    """
    nn_path = MODELS_DIR / "nn_index.pkl"
    labels_path = MODELS_DIR / "nn_labels.pkl"

    if not nn_path.exists() or not labels_path.exists():
        return {
            "label": "unknown",
            "category": "unknown",
            "similarity": 0.0,
            "distance": None,
            "found": False,
        }

    nn: object = joblib.load(nn_path)
    y_labels: np.ndarray = joblib.load(labels_path)

    distances, indices = nn.kneighbors(X_query_scaled, n_neighbors=min(k, len(y_labels)))

    # Pick the top-1 closest result
    dist = float(distances[0, 0])
    idx = int(indices[0, 0])
    label = str(y_labels[idx])

    # Convert raw Euclidean distance to a [0, 1] similarity score.
    # We use a soft exponential transform: similarity = exp(-dist).
    # At dist=0 -> similarity=1.0; at dist~2 -> similarity~0.14.
    similarity = float(np.exp(-dist))

    from src.attack_mapping import get_attack_category
    category = get_attack_category(label)

    return {
        "label": label,
        "category": category,
        "similarity": round(similarity, 4),
        "distance": round(dist, 4),
        "found": True,
    }

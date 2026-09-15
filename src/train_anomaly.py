"""
src/train_anomaly.py
--------------------
Trains an Isolation Forest anomaly detector on normal-traffic-only samples.

The key design decision: the Isolation Forest is trained ONLY on normal
traffic.  This means it learns the statistical shape of "normal" and can
flag attack patterns that the supervised binary classifier was never trained
on -- complementing the supervised model's output.

Anomaly score:
  sklearn's IsolationForest.decision_function() returns a score where
  more negative = more anomalous.  We transform this to [0, 1] where
  1 = most anomalous, to match the risk-scoring module's expectations.

Usage:
    python src/train_anomaly.py
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import load_data, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Contamination: expected fraction of anomalies in the training data.
# We train on normal-only, so set this low (the IF will flag the most
# extreme outliers even in the normal subset due to noise).
CONTAMINATION = 0.01


def score_to_0_1(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert raw IsolationForest decision_function scores to [0, 1].

    decision_function: negative = anomalous, positive = normal.
    We negate and min-max scale so that 1.0 = highest anomaly.
    """
    negated = -raw_scores  # now positive = more anomalous
    lo, hi = negated.min(), negated.max()
    if hi == lo:
        return np.zeros_like(negated)
    return (negated - lo) / (hi - lo)


def train_anomaly_detector(use_20_percent: bool = True) -> IsolationForest:
    """Train IsolationForest on normal-traffic-only rows."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load + preprocess (reuses encoders/scaler from a prior classifier training run
    # if they exist; otherwise fits fresh ones -- this is fine because we only need
    # the scaled feature matrix here, not the labels)
    train_df, test_df = load_data(use_20_percent=use_20_percent)
    data = preprocess(train_df, test_df, fit=False)

    X_train_scaled = data["X_train_scaled"]
    y_binary_train = data["y_binary_train"]

    # ------------------------------------------------------------------ #
    # Train ONLY on normal traffic                                         #
    # ------------------------------------------------------------------ #
    normal_mask = y_binary_train == 0
    X_normal = X_train_scaled[normal_mask]
    log.info(
        "Training IsolationForest on %d normal-traffic samples "
        "(out of %d total train rows) …",
        X_normal.shape[0], X_train_scaled.shape[0],
    )

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",       # ~256 samples per tree by default
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_normal)

    # ------------------------------------------------------------------ #
    # Quick evaluation: anomaly scores on normal vs attack test rows       #
    # ------------------------------------------------------------------ #
    X_test = data["X_test_scaled"]
    y_test = data["y_binary_test"]

    raw_scores = iso_forest.decision_function(X_test)
    anomaly_scores = score_to_0_1(raw_scores)

    mean_normal_score = anomaly_scores[y_test == 0].mean()
    mean_attack_score = anomaly_scores[y_test == 1].mean()
    log.info(
        "Anomaly score stats on test set:\n"
        "  Normal traffic  → mean score %.4f (lower is better)\n"
        "  Attack traffic  → mean score %.4f (higher is better)",
        mean_normal_score,
        mean_attack_score,
    )

    # ------------------------------------------------------------------ #
    # Save model + score stats for the risk-scoring module                 #
    # ------------------------------------------------------------------ #
    model_path = MODELS_DIR / "isolation_forest.pkl"
    joblib.dump(iso_forest, model_path)
    log.info("Saved IsolationForest -> %s", model_path)

    # Save normalization parameters so inference uses the same scale
    score_params = {"lo": -iso_forest.decision_function(X_normal).min(),
                    "hi": -iso_forest.decision_function(X_normal).max()}
    # Re-derive lo/hi properly
    all_raw = iso_forest.decision_function(X_train_scaled)
    negated = -all_raw
    joblib.dump(
        {"lo": float(negated.min()), "hi": float(negated.max())},
        MODELS_DIR / "anomaly_score_params.pkl",
    )
    log.info("Saved anomaly score normalisation params.")

    return iso_forest


def get_anomaly_score(X_scaled: np.ndarray) -> np.ndarray:
    """
    Compute per-row anomaly scores in [0, 1] for an already-scaled
    feature matrix X_scaled.

    Loads the saved IsolationForest + normalisation params.
    Returns a 1-D array of float32 anomaly scores.
    """
    iso_forest: IsolationForest = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    params = joblib.load(MODELS_DIR / "anomaly_score_params.pkl")

    raw = iso_forest.decision_function(X_scaled)
    negated = -raw
    lo, hi = params["lo"], params["hi"]
    if hi == lo:
        return np.zeros(len(raw), dtype=np.float32)
    scores = (negated - lo) / (hi - lo)
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    train_anomaly_detector()

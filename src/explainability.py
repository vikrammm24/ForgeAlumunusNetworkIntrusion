"""
src/explainability.py
---------------------
SHAP-based "why was this flagged?" explanations for each alert.

Uses TreeExplainer (fast, exact for tree-based models) to compute per-record
feature contributions from the binary XGBoost classifier.

For each alert we extract the top 3-5 features by |SHAP value| and convert
them into a plain-English reason string, e.g.:

  "Flagged due to: unusually high count (247 vs avg 3.1),
   high srv_serror_rate (0.98 vs avg 0.01),
   abnormal dst_bytes (0 vs avg 2841.7)"

Fallback: if SHAP is unavailable or too slow, we fall back to computing
the top-N features by raw deviation from the normal-class training mean
and producing an equivalent string.  Either path produces the same output
shape so downstream code doesn't need to branch.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Number of top features to surface in the explanation
TOP_N = 5

# Human-readable units / context for selected high-value features
FEATURE_CONTEXT: dict[str, str] = {
    "duration": "connection duration (s)",
    "src_bytes": "bytes sent",
    "dst_bytes": "bytes received",
    "count": "connections to same host (last 2s)",
    "srv_count": "connections to same service (last 2s)",
    "serror_rate": "SYN error rate",
    "srv_serror_rate": "SYN error rate (same service)",
    "rerror_rate": "REJ error rate",
    "dst_host_count": "connections to dst host",
    "dst_host_srv_count": "connections to dst host/service",
    "dst_host_diff_srv_rate": "distinct services on dst host",
    "dst_host_same_src_port_rate": "same-src-port rate on dst host",
    "num_failed_logins": "failed login attempts",
    "logged_in": "login success flag",
    "root_shell": "root shell obtained",
    "su_attempted": "su command attempted",
    "num_root": "root accesses",
    "num_compromised": "compromised conditions",
    "hot": "hot indicators",
}


def _load_shap_explainer():
    """Load the binary XGBoost model and create a SHAP TreeExplainer."""
    import shap  # defer import -- fails gracefully if not installed
    clf = joblib.load(MODELS_DIR / "binary_classifier.pkl")
    explainer = shap.TreeExplainer(clf)
    return explainer


def _load_normal_means() -> Optional[np.ndarray]:
    """Load per-feature means of the normal-traffic training samples (fallback)."""
    path = MODELS_DIR / "normal_feature_means.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def compute_and_save_normal_means(X_train_scaled: np.ndarray, y_binary: np.ndarray, feature_names: list[str]) -> None:
    """
    Precompute and save per-feature means for normal-traffic rows.
    Called during model training so the fallback path has reference values.
    """
    normal_mask = y_binary == 0
    means = X_train_scaled[normal_mask].mean(axis=0)
    joblib.dump(means, MODELS_DIR / "normal_feature_means.pkl")

    # Also save the unscaled (raw) means for human-readable display
    # We'll approximate from scaler params
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    raw_means = scaler.inverse_transform(means.reshape(1, -1)).flatten()
    joblib.dump(raw_means, MODELS_DIR / "normal_feature_raw_means.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")


def _describe_feature(name: str, value: float, normal_mean: float) -> str:
    """Build a short plain-English description of one feature deviation."""
    label = FEATURE_CONTEXT.get(name, name.replace("_", " "))
    direction = "high" if value > normal_mean else "low"

    # Format numbers sensibly
    if abs(value) < 1 and abs(normal_mean) < 1:
        # Rate / fraction -- show as percentage-style
        return f"{label}: {value:.2f} (avg {normal_mean:.2f})"
    elif abs(value) > 100 or abs(normal_mean) > 100:
        return f"{label}: {int(value):,} (avg {normal_mean:.1f})"
    else:
        return f"{label}: {value:.1f} (avg {normal_mean:.1f})"


def explain_record(
    X_scaled: np.ndarray,    # shape (1, n_features) -- the scaled inference row
    X_raw: np.ndarray,       # shape (1, n_features) -- unscaled version
    feature_names: list[str],
) -> dict:
    """
    Generate a SHAP-based explanation for one record.

    Returns:
        {
          "top_features": [{"name", "shap_value", "raw_value", "normal_mean"}, ...],
          "reason_string": "Flagged due to: ...",
          "method": "shap" | "deviation"
        }
    """
    # Try SHAP first
    try:
        explainer = _load_shap_explainer()
        shap_values = explainer.shap_values(X_scaled)

        # For XGBClassifier shap_values returns shape (1, n_features)
        if isinstance(shap_values, list):
            # Some older SHAP versions return [neg_class, pos_class]
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        # Top N by absolute contribution
        top_indices = np.argsort(np.abs(sv))[::-1][:TOP_N]

        # Load raw means for display
        raw_means = _load_normal_means()
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        raw_means_unscaled = (
            scaler.inverse_transform(raw_means.reshape(1, -1)).flatten()
            if raw_means is not None else np.zeros(len(feature_names))
        )

        top_features = [
            {
                "name": feature_names[i],
                "shap_value": float(sv[i]),
                "raw_value": float(X_raw[0, i]),
                "normal_mean": float(raw_means_unscaled[i]),
            }
            for i in top_indices
        ]

        reason_parts = [
            _describe_feature(f["name"], f["raw_value"], f["normal_mean"])
            for f in top_features
        ]
        reason_string = "Flagged due to: " + "; ".join(reason_parts)

        return {
            "top_features": top_features,
            "reason_string": reason_string,
            "method": "shap",
        }

    except Exception as exc:
        log.warning("SHAP failed (%s), falling back to deviation method.", exc)

    # ------------------------------------------------------------------ #
    # Fallback: top-N deviation from normal-traffic mean                   #
    # ------------------------------------------------------------------ #
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    normal_means_scaled = _load_normal_means()
    if normal_means_scaled is None:
        # No reference: just flag the largest raw values
        normal_means_scaled = np.zeros(X_scaled.shape[1])

    deviations = np.abs(X_scaled[0] - normal_means_scaled)
    top_indices = np.argsort(deviations)[::-1][:TOP_N]

    raw_means_unscaled = scaler.inverse_transform(
        normal_means_scaled.reshape(1, -1)
    ).flatten()

    top_features = [
        {
            "name": feature_names[i],
            "shap_value": float(deviations[i]),   # deviation as proxy
            "raw_value": float(X_raw[0, i]),
            "normal_mean": float(raw_means_unscaled[i]),
        }
        for i in top_indices
    ]

    reason_parts = [
        _describe_feature(f["name"], f["raw_value"], f["normal_mean"])
        for f in top_features
    ]
    reason_string = "Flagged due to (deviation from normal): " + "; ".join(reason_parts)

    return {
        "top_features": top_features,
        "reason_string": reason_string,
        "method": "deviation",
    }

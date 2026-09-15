"""
src/predict.py
--------------
Inference pipeline used by the Flask app.

Takes a batch of raw traffic records (list of dicts or a DataFrame slice),
runs them through the full pipeline, and returns enriched result dicts ready
for writing to the alerts database and displaying on the dashboard.

Pipeline per record:
  1. Preprocess (encode + scale)             [preprocessing.py]
  2. Binary classification + confidence      [binary_classifier.pkl]
  3. Attack-category classification          [multiclass_classifier.pkl]
  4. Anomaly score                           [isolation_forest.pkl]
  5. Risk scoring                            [risk_scoring.py]
  6. SHAP-based explainability               [explainability.py]
  7. MITRE mapping                           [attack_mapping.py]
  8. SHAP-to-MITRE feature reasoning         [technique_reasoning.py]
  9. Historical similarity match             [historical_similarity.py]
  10. LLM / template incident summary        [incident_summary.py]
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.attack_mapping import get_mitre_mapping
from src.explainability import explain_record
from src.historical_similarity import find_nearest
from src.incident_summary import generate_summary
from src.preprocessing import FEATURE_COLS, CATEGORICAL_COLS, preprocess_single
from src.risk_scoring import compute_risk_score, is_alert, DEFAULT_RISK_THRESHOLD
from src.technique_reasoning import get_technique_reasoning
from src.train_anomaly import get_anomaly_score

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Lazy-loaded model handles (loaded on first call, not at import time)
_binary_clf = None
_multi_clf = None
_category_classes = None
_feature_names = None
_scaler = None


def _load_models():
    global _binary_clf, _multi_clf, _category_classes, _feature_names, _scaler
    if _binary_clf is None:
        log.info("Loading models from disk …")
        _binary_clf = joblib.load(MODELS_DIR / "binary_classifier.pkl")
        _multi_clf = joblib.load(MODELS_DIR / "multiclass_classifier.pkl")
        _category_classes = joblib.load(MODELS_DIR / "category_classes.pkl")
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        feat_path = MODELS_DIR / "feature_names.pkl"
        _feature_names = joblib.load(feat_path) if feat_path.exists() else FEATURE_COLS
        log.info("Models loaded.")


def predict_batch(records: list[dict]) -> list[dict]:
    """
    Run the full inference pipeline on a list of raw traffic records.

    Each record should be a dict with NSL-KDD feature names as keys.
    Keys not present default to 0.

    Returns a list of result dicts -- one per input record -- with all
    pipeline outputs attached.  Records below the alert threshold have
    is_alert=False but still carry all pipeline outputs for analytics.
    """
    _load_models()

    if not records:
        return []

    # ------------------------------------------------------------------ #
    # 1. Build feature matrix                                              #
    # ------------------------------------------------------------------ #
    rows = []
    for rec in records:
        row = []
        for col in FEATURE_COLS:
            val = rec.get(col, 0)
            if col in CATEGORICAL_COLS:
                # categorical encoding happens inside preprocess_single;
                # for batch we encode inline here
                row.append(val)
            else:
                try:
                    row.append(float(val))
                except (TypeError, ValueError):
                    row.append(0.0)
        rows.append(row)

    # Build raw DataFrame for pass-through
    df_raw = pd.DataFrame(rows, columns=FEATURE_COLS)

    # Apply categorical encoding using saved encoders
    encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        known = set(le.classes_)
        df_raw[col] = df_raw[col].astype(str).map(
            lambda v, le=le, known=known: v if v in known else le.classes_[0]
        )
        df_raw[col] = le.transform(df_raw[col])

    X_raw = df_raw.values.astype(float)
    X_scaled = _scaler.transform(X_raw)

    # ------------------------------------------------------------------ #
    # 2. Binary classification                                             #
    # ------------------------------------------------------------------ #
    clf_pred = _binary_clf.predict(X_scaled)           # 0 or 1
    clf_proba = _binary_clf.predict_proba(X_scaled)[:, 1]  # P(attack)

    # ------------------------------------------------------------------ #
    # 3. Multi-class attack category                                       #
    # ------------------------------------------------------------------ #
    cat_pred = _multi_clf.predict(X_scaled)            # category label

    # ------------------------------------------------------------------ #
    # 4. Anomaly scores                                                    #
    # ------------------------------------------------------------------ #
    anomaly_scores = get_anomaly_score(X_scaled)

    results = []
    for i, rec in enumerate(records):
        raw_attack_category = str(cat_pred[i])
        clf_confidence = float(clf_proba[i])
        anomaly_score = float(anomaly_scores[i])

        # ---------------------------------------------------------------- #
        # Three-way outcome classification (intentional -- do not simplify) #
        #                                                                   #
        # Case A: normal + normal (clf=normal, anomaly=normal)             #
        #   → not an alert; never surfaces in the alert feed               #
        #                                                                   #
        # Case B: normal + anomalous (clf=normal, anomaly=outlier)         #
        #   → genuine unknown-attack candidate; surfaces as alert but with  #
        #     attack_category="UNKNOWN_ANOMALY", no MITRE technique tag,   #
        #     and a summary that clearly says "manual review needed"        #
        #                                                                   #
        # Case C: attack + anything (clf predicted an attack category)     #
        #   → standard alert with MITRE mapping derived from attack type   #
        # ---------------------------------------------------------------- #
        ANOMALY_OUTLIER_THRESHOLD = 0.5   # anomaly_score above this = outlier
        is_clf_normal = raw_attack_category.lower() == "normal"
        is_anomaly_outlier = anomaly_score >= ANOMALY_OUTLIER_THRESHOLD

        # Resolve effective attack_category for downstream use
        if is_clf_normal and is_anomaly_outlier:
            # Case B: classifier says normal but anomaly detector disagrees
            attack_category = "UNKNOWN_ANOMALY"
        else:
            attack_category = raw_attack_category

        # ---------------------------------------------------------------- #
        # 5. Risk score                                                     #
        # ---------------------------------------------------------------- #
        risk = compute_risk_score(clf_confidence, anomaly_score, attack_category)
        alert_flag = is_alert(risk)

        # ---------------------------------------------------------------- #
        # 6. SHAP explainability                                            #
        # ---------------------------------------------------------------- #
        try:
            expl = explain_record(
                X_scaled[i:i+1],
                X_raw[i:i+1],
                _feature_names,
            )
        except Exception as exc:
            log.warning("Explainability failed for record %d: %s", i, exc)
            expl = {
                "top_features": [],
                "reason_string": "Explanation unavailable.",
                "method": "error",
            }

        # ---------------------------------------------------------------- #
        # 7. MITRE mapping                                                  #
        # MITRE only applies when the attack-type classifier predicted a   #
        # known attack category (Case C). For UNKNOWN_ANOMALY (Case B) we  #
        # intentionally return no mapping -- a fabricated MITRE tag would  #
        # be misleading for traffic the classifier called normal.           #
        # ---------------------------------------------------------------- #
        attack_label = str(rec.get("label", raw_attack_category))
        if attack_category == "UNKNOWN_ANOMALY":
            # Case B: no MITRE mapping -- anomaly detector disagreed with classifier
            mitre = {"technique_id": None, "technique_name": None, "tactic": None}
        else:
            mitre = get_mitre_mapping(attack_label)
            if mitre["technique_id"] is None and attack_category != "normal":
                mitre = get_mitre_mapping(attack_category)

        # ---------------------------------------------------------------- #
        # 8. SHAP-to-MITRE technique reasoning                             #
        # ---------------------------------------------------------------- #
        technique_reasoning = get_technique_reasoning(
            technique_id=mitre["technique_id"],
            technique_name=mitre["technique_name"],
            top_features=expl["top_features"],
            attack_category=attack_category,
        )

        # ---------------------------------------------------------------- #
        # 9. Historical similarity                                          #
        # ---------------------------------------------------------------- #
        nearest = find_nearest(X_scaled[i:i+1], k=1)

        # ---------------------------------------------------------------- #
        # 10. Incident summary (only for alerts to conserve API quota)     #
        # ---------------------------------------------------------------- #
        summary_data = {
            "risk_score": risk,
            "attack_category": attack_category,
            "attack_label": attack_label,
            "technique_id": mitre["technique_id"],
            "technique_name": mitre["technique_name"],
            "technique_reasoning": technique_reasoning,
            "reason_string": expl["reason_string"],
            "nearest_label": nearest.get("label", "unknown"),
            "nearest_similarity": nearest.get("similarity", 0.0),
        }

        if alert_flag:
            summary, summary_mode = generate_summary(summary_data)
        else:
            summary = ""
            summary_mode = "none"

        result = {
            # Core classification outputs
            "binary_pred": int(clf_pred[i]),
            "clf_confidence": round(clf_confidence, 4),
            "attack_category": attack_category,
            "attack_label": attack_label,
            "anomaly_score": round(anomaly_score, 4),

            # Risk
            "risk_score": risk,
            "is_alert": alert_flag,

            # Explainability
            "top_features": expl["top_features"],
            "reason_string": expl["reason_string"],
            "explain_method": expl["method"],

            # MITRE
            "technique_id": mitre["technique_id"],
            "technique_name": mitre["technique_name"],
            "tactic": mitre["tactic"],
            "technique_reasoning": technique_reasoning,

            # Historical
            "nearest_label": nearest.get("label"),
            "nearest_similarity": nearest.get("similarity"),
            "nearest_distance": nearest.get("distance"),

            # Summary
            "incident_summary": summary,
            "summary_mode": summary_mode,

            # Metadata
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw": rec,
        }
        results.append(result)

    return results


def predict_single(record: dict) -> dict:
    """Convenience wrapper for a single record."""
    results = predict_batch([record])
    return results[0] if results else {}

"""
src/risk_scoring.py
-------------------
Combines three signals into a single 0-100 risk score per traffic record.

Formula (weights are adjustable constants at the top of this file):

    risk = W_CLASSIFIER * classifier_confidence
         + W_ANOMALY    * anomaly_score
         + W_SEVERITY   * severity_weight
         (clamped to [0, 1], then multiplied by 100)

Where:
  classifier_confidence : float [0, 1]  -- binary classifier's P(attack)
  anomaly_score         : float [0, 1]  -- IsolationForest anomaly score
  severity_weight       : float [0, 1]  -- per-attack-category severity

Severity weights (expert-assigned, documented below):
  normal  → 0.0
  probe   → 0.3  (low impact: information gathering, no direct damage)
  dos     → 0.7  (high impact: service disruption, but typically detectable)
  r2l     → 0.8  (very high: remote code / auth bypass with foothold)
  u2r     → 1.0  (critical: privilege escalation = full system compromise)
"""

# ---------------------------------------------------------------------------
# Tunable weights -- must sum to 1.0
# ---------------------------------------------------------------------------
W_CLASSIFIER = 0.4   # supervised classifier's confidence carries most weight
W_ANOMALY    = 0.3   # anomaly score catches zero-day-style outliers
W_SEVERITY   = 0.3   # attack-type severity adds contextual escalation

# Risk threshold: records at or above this score generate an alert
DEFAULT_RISK_THRESHOLD = 50  # out of 100

# ---------------------------------------------------------------------------
# Per-attack-category severity weights [0, 1]
# ---------------------------------------------------------------------------
SEVERITY_WEIGHTS: dict[str, float] = {
    "normal":         0.0,
    # Case B: classifier said normal but anomaly detector flagged as outlier.
    # We give this a non-zero severity so it can breach the alert threshold
    # when anomaly_score is high, but lower than confirmed attack categories.
    "unknown_anomaly": 0.4,
    "probe":          0.3,
    "dos":            0.7,
    "r2l":            0.8,
    "u2r":            1.0,
}


def get_severity_weight(attack_category: str) -> float:
    """Return the severity weight for a given attack category string."""
    return SEVERITY_WEIGHTS.get(str(attack_category).lower(), 0.5)


def compute_risk_score(
    classifier_confidence: float,
    anomaly_score: float,
    attack_category: str,
) -> float:
    """
    Compute a single risk score in [0, 100].

    Args:
        classifier_confidence : Binary classifier's P(attack) in [0, 1].
        anomaly_score         : Isolation Forest anomaly score in [0, 1].
        attack_category       : Coarse attack category string.

    Returns:
        float in [0, 100] -- higher = higher risk.
    """
    severity = get_severity_weight(attack_category)

    raw = (
        W_CLASSIFIER * classifier_confidence
        + W_ANOMALY    * anomaly_score
        + W_SEVERITY   * severity
    )

    # Clamp to [0, 1] before scaling (floating-point arithmetic can push
    # slightly outside bounds)
    raw = max(0.0, min(1.0, raw))
    return round(raw * 100, 2)


def is_alert(risk_score: float, threshold: int = DEFAULT_RISK_THRESHOLD) -> bool:
    """Return True if *risk_score* meets or exceeds the alert threshold."""
    return risk_score >= threshold

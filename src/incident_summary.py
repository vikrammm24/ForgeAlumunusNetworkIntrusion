"""
src/incident_summary.py
-----------------------
Generates a 2-3 sentence natural-language incident summary for each alert.

Two modes:
  LLM mode   : Calls the Anthropic API (claude-3-haiku, cost-effective)
               with a structured prompt.  Requires ANTHROPIC_API_KEY in .env.
  Offline mode: Template-based fallback.  Produces a readable paragraph
               from the same structured inputs with no API call.

The mode is chosen automatically:
  - If ANTHROPIC_API_KEY is set and the API is reachable → LLM mode.
  - Otherwise → offline template mode.

Summaries are cached per alert_id in models/summary_cache.pkl to avoid
re-generating on every dashboard refresh (and to conserve API quota during
a live demo).

Rate limiting: LLM calls are only made for alerts with risk_score >= threshold
(default 50/100) to avoid burning through quota on low-confidence events.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CACHE_FILE = MODELS_DIR / "summary_cache.pkl"

_CACHE: dict[str, str] = {}
_CACHE_LOADED = False

LLM_MIN_RISK_SCORE = 50   # Only call the API for alerts at or above this score


def _load_cache() -> None:
    global _CACHE, _CACHE_LOADED
    if _CACHE_LOADED:
        return
    import joblib
    if CACHE_FILE.exists():
        try:
            _CACHE = joblib.load(CACHE_FILE)
        except Exception:
            _CACHE = {}
    _CACHE_LOADED = True


def _save_cache() -> None:
    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(_CACHE, CACHE_FILE)


def _cache_key(alert_data: dict) -> str:
    """Derive a stable cache key from the alert's core facts."""
    key_str = (
        f"{alert_data.get('attack_category','')}"
        f"{alert_data.get('technique_id','')}"
        f"{alert_data.get('risk_score','')}"
        f"{alert_data.get('nearest_label','')}"
        # Include the first few feature names to vary by pattern
        f"{str(alert_data.get('top_features', []))[:120]}"
    )
    return hashlib.md5(key_str.encode()).hexdigest()


def _template_summary(alert_data: dict) -> str:
    """
    Offline fallback: produce a readable summary from structured inputs
    without an API call.
    """
    risk = alert_data.get("risk_score", 0)
    category = alert_data.get("attack_category", "unknown").upper()
    attack_label = alert_data.get("attack_label", "unknown")
    technique_id = alert_data.get("technique_id") or "N/A"
    technique_name = alert_data.get("technique_name", "Unknown")
    technique_reasoning = alert_data.get("technique_reasoning", "")
    nearest_label = alert_data.get("nearest_label", "unknown")
    nearest_similarity = alert_data.get("nearest_similarity", 0.0)
    reason_string = alert_data.get("reason_string", "unusual traffic patterns detected")

    # Case B: classifier said normal but anomaly detector flagged this record.
    # Use a clear, non-contradictory summary instead of the generic attack template.
    if category == "UNKNOWN_ANOMALY":
        nearest_str = ""
        if nearest_label and nearest_label != "unknown" and nearest_similarity > 0.3:
            nearest_str = (
                f" Closest historical match: '{nearest_label}' "
                f"({nearest_similarity:.0%} feature similarity)."
            )
        return (
            f"[Offline summary] Traffic classified as normal by the primary model "
            f"but flagged as statistically anomalous by the outlier detector "
            f"(risk score {risk:.0f}/100). {reason_string.rstrip('.')}.{nearest_str} "
            f"No confirmed attack type — recommend manual review to rule out "
            f"a novel or evasive attack pattern."
        )

    severity = "critical" if risk >= 80 else "high" if risk >= 60 else "moderate"

    nearest_str = ""
    if nearest_label and nearest_label != "unknown" and nearest_similarity > 0.3:
        nearest_str = (
            f" This is {nearest_similarity:.0%} similar to a historical "
            f"'{nearest_label}' incident pattern."
        )

    summary = (
        f"[Offline summary] {severity.capitalize()}-severity {category} event "
        f"detected (risk score {risk:.0f}/100), consistent with {technique_name} "
        f"({technique_id}). {reason_string.rstrip('.')}.{nearest_str} "
        f"Recommend reviewing source IP activity and considering rate-limiting "
        f"or blocking if pattern persists."
    )
    return summary


def _llm_summary(alert_data: dict) -> Optional[str]:
    """
    Call the Anthropic API to generate a SOC-analyst-style summary.
    Returns the summary string, or None on error.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        risk = alert_data.get("risk_score", 0)
        category = alert_data.get("attack_category", "unknown")
        attack_label = alert_data.get("attack_label", "unknown")
        technique_id = alert_data.get("technique_id") or "N/A"
        technique_name = alert_data.get("technique_name", "Unknown")
        technique_reasoning = alert_data.get("technique_reasoning", "")
        reason_string = alert_data.get("reason_string", "")
        nearest_label = alert_data.get("nearest_label", "unknown")
        nearest_similarity = alert_data.get("nearest_similarity", 0.0)

        nearest_context = ""
        if nearest_label != "unknown" and nearest_similarity > 0.3:
            nearest_context = (
                f"  - Most similar historical incident: '{nearest_label}' "
                f"pattern at {nearest_similarity:.0%} feature similarity"
            )

        # Case B: anomaly-only alert — craft a prompt that reflects the
        # genuine uncertainty rather than inventing an attack narrative.
        if category.upper() == "UNKNOWN_ANOMALY":
            prompt = f"""You are a senior SOC analyst writing a brief incident note for an anomaly alert.
The primary traffic classifier labeled this record as NORMAL, but the statistical outlier detector flagged it.
Write exactly 2-3 sentences in plain English suitable for a first-responder.
Emphasise the uncertainty, describe what the anomalous features suggest, and recommend manual review.
Do NOT assert it is a confirmed attack. Do NOT start with "I". Do NOT use bullet points.

Alert details:
  - Risk score: {risk:.0f}/100
  - Primary classifier verdict: NORMAL (classifier was not confident this is a known attack type)
  - Outlier detector verdict: statistically anomalous
  - Top contributing features: {reason_string}
{nearest_context}

Write the 2-3 sentence summary:"""
        else:
            prompt = f"""You are a senior SOC analyst writing a brief incident summary for a network alert.
Write exactly 2-3 sentences in plain English that a first-responder could read at a glance.
Be specific about what the traffic pattern suggests, reference the MITRE technique, and give a concrete recommendation.
Do NOT start with "I" and do NOT use bullet points.

Alert details:
  - Risk score: {risk:.0f}/100
  - Detected attack type: {attack_label} (category: {category})
  - MITRE technique: {technique_id} — {technique_name}
  - Why this technique was matched: {technique_reasoning}
  - Top contributing features: {reason_string}
{nearest_context}

Write the 2-3 sentence summary:"""

        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    except Exception as exc:
        log.warning("LLM summary failed: %s. Falling back to template.", exc)
        return None


def generate_summary(alert_data: dict) -> tuple[str, str]:
    """
    Generate (or retrieve from cache) a natural-language incident summary.

    Args:
        alert_data: dict with keys:
            risk_score, attack_category, attack_label, technique_id,
            technique_name, technique_reasoning, reason_string,
            nearest_label, nearest_similarity

    Returns:
        (summary_text, mode)  where mode is "llm" or "template"
    """
    _load_cache()

    # Rate-limit: only call the API for high-confidence alerts
    risk = alert_data.get("risk_score", 0)
    if risk < LLM_MIN_RISK_SCORE:
        summary = _template_summary(alert_data)
        return summary, "template"

    cache_key = _cache_key(alert_data)
    if cache_key in _CACHE:
        log.debug("Summary cache hit for key %s", cache_key[:8])
        return _CACHE[cache_key], "cache"

    # Try LLM first
    summary = _llm_summary(alert_data)
    mode = "llm"
    if summary is None:
        summary = _template_summary(alert_data)
        mode = "template"

    _CACHE[cache_key] = summary
    _save_cache()

    return summary, mode

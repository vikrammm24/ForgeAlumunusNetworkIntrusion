"""
src/technique_reasoning.py
--------------------------
Connects SHAP top-contributing features to WHY a specific MITRE ATT&CK
technique was matched.

Instead of just tagging an alert with "T1498 Network Denial of Service", this
module cross-references the model's top SHAP features against a hand-coded
table of "expected indicator features" per technique and produces a natural-
language reasoning string such as:

  "Matched to T1498 (Network Denial of Service) because the top contributing
   features -- connection count (247 vs avg 3) and same-service connection
   rate (0.98) -- are classic indicators of a flooding attack targeting a
   single service."

If the SHAP features don't clearly match the technique's known indicators,
the module says so honestly rather than forcing a confident explanation:

  "Technique tag T1498 (Network Denial of Service) is based on the attack-type
   classification; feature-level match is inconclusive for this record."
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-technique "expected indicator" feature sets
# Each entry maps a MITRE technique_id to:
#   - features: list of NSL-KDD column names strongly associated with this technique
#   - description: short human-readable string summarising what to look for
# ---------------------------------------------------------------------------
TECHNIQUE_INDICATORS: dict[str, dict] = {

    # ---- DoS ---------------------------------------------------------------
    "T1498": {
        "features": ["count", "srv_count", "serror_rate", "srv_serror_rate",
                     "same_srv_rate", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate"],
        "description": (
            "a flooding attack targeting a single service — expect high "
            "connection counts and same-service rates"
        ),
    },
    "T1498.001": {  # Direct Network Flood (SYN flood = neptune)
        "features": ["count", "srv_count", "serror_rate", "srv_serror_rate",
                     "dst_host_serror_rate", "dst_host_srv_serror_rate"],
        "description": (
            "a SYN flood — expect high SYN error rates and connection counts "
            "with no established connections"
        ),
    },
    "T1498.002": {  # Reflection Amplification (smurf)
        "features": ["count", "dst_bytes", "src_bytes", "wrong_fragment"],
        "description": (
            "a reflection/amplification attack — expect large dst_bytes "
            "with minimal src_bytes and high connection rates"
        ),
    },
    "T1499": {  # Endpoint DoS (teardrop)
        "features": ["wrong_fragment", "urgent", "duration"],
        "description": (
            "an endpoint-targeted DoS exploiting protocol-level vulnerabilities "
            "— expect malformed fragment flags"
        ),
    },
    "T1499.002": {  # Service Exhaustion Flood (back)
        "features": ["count", "srv_count", "dst_bytes", "dst_host_srv_count"],
        "description": (
            "a service exhaustion attack — expect high dst_bytes and "
            "sustained connections to a single service"
        ),
    },

    # ---- Reconnaissance ----------------------------------------------------
    "T1595": {
        "features": ["dst_host_diff_srv_rate", "diff_srv_rate",
                     "dst_host_count", "srv_count", "rerror_rate",
                     "dst_host_same_src_port_rate"],
        "description": (
            "active scanning — expect high rates of distinct services "
            "or ports contacted on the destination host"
        ),
    },
    "T1595.001": {  # IP block scanning (ipsweep)
        "features": ["dst_host_count", "dst_host_diff_srv_rate",
                     "rerror_rate", "serror_rate"],
        "description": (
            "IP block scanning — expect high dst_host_count with varied "
            "rejection/error rates across many hosts"
        ),
    },
    "T1595.002": {  # Vulnerability / port scanning
        "features": ["dst_host_diff_srv_rate", "diff_srv_rate",
                     "dst_host_same_src_port_rate", "srv_count"],
        "description": (
            "port/vulnerability scanning — expect high service diversity "
            "rate and distinct service counts from the same source"
        ),
    },

    # ---- Credential Access -------------------------------------------------
    "T1110": {  # Brute Force (guess_passwd, snmpguess)
        "features": ["num_failed_logins", "logged_in", "count",
                     "srv_count", "duration"],
        "description": (
            "credential brute force — expect elevated failed login attempts "
            "and repeated connection attempts to the same service"
        ),
    },

    # ---- Initial Access ----------------------------------------------------
    "T1190": {  # Exploit Public-Facing Application
        "features": ["hot", "num_compromised", "root_shell",
                     "num_root", "num_access_files", "dst_bytes", "src_bytes"],
        "description": (
            "exploitation of a public-facing service — expect elevated "
            "'hot' indicators, compromised conditions, or root-shell access"
        ),
    },
    "T1078": {  # Valid Accounts (warezclient, warezmaster)
        "features": ["num_failed_logins", "logged_in", "is_guest_login",
                     "is_host_login", "num_access_files"],
        "description": (
            "valid-account abuse — expect eventual successful login after "
            "some failures, with unusual file access patterns"
        ),
    },

    # ---- Privilege Escalation ----------------------------------------------
    "T1068": {  # Exploitation for Privilege Escalation
        "features": ["root_shell", "su_attempted", "num_root",
                     "num_compromised", "num_file_creations", "num_shells"],
        "description": (
            "privilege escalation exploit — expect root_shell acquisition, "
            "su attempts, and elevated num_root / num_compromised counts"
        ),
    },
    "T1055": {  # Process Injection (loadmodule)
        "features": ["root_shell", "num_root", "su_attempted", "num_shells"],
        "description": (
            "process injection for privilege escalation — expect shell "
            "spawning and root access patterns"
        ),
    },

    # ---- Defense Evasion ---------------------------------------------------
    "T1014": {  # Rootkit
        "features": ["root_shell", "num_root", "num_file_creations",
                     "num_compromised", "num_access_files"],
        "description": (
            "rootkit installation — expect root shell + extensive file "
            "creation and access to system files"
        ),
    },

    # ---- C2 / Tunneling ----------------------------------------------------
    "T1572": {  # Protocol Tunneling (httptunnel)
        "features": ["service", "dst_bytes", "src_bytes", "duration",
                     "dst_host_same_src_port_rate"],
        "description": (
            "protocol tunneling — expect large bidirectional byte transfer "
            "over a normally low-bandwidth protocol (e.g. HTTP)"
        ),
    },
    "T1090": {  # Proxy (multihop)
        "features": ["count", "dst_host_diff_srv_rate", "diff_srv_rate",
                     "src_bytes", "dst_bytes"],
        "description": (
            "multi-hop proxy usage — expect unusual traffic routed through "
            "intermediate services or hosts"
        ),
    },

    # ---- Collection --------------------------------------------------------
    "T1056": {  # Input Capture (spy, xsnoop)
        "features": ["duration", "dst_bytes", "src_bytes", "logged_in"],
        "description": (
            "input capture / credential harvesting — expect long-duration "
            "sessions with ongoing data exfiltration patterns"
        ),
    },
}

# Minimum fraction of expected indicator features that must appear in the
# SHAP top features to consider the match "conclusive"
MATCH_THRESHOLD = 0.4   # at least 40% of the indicator features must appear


def get_technique_reasoning(
    technique_id: str | None,
    technique_name: str,
    top_features: list[dict],     # from explainability.explain_record()
    attack_category: str,
) -> str:
    """
    Cross-reference the SHAP top features against the technique's expected
    indicators and return a human-readable reasoning string.

    Args:
        technique_id   : MITRE technique ID (e.g. "T1498") or None.
        technique_name : Human-readable technique name.
        top_features   : List of {"name", "shap_value", "raw_value", "normal_mean"}
                         from explainability.explain_record().
        attack_category: Coarse category ("dos", "probe", "r2l", "u2r", "normal").

    Returns:
        A plain-English string explaining why the technique matched (or
        an honest "match inconclusive" notice if it doesn't).
    """
    if attack_category == "normal" or technique_id is None:
        return "No attack technique flagged -- classified as normal traffic."

    # Get the expected indicator feature set for this technique
    indicators = TECHNIQUE_INDICATORS.get(technique_id, {})

    if not indicators:
        # We have no indicator table for this technique
        return (
            f"Technique tag {technique_id} ({technique_name}) is based on "
            f"the attack-type classification; no feature-level indicator "
            f"definition is available for this technique."
        )

    expected_features: list[str] = indicators["features"]
    description: str = indicators["description"]

    # Names of the top SHAP-driving features
    top_names = {f["name"] for f in top_features}

    # How many expected indicators appear in the top features?
    matched = [f for f in expected_features if f in top_names]
    match_ratio = len(matched) / max(len(expected_features), 1)

    if match_ratio >= MATCH_THRESHOLD:
        # Build a concrete reasoning string with actual values
        feature_detail_parts = []
        for feat in top_features:
            if feat["name"] in matched:
                name = feat["name"].replace("_", " ")
                val = feat["raw_value"]
                mean = feat["normal_mean"]
                if abs(val) > 100 or abs(mean) > 100:
                    feature_detail_parts.append(
                        f"{name} ({int(val):,} vs avg {mean:.1f})"
                    )
                else:
                    feature_detail_parts.append(
                        f"{name} ({val:.2f} vs avg {mean:.2f})"
                    )

        feature_str = ", ".join(feature_detail_parts) if feature_detail_parts else (
            ", ".join(matched[:3])
        )

        return (
            f"Matched to {technique_id} ({technique_name}) because the top "
            f"contributing feature(s) — {feature_str} — are consistent with "
            f"{description}."
        )
    else:
        # Honest inconclusive notice
        return (
            f"Technique tag {technique_id} ({technique_name}) is based on the "
            f"attack-type classification; feature-level match is inconclusive "
            f"for this record (expected indicators: "
            f"{', '.join(expected_features[:4])}{'...' if len(expected_features)>4 else ''}; "
            f"observed top features: {', '.join(list(top_names)[:4])})."
        )

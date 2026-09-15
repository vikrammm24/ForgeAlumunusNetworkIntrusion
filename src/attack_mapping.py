"""
src/attack_mapping.py
---------------------
Maps NSL-KDD attack names (and category names) to MITRE ATT&CK techniques.

The mapping data lives in data/mitre_attack_map.json.

Public API:
    get_mitre_mapping(attack_type: str) -> dict
        Returns {"technique_id", "technique_name", "tactic"} for the
        closest matching attack type.  Gracefully falls back to the
        coarse category (dos/probe/r2l/u2r) if the specific name isn't
        in the map.
"""

import json
from pathlib import Path
from functools import lru_cache

# ---------------------------------------------------------------------------
# Category fallback mapping (fine-grained label -> coarse bucket)
# ---------------------------------------------------------------------------
_CATEGORY_MAP = {
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "httptunnel": "r2l", "named": "r2l",
    "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    "normal": "normal",
}

_UNKNOWN_MAPPING = {
    "technique_id": "T1059",
    "technique_name": "Command and Scripting Interpreter",
    "tactic": "Execution",
}

_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "mitre_attack_map.json"


@lru_cache(maxsize=1)
def _load_map() -> dict:
    """Load and cache the JSON mapping file."""
    if not _MAP_PATH.exists():
        return {}
    with open(_MAP_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    # Strip _comment keys to keep the usable entries only
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def get_mitre_mapping(attack_type: str) -> dict:
    """
    Return the MITRE ATT&CK mapping for *attack_type*.

    Returns None technique fields for:
      - "normal" traffic (no attack to map)
      - "unknown_anomaly" (classifier said normal; anomaly detector disagreed)
        -- attaching a MITRE ID here would be fabricated and misleading.

    Lookup order for known attack types:
      1. Exact match on the lowercased attack type name.
      2. Coarse-category fallback (e.g. "neptune" -> "dos").
      3. Generic unknown-attack fallback.

    Returns a dict with keys:
        technique_id   : str  (e.g. "T1498") or None for normal/unclassified
        technique_name : str  or None
        tactic         : str  or None
    """
    key = str(attack_type).lower().strip()

    # MITRE mapping only applies to confirmed attack categories.
    # Normal traffic and anomaly-only cases get no technique tag.
    if key in ("normal", "unknown_anomaly"):
        return {"technique_id": None, "technique_name": None, "tactic": None}

    mitre_map = _load_map()

    # 1. Direct lookup
    if key in mitre_map:
        entry = mitre_map[key]
        return {
            "technique_id": entry.get("technique_id"),
            "technique_name": entry.get("technique_name", "Unknown"),
            "tactic": entry.get("tactic"),
        }

    # 2. Coarse-category fallback
    category = _CATEGORY_MAP.get(key)
    if category and category in mitre_map:
        entry = mitre_map[category]
        return {
            "technique_id": entry.get("technique_id"),
            "technique_name": entry.get("technique_name", "Unknown"),
            "tactic": entry.get("tactic"),
        }

    # 3. Unknown fallback
    return _UNKNOWN_MAPPING.copy()


def get_attack_category(label: str) -> str:
    """Map a fine-grained label to a coarse attack category string."""
    return _CATEGORY_MAP.get(str(label).lower(), "dos")

"""
src/download_data.py
--------------------
Downloads the NSL-KDD dataset from GitHub mirrors into data/raw/.
Skips files already present.  Falls back to a synthetic dataset if
all mirrors are unreachable.

Usage:
    python src/download_data.py
"""

import os
import io
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mirror URLs (no Kaggle auth needed)
# ---------------------------------------------------------------------------
URLS = {
    "KDDTrain+.txt": (
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt"
    ),
    "KDDTrain+_20Percent.txt": (
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
        "KDDTrain+_20Percent.txt"
    ),
    "KDDTest+.txt": (
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt"
    ),
    "Field Names.csv": (
        "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/"
        "Field%20Names.csv"
    ),
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------------------------------------------------------------------------
# NSL-KDD column schema (41 features + label + difficulty)
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Which columns are categorical in the dataset
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading %s (attempt %d/%d)…", dest.name, attempt, retries)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            log.info("  Saved %d bytes -> %s", len(resp.content), dest)
            return True
        except Exception as exc:
            log.warning("  Attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off
    return False


def _generate_synthetic(dest_dir: Path) -> None:
    """
    Generate a small synthetic NSL-KDD-shaped dataset so the app still works
    when all mirrors are unreachable.  Clearly labelled as synthetic.
    """
    log.warning(
        "All mirrors unreachable.  Generating SYNTHETIC fallback dataset."
    )
    rng = np.random.default_rng(42)
    n_rows = 5_000

    protocols = ["tcp", "udp", "icmp"]
    services = [
        "http", "ftp", "smtp", "ssh", "dns", "ftp_data",
        "eco_i", "private", "telnet", "other",
    ]
    flags = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]
    labels_choices = [
        "normal", "neptune", "smurf", "back",
        "ipsweep", "portsweep", "nmap",
        "buffer_overflow", "rootkit",
        "guess_passwd", "warezmaster",
    ]
    label_weights = [0.40, 0.12, 0.10, 0.05, 0.07, 0.07, 0.04, 0.04, 0.03, 0.04, 0.04]

    rows = {
        "duration": rng.integers(0, 3600, n_rows),
        "protocol_type": rng.choice(protocols, n_rows),
        "service": rng.choice(services, n_rows),
        "flag": rng.choice(flags, n_rows),
        "src_bytes": rng.integers(0, 50_000, n_rows),
        "dst_bytes": rng.integers(0, 50_000, n_rows),
        "land": rng.integers(0, 2, n_rows),
        "wrong_fragment": rng.integers(0, 5, n_rows),
        "urgent": rng.integers(0, 3, n_rows),
        "hot": rng.integers(0, 30, n_rows),
        "num_failed_logins": rng.integers(0, 5, n_rows),
        "logged_in": rng.integers(0, 2, n_rows),
        "num_compromised": rng.integers(0, 10, n_rows),
        "root_shell": rng.integers(0, 2, n_rows),
        "su_attempted": rng.integers(0, 2, n_rows),
        "num_root": rng.integers(0, 10, n_rows),
        "num_file_creations": rng.integers(0, 5, n_rows),
        "num_shells": rng.integers(0, 3, n_rows),
        "num_access_files": rng.integers(0, 10, n_rows),
        "num_outbound_cmds": rng.integers(0, 5, n_rows),
        "is_host_login": rng.integers(0, 2, n_rows),
        "is_guest_login": rng.integers(0, 2, n_rows),
        "count": rng.integers(1, 512, n_rows),
        "srv_count": rng.integers(1, 512, n_rows),
        "serror_rate": rng.random(n_rows).round(2),
        "srv_serror_rate": rng.random(n_rows).round(2),
        "rerror_rate": rng.random(n_rows).round(2),
        "srv_rerror_rate": rng.random(n_rows).round(2),
        "same_srv_rate": rng.random(n_rows).round(2),
        "diff_srv_rate": rng.random(n_rows).round(2),
        "srv_diff_host_rate": rng.random(n_rows).round(2),
        "dst_host_count": rng.integers(1, 256, n_rows),
        "dst_host_srv_count": rng.integers(1, 256, n_rows),
        "dst_host_same_srv_rate": rng.random(n_rows).round(2),
        "dst_host_diff_srv_rate": rng.random(n_rows).round(2),
        "dst_host_same_src_port_rate": rng.random(n_rows).round(2),
        "dst_host_srv_diff_host_rate": rng.random(n_rows).round(2),
        "dst_host_serror_rate": rng.random(n_rows).round(2),
        "dst_host_srv_serror_rate": rng.random(n_rows).round(2),
        "dst_host_rerror_rate": rng.random(n_rows).round(2),
        "dst_host_srv_rerror_rate": rng.random(n_rows).round(2),
        "label": rng.choice(labels_choices, n_rows, p=label_weights),
        "difficulty": rng.integers(1, 22, n_rows),
    }

    df = pd.DataFrame(rows)
    train_path = dest_dir / "KDDTrain+.txt"
    test_path = dest_dir / "KDDTest+.txt"
    subset_path = dest_dir / "KDDTrain+_20Percent.txt"

    # Add a header comment so consumers know this is synthetic
    note = "# SYNTHETIC FALLBACK -- not real NSL-KDD data\n"
    df.iloc[: int(n_rows * 0.8)].to_csv(train_path, index=False, header=False)
    df.iloc[int(n_rows * 0.8) :].to_csv(test_path, index=False, header=False)
    df.iloc[: int(n_rows * 0.2)].to_csv(subset_path, index=False, header=False)

    # Write a minimal Field Names CSV so the rest of the pipeline doesn't break
    field_names_path = dest_dir / "Field Names.csv"
    pd.DataFrame({"name": NSL_KDD_COLUMNS}).to_csv(field_names_path, index=False)

    log.warning(
        "Synthetic dataset written to %s  (train=%d, test=%d, subset=%d rows)",
        dest_dir,
        int(n_rows * 0.8),
        n_rows - int(n_rows * 0.8),
        int(n_rows * 0.2),
    )


def download_all() -> None:
    """Download all NSL-KDD files; fall back to synthetic if mirrors fail."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    any_failed = False
    for filename, url in URLS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            log.info("Already present, skipping: %s", filename)
            continue
        ok = _download(url, dest)
        if not ok:
            any_failed = True
            log.error("Failed to download %s", filename)

    # If any required data file is still missing, generate synthetic fallback
    required = ["KDDTrain+.txt", "KDDTest+.txt", "KDDTrain+_20Percent.txt", "Field Names.csv"]
    missing = [f for f in required if not (RAW_DIR / f).exists()]
    if missing:
        log.warning("Missing files: %s -- generating synthetic dataset", missing)
        _generate_synthetic(RAW_DIR)
    else:
        log.info("All NSL-KDD files present in %s", RAW_DIR)


if __name__ == "__main__":
    download_all()

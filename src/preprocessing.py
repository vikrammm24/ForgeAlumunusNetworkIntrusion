"""
src/preprocessing.py
--------------------
Loads the NSL-KDD dataset, encodes categorical features, scales numeric
features, and returns train/test splits ready for model training.

The fitted encoders + scaler are saved to models/ so that the inference
pipeline uses exactly the same transforms seen during training.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"

# NSL-KDD has 41 features + label + difficulty (no header row in raw files)
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

# Three categorical columns that need encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Features used for model training (everything except label + difficulty)
FEATURE_COLS = [c for c in NSL_KDD_COLUMNS if c not in ("label", "difficulty")]

# NSL-KDD attack category groupings
ATTACK_CATEGORY_MAP = {
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "httptunnel": "r2l", "named": "r2l",
    "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
    # Normal
    "normal": "normal",
}


def _read_raw(filepath: Path) -> pd.DataFrame:
    """
    Load a raw NSL-KDD file.  If the file has a header row (synthetic fallback),
    use it; otherwise assign the canonical column names.
    """
    # Peek at first line to decide whether a header exists
    first_line = filepath.open().readline().strip()
    has_header = first_line.split(",")[0] == "duration" or first_line.startswith("#")

    if has_header:
        df = pd.read_csv(filepath, comment="#")
        # Make sure column names match even if they come from the file
        if "label" not in df.columns and len(df.columns) == len(NSL_KDD_COLUMNS):
            df.columns = NSL_KDD_COLUMNS
    else:
        df = pd.read_csv(filepath, header=None)
        if len(df.columns) == len(NSL_KDD_COLUMNS):
            df.columns = NSL_KDD_COLUMNS
        elif len(df.columns) == len(NSL_KDD_COLUMNS) - 1:
            # Some versions omit the difficulty column
            df.columns = NSL_KDD_COLUMNS[:-1]
            df["difficulty"] = 0

    return df


def load_data(use_20_percent: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test DataFrames.

    Args:
        use_20_percent: If True (default), train on the 20% subset for speed.
                        Set False to use the full training set.

    Returns:
        (train_df, test_df) with raw labels intact.
    """
    train_file = "KDDTrain+_20Percent.txt" if use_20_percent else "KDDTrain+.txt"
    train_path = RAW_DIR / train_file
    test_path = RAW_DIR / "KDDTest+.txt"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset not found in {RAW_DIR}. "
            "Run `python src/download_data.py` first."
        )

    log.info("Loading training data from %s …", train_path.name)
    train_df = _read_raw(train_path)
    log.info("  %d rows loaded", len(train_df))

    log.info("Loading test data from %s …", test_path.name)
    test_df = _read_raw(test_path)
    log.info("  %d rows loaded", len(test_df))

    return train_df, test_df


def _add_attack_category(df: pd.DataFrame) -> pd.DataFrame:
    """Map fine-grained attack label -> coarse category (dos/probe/r2l/u2r/normal)."""
    df = df.copy()
    df["attack_category"] = (
        df["label"].str.lower().map(ATTACK_CATEGORY_MAP).fillna("dos")
    )
    return df


def _add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary_label: 0 = normal, 1 = attack."""
    df = df.copy()
    df["binary_label"] = (df["label"].str.lower() != "normal").astype(int)
    return df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit: bool = True,
) -> dict:
    """
    Full preprocessing pipeline.

    When fit=True (training time):
      - Fits LabelEncoders for categorical columns.
      - Fits StandardScaler on numeric feature columns.
      - Saves all artefacts to models/.

    When fit=False (inference time):
      - Loads the saved artefacts and applies them.

    Returns a dict with keys:
        X_train, y_binary_train, y_category_train,
        X_test,  y_binary_test,  y_category_test,
        feature_names, encoders, scaler,
        X_train_scaled, X_test_scaled, train_df_proc, test_df_proc
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _add_attack_category(_add_binary_label(train_df))
    test_df = _add_attack_category(_add_binary_label(test_df))

    # ------------------------------------------------------------------ #
    # 1. Encode categorical columns                                        #
    # ------------------------------------------------------------------ #
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        if fit:
            # Fit on the union of train and test values to avoid unseen-class errors
            all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
            le.fit(all_vals)
            encoders[col] = le
        else:
            encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
            le = encoders[col]

        # Handle unseen categories at inference time: map to the most frequent
        def _safe_transform(series: pd.Series, encoder: LabelEncoder) -> np.ndarray:
            known = set(encoder.classes_)
            arr = series.astype(str).map(
                lambda v: v if v in known else encoder.classes_[0]
            )
            return encoder.transform(arr)

        train_df[col] = _safe_transform(train_df[col], le)
        test_df[col] = _safe_transform(test_df[col], le)

    if fit:
        joblib.dump(encoders, MODELS_DIR / "label_encoders.pkl")
        log.info("Saved label encoders -> models/label_encoders.pkl")

    # ------------------------------------------------------------------ #
    # 2. Extract feature matrix                                            #
    # ------------------------------------------------------------------ #
    X_train = train_df[FEATURE_COLS].values.astype(float)
    X_test = test_df[FEATURE_COLS].values.astype(float)

    y_binary_train = train_df["binary_label"].values
    y_binary_test = test_df["binary_label"].values

    y_category_train = train_df["attack_category"].values
    y_category_test = test_df["attack_category"].values

    # ------------------------------------------------------------------ #
    # 3. Scale numeric features                                            #
    # ------------------------------------------------------------------ #
    if fit:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        log.info("Saved scaler -> models/scaler.pkl")
    else:
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    log.info(
        "Preprocessing complete: X_train=%s  X_test=%s",
        X_train_scaled.shape,
        X_test_scaled.shape,
    )

    return dict(
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_binary_train=y_binary_train,
        y_binary_test=y_binary_test,
        y_category_train=y_category_train,
        y_category_test=y_category_test,
        feature_names=FEATURE_COLS,
        encoders=encoders,
        scaler=scaler,
        train_df_proc=train_df,
        test_df_proc=test_df,
    )


def preprocess_single(record: dict) -> np.ndarray:
    """
    Preprocess a single inference record (dict of raw feature values).
    Loads saved encoders + scaler from models/.
    Returns a (1, n_features) scaled numpy array.
    """
    encoders: dict[str, LabelEncoder] = joblib.load(MODELS_DIR / "label_encoders.pkl")
    scaler: StandardScaler = joblib.load(MODELS_DIR / "scaler.pkl")

    row = {}
    for col in FEATURE_COLS:
        val = record.get(col, 0)
        if col in CATEGORICAL_COLS:
            le = encoders[col]
            known = set(le.classes_)
            val = str(val) if str(val) in known else le.classes_[0]
            val = le.transform([val])[0]
        row[col] = float(val)

    X = np.array([[row[c] for c in FEATURE_COLS]])
    return scaler.transform(X)

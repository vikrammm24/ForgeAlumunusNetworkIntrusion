"""
src/train_classifier.py
-----------------------
Trains two supervised classifiers on preprocessed NSL-KDD data:

  1. Binary classifier   -- normal (0) vs attack (1)     [XGBoost]
  2. Multi-class classifier -- attack category             [Random Forest]
     Categories: normal / dos / probe / r2l / u2r

Both models are saved to models/ alongside the preprocessing artefacts.

Usage:
    python src/train_classifier.py            # uses 20% subset
    python src/train_classifier.py --full     # uses full training set
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import load_data, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_binary_classifier(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Train an XGBoost binary classifier (normal vs attack).

    XGBoost is chosen over plain Random Forest for:
      - Better handling of class imbalance via scale_pos_weight
      - Faster inference at demo time
      - Native feature importance compatible with SHAP TreeExplainer
    """
    # Compute positive-class weight to handle imbalanced training data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_weight = n_neg / max(n_pos, 1)

    log.info(
        "Training binary XGBoost  (normal=%d, attack=%d, scale_pos_weight=%.2f) …",
        n_neg, n_pos, scale_weight,
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf


def train_multiclass_classifier(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """
    Train a Random Forest multi-class classifier for attack category.

    Random Forest is chosen for multi-class because:
      - Robust to class imbalance with class_weight='balanced'
      - Works well out of the box with little hyperparameter tuning
      - Also compatible with SHAP TreeExplainer
    """
    log.info("Training multi-class Random Forest (attack category) …")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test: np.ndarray, y_test: np.ndarray, label: str) -> dict:
    """Evaluate *clf* on test data and log + return metrics."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    avg = "binary" if len(np.unique(y_test)) == 2 else "macro"

    metrics = {
        "accuracy": acc,
        "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=avg, zero_division=0),
    }

    log.info("--- %s ---", label)
    log.info("  Accuracy : %.4f", metrics["accuracy"])
    log.info("  Precision: %.4f", metrics["precision"])
    log.info("  Recall   : %.4f", metrics["recall"])
    log.info("  F1       : %.4f", metrics["f1"])
    log.info("\n%s", classification_report(y_test, y_pred, zero_division=0))

    return metrics


def train_and_save(use_full: bool = False) -> dict:
    """
    Full training pipeline.  Returns a dict of evaluation metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load and preprocess data                                          #
    # ------------------------------------------------------------------ #
    log.info("use_full=%s → loading data …", use_full)
    train_df, test_df = load_data(use_20_percent=not use_full)
    data = preprocess(train_df, test_df, fit=True)

    X_train = data["X_train_scaled"]
    X_test = data["X_test_scaled"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]
    y_cat_train = data["y_category_train"]
    y_cat_test = data["y_category_test"]

    # ------------------------------------------------------------------ #
    # 2. Train binary classifier                                           #
    # ------------------------------------------------------------------ #
    binary_clf = train_binary_classifier(X_train, y_binary_train)
    binary_path = MODELS_DIR / "binary_classifier.pkl"
    joblib.dump(binary_clf, binary_path)
    log.info("Saved binary classifier -> %s", binary_path)

    binary_metrics = evaluate(binary_clf, X_test, y_binary_test, "Binary Classifier")

    # ------------------------------------------------------------------ #
    # 3. Train multi-class classifier                                      #
    # ------------------------------------------------------------------ #
    multi_clf = train_multiclass_classifier(X_train, y_cat_train)
    multi_path = MODELS_DIR / "multiclass_classifier.pkl"
    joblib.dump(multi_clf, multi_path)
    log.info("Saved multi-class classifier -> %s", multi_path)

    multi_metrics = evaluate(multi_clf, X_test, y_cat_test, "Multi-class Classifier")

    # ------------------------------------------------------------------ #
    # 4. Save class names for downstream use                               #
    # ------------------------------------------------------------------ #
    category_classes = list(multi_clf.classes_)
    joblib.dump(category_classes, MODELS_DIR / "category_classes.pkl")

    log.info("Training complete.")
    return {"binary": binary_metrics, "multiclass": multi_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NSL-KDD classifiers")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full training set instead of the 20% subset",
    )
    args = parser.parse_args()
    train_and_save(use_full=args.full)

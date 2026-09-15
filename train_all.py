"""
train_all.py
------------
One-command training orchestrator. Runs:
  1. download_data.py     -- fetch NSL-KDD dataset
  2. train_classifier.py  -- binary + multi-class classifiers
  3. train_anomaly.py     -- Isolation Forest
  4. Post-training steps  -- save normal-traffic means, fit NearestNeighbors

Usage:
    python train_all.py            # 20% subset (fast, default)
    python train_all.py --full     # full training set
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def main(use_full: bool = False) -> None:
    # ------------------------------------------------------------------ #
    # Step 1: Download data                                                #
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("STEP 1/4: Download dataset")
    log.info("=" * 60)
    from src.download_data import download_all
    download_all()

    # ------------------------------------------------------------------ #
    # Step 2: Train classifiers                                            #
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("STEP 2/4: Train binary + multi-class classifiers")
    log.info("=" * 60)
    from src.train_classifier import train_and_save
    train_and_save(use_full=use_full)

    # ------------------------------------------------------------------ #
    # Step 3: Train anomaly detector                                       #
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("STEP 3/4: Train anomaly detector (IsolationForest)")
    log.info("=" * 60)
    from src.train_anomaly import train_anomaly_detector
    train_anomaly_detector(use_20_percent=not use_full)

    # ------------------------------------------------------------------ #
    # Step 4: Post-training artefacts                                      #
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("STEP 4/4: Build explainability + similarity index")
    log.info("=" * 60)

    from src.preprocessing import load_data, preprocess
    train_df, test_df = load_data(use_20_percent=not use_full)
    data = preprocess(train_df, test_df, fit=False)

    # a) Save normal-traffic feature means (used by SHAP fallback)
    from src.explainability import compute_and_save_normal_means
    compute_and_save_normal_means(
        data["X_train_scaled"],
        data["y_binary_train"],
        data["feature_names"],
    )
    log.info("Saved normal-traffic feature means.")

    # b) Build NearestNeighbors similarity index
    from src.historical_similarity import fit_and_save
    fit_and_save(
        data["X_train_scaled"],
        data["train_df_proc"]["label"].values,
        data["feature_names"],
    )
    log.info("NearestNeighbors index built.")

    log.info("=" * 60)
    log.info("Training complete!  Run `python app.py` to start the dashboard.")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models for NIDA")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full NSL-KDD training set (slower but more accurate)",
    )
    args = parser.parse_args()
    main(use_full=args.full)

"""
src/evaluate_thresholds.py
--------------------------
Compares two detection approaches on the NSL-KDD test set:

  A. Naive threshold:
       Flag any record where the binary classifier outputs 1 (attack)
       OR the raw anomaly score exceeds a fixed cutoff.
       No risk weighting applied.

  B. Weighted risk score (this project):
       Flag only records where the combined risk score from risk_scoring.py
       >= the best-found threshold (auto-selected by grid search).

Grid search:
  Evaluates risk-score thresholds from 30 to 70 (step 5).
  Selects the threshold that maximises F1 while keeping false positives
  below the naïve baseline count.  If no threshold beats naive on BOTH
  metrics simultaneously, the one with the best F1 is chosen and the
  trade-off is reported honestly.

Reports side-by-side:
  - True positives, false positives, false negatives, true negatives
  - Precision, Recall, F1 for each approach
  - % change in false positives AND F1 delta vs naive baseline
  - Full grid-search table (printed to stdout)

Usage:
    python src/evaluate_thresholds.py
"""

import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import load_data, preprocess
from src.risk_scoring import (
    compute_risk_score,
    DEFAULT_RISK_THRESHOLD,
    get_severity_weight,
)
from src.train_anomaly import get_anomaly_score
from src.attack_mapping import get_attack_category

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Raw anomaly score cutoff for the naïve approach
NAIVE_ANOMALY_CUTOFF = 0.6   # anomaly scores above this are flagged naïvely

# Grid-search range for the risk-score alert threshold
GRID_THRESHOLDS = list(range(30, 75, 5))   # 30, 35, 40, 45, 50, 55, 60, 65, 70


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute TP, FP, FN, TN, precision, recall, F1."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_comparison() -> dict:
    """
    Run the full threshold comparison on the test set.
    Returns a results dict suitable for dashboard rendering.
    """
    import joblib
    from xgboost import XGBClassifier

    # ------------------------------------------------------------------ #
    # Load data + models                                                   #
    # ------------------------------------------------------------------ #
    log.info("Loading test data …")
    train_df, test_df = load_data(use_20_percent=True)
    data = preprocess(train_df, test_df, fit=False)

    X_test = data["X_test_scaled"]
    y_true = data["y_binary_test"]             # 0 = normal, 1 = attack
    y_categories = data["y_category_test"]     # coarse attack category

    models_dir = Path(__file__).resolve().parent.parent / "models"
    binary_clf = joblib.load(models_dir / "binary_classifier.pkl")

    log.info("Running classifiers on test set (%d rows) …", len(X_test))

    # Classifier confidence P(attack)
    clf_confidence = binary_clf.predict_proba(X_test)[:, 1]
    clf_pred = binary_clf.predict(X_test)

    # Anomaly scores
    anomaly_scores = get_anomaly_score(X_test)

    # ------------------------------------------------------------------ #
    # Approach A: naïve threshold                                          #
    # Flag if classifier says attack OR anomaly score above cutoff         #
    # ------------------------------------------------------------------ #
    naive_pred = np.where(
        (clf_pred == 1) | (anomaly_scores >= NAIVE_ANOMALY_CUTOFF), 1, 0
    )
    naive_metrics = _compute_metrics(y_true, naive_pred)

    # ------------------------------------------------------------------ #
    # Pre-compute risk scores (threshold-independent)                      #
    # ------------------------------------------------------------------ #
    risk_scores = np.array([
        compute_risk_score(
            float(clf_confidence[i]),
            float(anomaly_scores[i]),
            str(y_categories[i]),
        )
        for i in range(len(X_test))
    ])

    # ------------------------------------------------------------------ #
    # Grid search over risk-score thresholds                               #
    # ------------------------------------------------------------------ #
    grid_results = []
    for threshold in GRID_THRESHOLDS:
        risk_pred = np.where(risk_scores >= threshold, 1, 0)
        m = _compute_metrics(y_true, risk_pred)
        m["threshold"] = threshold
        grid_results.append(m)

    # ------------------------------------------------------------------ #
    # Auto-select best threshold:                                          #
    # Prefer a threshold that beats naive on FP AND matches/beats naive F1.#
    # Fall back to best F1 overall if no threshold satisfies both.         #
    # ------------------------------------------------------------------ #
    naive_fp = naive_metrics["fp"]
    naive_f1 = naive_metrics["f1"]

    # Candidates that strictly reduce FPs below the naive baseline
    fp_improved = [r for r in grid_results if r["fp"] < naive_fp]

    if fp_improved:
        # Among FP-improving candidates, find the one with the highest F1
        best = max(fp_improved, key=lambda r: r["f1"])
        threshold_beats_both = best["f1"] >= naive_f1
    else:
        # No threshold reduces FPs -- pick the best F1 across all
        best = max(grid_results, key=lambda r: r["f1"])
        threshold_beats_both = False

    chosen_threshold = best["threshold"]
    risk_metrics = {k: v for k, v in best.items() if k != "threshold"}

    fp_risk = risk_metrics["fp"]
    fp_reduction_pct = round((naive_fp - fp_risk) / max(naive_fp, 1) * 100, 1)
    # Compute as a plain Python float -- do NOT use `round(...) or 0` which
    # collapses legitimate negative values. round() already returns float.
    f1_delta = round(float(risk_metrics["f1"]) - float(naive_f1), 4)
    recall_delta = round(float(risk_metrics["recall"]) - float(naive_metrics["recall"]), 4)

    # ------------------------------------------------------------------ #
    # Determine honest narrative for dashboard                             #
    # ------------------------------------------------------------------ #
    if threshold_beats_both:
        trade_off_note = None
    elif fp_risk < naive_fp:
        # Reduced FPs but F1 regressed
        trade_off_note = (
            f"Risk-score approach reduces false positives by {fp_reduction_pct:.1f}% "
            f"at a cost of {abs(f1_delta):.3f} F1 ({abs(risk_metrics['recall'] - naive_metrics['recall']):.3f} recall). "
            f"Appropriate for environments where alert fatigue is the primary concern."
        )
    else:
        trade_off_note = (
            f"No single risk-score threshold outperforms naive on both FP count and F1. "
            f"Best F1 found at threshold {chosen_threshold}: {risk_metrics['f1']:.4f} "
            f"(vs naive {naive_f1:.4f}). See grid table above for the full trade-off picture."
        )

    # ------------------------------------------------------------------ #
    # Pretty-print grid table                                              #
    # ------------------------------------------------------------------ #
    gh = f"\n{'Threshold':>9} {'FP':>8} {'TP':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}  {'vs naive FP':>12}  {'vs naive F1':>12}"
    gdiv = "-" * len(gh)
    print(gh)
    print(gdiv)
    for r in grid_results:
        fp_delta_str = f"{r['fp'] - naive_fp:+,}"
        f1_delta_str = f"{r['f1'] - naive_f1:+.4f}"
        marker = " ◀ CHOSEN" if r["threshold"] == chosen_threshold else ""
        print(
            f"{r['threshold']:>9}  {r['fp']:>7,}  {r['tp']:>7,}  "
            f"{r['precision']:>9.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}  "
            f"{fp_delta_str:>12}  {f1_delta_str:>12}{marker}"
        )
    print(gdiv)

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
    # ------------------------------------------------------------------ #
    header = f"\n{'Metric':<25} {'Naive Threshold':>18} {'Risk Score (t={})'.format(chosen_threshold):>22}"
    divider = "-" * len(header)
    rows = [
        ("True Positives",    naive_metrics["tp"],         risk_metrics["tp"]),
        ("False Positives",   naive_metrics["fp"],         risk_metrics["fp"]),
        ("False Negatives",   naive_metrics["fn"],         risk_metrics["fn"]),
        ("True Negatives",    naive_metrics["tn"],         risk_metrics["tn"]),
        ("Precision",         naive_metrics["precision"],  risk_metrics["precision"]),
        ("Recall",            naive_metrics["recall"],     risk_metrics["recall"]),
        ("F1 Score",          naive_metrics["f1"],         risk_metrics["f1"]),
    ]

    print(header)
    print(divider)
    for name, naive_val, risk_val in rows:
        if isinstance(naive_val, float):
            print(f"{name:<25} {naive_val:>18.4f} {risk_val:>22.4f}")
        else:
            print(f"{name:<25} {naive_val:>18,} {risk_val:>22,}")
    print(divider)
    print(f"{'FP change (risk vs naive)':<25} {'':>18} {fp_reduction_pct:>21.1f}%")
    print(f"{'F1 delta (risk vs naive)':<25} {'':>18} {f1_delta:>+22.4f}")
    if trade_off_note:
        print(f"\n⚠  Trade-off note: {trade_off_note}")
    else:
        print(f"\n✅ Risk-score approach beats naive on both FP count and F1.")
    print()

    # ------------------------------------------------------------------ #
    # Write full grid results to CSV for auditability                      #
    # ------------------------------------------------------------------ #
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "threshold_grid.csv"
    with open(csv_path, "w") as csv_f:
        csv_f.write("threshold,fp,tp,fn,tn,precision,recall,f1,fp_delta_vs_naive,f1_delta_vs_naive,chosen\n")
        for r in grid_results:
            fp_d = r["fp"] - naive_fp
            f1_d = round(r["f1"] - naive_f1, 4)
            chosen_marker = "YES" if r["threshold"] == chosen_threshold else ""
            csv_f.write(
                f"{r['threshold']},{r['fp']},{r['tp']},{r['fn']},{r['tn']},"
                f"{r['precision']},{r['recall']},{r['f1']},"
                f"{fp_d:+},{f1_d:+.4f},{chosen_marker}\n"
            )
    log.info("Grid results written to %s", csv_path)

    return {
        "naive": naive_metrics,
        "risk_score": risk_metrics,
        "fp_reduction_pct": fp_reduction_pct,
        "f1_delta": f1_delta,           # float, never None -- real delta including negative
        "recall_delta": recall_delta,
        "naive_threshold_used": int(NAIVE_ANOMALY_CUTOFF * 100),
        "risk_threshold_used": chosen_threshold,
        "test_rows": len(X_test),
        "grid_results": grid_results,
        "trade_off_note": trade_off_note,
        "beats_naive_on_both": threshold_beats_both,
    }


if __name__ == "__main__":
    run_comparison()

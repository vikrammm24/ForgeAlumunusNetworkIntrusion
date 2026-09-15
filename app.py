"""
app.py
------
Flask application entry-point for the Network Intrusion Detection Agent.

Routes:
  GET  /                    → main dashboard (charts + alert feed)
  GET  /api/stats           → JSON: traffic stats for charts
  GET  /api/alerts          → JSON: recent alerts (paginated)
  GET  /api/alert/<id>      → JSON: single alert detail
  GET  /api/threshold-comparison → JSON: naive vs risk-score comparison
  POST /api/simulate        → trigger detection on a batch from the test set
  POST /api/detect          → submit one raw record for instant detection

Run with:
    python app.py
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("app")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "intrusion-demo-key-change-in-prod")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
from flask_sqlalchemy import SQLAlchemy

DB_PATH = BASE_DIR / "data" / "alerts.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Alert(db.Model):
    """One row per triggered alert."""
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(40), nullable=False)
    attack_label = db.Column(db.String(64))
    attack_category = db.Column(db.String(32))
    risk_score = db.Column(db.Float)
    clf_confidence = db.Column(db.Float)
    anomaly_score = db.Column(db.Float)
    binary_pred = db.Column(db.Integer)

    # Explainability
    reason_string = db.Column(db.Text)
    explain_method = db.Column(db.String(16))
    top_features_json = db.Column(db.Text)    # JSON array

    # MITRE
    technique_id = db.Column(db.String(16))
    technique_name = db.Column(db.String(128))
    tactic = db.Column(db.String(64))
    technique_reasoning = db.Column(db.Text)

    # Historical similarity
    nearest_label = db.Column(db.String(64))
    nearest_similarity = db.Column(db.Float)

    # Incident summary
    incident_summary = db.Column(db.Text)
    summary_mode = db.Column(db.String(16))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "attack_label": self.attack_label,
            "attack_category": self.attack_category,
            "risk_score": self.risk_score,
            "clf_confidence": self.clf_confidence,
            "anomaly_score": self.anomaly_score,
            "reason_string": self.reason_string,
            "explain_method": self.explain_method,
            "top_features": json.loads(self.top_features_json or "[]"),
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic": self.tactic,
            "technique_reasoning": self.technique_reasoning,
            "nearest_label": self.nearest_label,
            "nearest_similarity": self.nearest_similarity,
            "incident_summary": self.incident_summary,
            "summary_mode": self.summary_mode,
        }


class TrafficRecord(db.Model):
    """Lightweight log of all processed records (alert + non-alert) for charts."""
    __tablename__ = "traffic"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(40), nullable=False)
    attack_category = db.Column(db.String(32))
    risk_score = db.Column(db.Float)
    is_alert = db.Column(db.Boolean)


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# Lazy model loading (models are big -- only load when a route needs them)
# ---------------------------------------------------------------------------
_models_ready = False


def _check_models_ready() -> bool:
    """Return True if all required model files are present."""
    required = [
        "binary_classifier.pkl",
        "multiclass_classifier.pkl",
        "isolation_forest.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
    ]
    return all((BASE_DIR / "models" / f).exists() for f in required)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Main dashboard page."""
    models_ready = _check_models_ready()
    llm_mode = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    return render_template(
        "dashboard.html",
        models_ready=models_ready,
        llm_mode=llm_mode,
    )


@app.route("/api/stats")
def api_stats():
    """
    Return aggregated traffic statistics for Chart.js:
      - traffic volume per minute (last 60 mins)
      - normal vs suspicious ratio
      - attack-type breakdown
      - risk score distribution (histogram)
    """
    with app.app_context():
        total = TrafficRecord.query.count()
        alert_count = TrafficRecord.query.filter_by(is_alert=True).count()
        normal_count = total - alert_count

        # Attack-type breakdown (last 500 records)
        from sqlalchemy import func
        cat_counts = (
            db.session.query(
                TrafficRecord.attack_category,
                func.count(TrafficRecord.id).label("cnt"),
            )
            .group_by(TrafficRecord.attack_category)
            .all()
        )
        category_breakdown = {row.attack_category: row.cnt for row in cat_counts}

        # Risk score histogram (buckets of 10)
        risk_records = db.session.query(TrafficRecord.risk_score).limit(2000).all()
        buckets = [0] * 10
        for (rs,) in risk_records:
            if rs is not None:
                bucket = min(int(rs // 10), 9)
                buckets[bucket] += 1

        # Traffic volume by minute (last 60 minutes)
        from sqlalchemy import desc
        recent = (
            TrafficRecord.query.order_by(desc(TrafficRecord.id))
            .limit(3600)
            .all()
        )
        volume_map: dict[str, int] = {}
        for rec in recent:
            minute = rec.timestamp[:16] if rec.timestamp else "unknown"
            volume_map[minute] = volume_map.get(minute, 0) + 1
        # Sort and take last 30 minutes
        sorted_minutes = sorted(volume_map.items())[-30:]

    return jsonify({
        "total": total,
        "alerts": alert_count,
        "normal": normal_count,
        "category_breakdown": category_breakdown,
        "risk_histogram": {
            "labels": [f"{i*10}-{i*10+9}" for i in range(10)],
            "values": buckets,
        },
        "volume": {
            "labels": [m for m, _ in sorted_minutes],
            "values": [v for _, v in sorted_minutes],
        },
    })


@app.route("/api/alerts")
def api_alerts():
    """Return the most recent alerts (default 50)."""
    limit = min(int(request.args.get("limit", 50)), 200)
    alerts = (
        Alert.query.order_by(Alert.id.desc()).limit(limit).all()
    )
    return jsonify([a.to_dict() for a in alerts])


@app.route("/api/alert/<int:alert_id>")
def api_alert_detail(alert_id: int):
    """Return full detail for one alert."""
    alert = Alert.query.get_or_404(alert_id)
    return jsonify(alert.to_dict())


@app.route("/api/threshold-comparison")
def api_threshold_comparison():
    """
    Run (or return cached) naive-vs-risk-score comparison.
    Heavy operation -- cached in the DB-adjacent file.
    """
    cache_path = BASE_DIR / "data" / "threshold_comparison.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return jsonify(json.load(f))

    if not _check_models_ready():
        return jsonify({"error": "Models not trained yet."}), 503

    try:
        from src.evaluate_thresholds import run_comparison
        result = run_comparison()
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        return jsonify(result)
    except Exception as exc:
        log.exception("Threshold comparison failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Run detection on a random sample from the test set.
    Query param: n (default 200, max 1000)
    """
    if not _check_models_ready():
        return jsonify({"error": "Models not trained yet. Run the training scripts first."}), 503

    n = min(int(request.args.get("n", 200)), 1000)

    try:
        import pandas as pd
        from src.preprocessing import NSL_KDD_COLUMNS, FEATURE_COLS, load_data, preprocess, ATTACK_CATEGORY_MAP

        # Load test data
        train_df, test_df = load_data(use_20_percent=True)
        data = preprocess(train_df, test_df, fit=False)
        test_raw = data["test_df_proc"]

        # Sample n rows
        sample = test_raw.sample(min(n, len(test_raw)), random_state=None)

        # Inverse-transform categorical columns back to strings for predict_batch
        import joblib
        from src.preprocessing import CATEGORICAL_COLS
        encoders = joblib.load(BASE_DIR / "models" / "label_encoders.pkl")
        scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

        records = []
        for _, row in sample.iterrows():
            rec = row[FEATURE_COLS].to_dict()
            # The categorical columns are already encoded ints -- decode for predict_batch
            for col in CATEGORICAL_COLS:
                le = encoders[col]
                rec[col] = le.inverse_transform([int(rec[col])])[0]
            rec["label"] = row.get("label", "unknown")
            records.append(rec)

        # Run inference
        from src.predict import predict_batch
        results = predict_batch(records)

        # Persist to DB
        new_alerts = 0
        with app.app_context():
            for res in results:
                # Always write traffic record
                tr = TrafficRecord(
                    timestamp=res["timestamp"],
                    attack_category=res["attack_category"],
                    risk_score=res["risk_score"],
                    is_alert=res["is_alert"],
                )
                db.session.add(tr)

                if res["is_alert"]:
                    al = Alert(
                        timestamp=res["timestamp"],
                        attack_label=res["attack_label"],
                        attack_category=res["attack_category"],
                        risk_score=res["risk_score"],
                        clf_confidence=res["clf_confidence"],
                        anomaly_score=res["anomaly_score"],
                        binary_pred=res["binary_pred"],
                        reason_string=res["reason_string"],
                        explain_method=res["explain_method"],
                        top_features_json=json.dumps(res["top_features"]),
                        technique_id=res["technique_id"],
                        technique_name=res["technique_name"],
                        tactic=res["tactic"],
                        technique_reasoning=res["technique_reasoning"],
                        nearest_label=res["nearest_label"],
                        nearest_similarity=res["nearest_similarity"],
                        incident_summary=res["incident_summary"],
                        summary_mode=res["summary_mode"],
                    )
                    db.session.add(al)
                    new_alerts += 1
            db.session.commit()

        alert_results = [r for r in results if r["is_alert"]]
        return jsonify({
            "processed": len(results),
            "new_alerts": new_alerts,
            "alerts": alert_results[:20],  # Return first 20 for immediate display
        })

    except Exception as exc:
        log.exception("Simulation failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Submit a single raw record for instant detection."""
    if not _check_models_ready():
        return jsonify({"error": "Models not trained yet."}), 503

    data = request.get_json(force=True) or {}
    try:
        from src.predict import predict_single
        result = predict_single(data)

        # Persist alert if triggered
        if result.get("is_alert"):
            with app.app_context():
                al = Alert(
                    timestamp=result["timestamp"],
                    attack_label=result["attack_label"],
                    attack_category=result["attack_category"],
                    risk_score=result["risk_score"],
                    clf_confidence=result["clf_confidence"],
                    anomaly_score=result["anomaly_score"],
                    binary_pred=result["binary_pred"],
                    reason_string=result["reason_string"],
                    explain_method=result["explain_method"],
                    top_features_json=json.dumps(result["top_features"]),
                    technique_id=result["technique_id"],
                    technique_name=result["technique_name"],
                    tactic=result["tactic"],
                    technique_reasoning=result["technique_reasoning"],
                    nearest_label=result["nearest_label"],
                    nearest_similarity=result["nearest_similarity"],
                    incident_summary=result["incident_summary"],
                    summary_mode=result["summary_mode"],
                )
                db.session.add(al)
                db.session.commit()

        return jsonify(result)
    except Exception as exc:
        log.exception("Single detection failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    log.info("Starting Network Intrusion Detection Agent on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)

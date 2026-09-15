"""
src/export.py
-------------
Export the alerts table to CSV or PDF.

CSV export:
    Pure pandas — no extra dependencies beyond what's already required.
    Returns an in-memory BytesIO so Flask can stream it directly.

PDF export:
    Uses reportlab (added to requirements.txt).
    Falls back to a plain HTML → text table if reportlab isn't installed,
    returned as a .txt file so the download never errors out.

Public API
----------
    export_csv(alerts: list[dict]) -> BytesIO
    export_pdf(alerts: list[dict]) -> BytesIO
"""

import io
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Columns to include in both formats
EXPORT_COLS = [
    "id", "timestamp", "attack_category", "attack_label",
    "risk_score", "technique_id", "technique_name",
    "clf_confidence", "anomaly_score",
    "reason_string", "incident_summary",
    "nearest_label", "nearest_similarity",
]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(alerts: list[dict]) -> io.BytesIO:
    """
    Return a BytesIO containing a UTF-8 CSV of the given alerts.
    Columns: see EXPORT_COLS.
    """
    import pandas as pd

    rows = []
    for a in alerts:
        row = {col: a.get(col, "") for col in EXPORT_COLS}
        # Flatten nested top_features to a short string
        features = a.get("top_features") or []
        if features:
            row["top_features"] = "; ".join(
                f"{f.get('name','?')}={f.get('raw_value','?')}" for f in features[:5]
            )
        rows.append(row)

    df = pd.DataFrame(rows, columns=EXPORT_COLS)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def export_pdf(alerts: list[dict]) -> tuple[io.BytesIO, str]:
    """
    Return (BytesIO, content_type).
    Tries reportlab first; falls back to plain-text table.
    """
    try:
        return _pdf_reportlab(alerts), "application/pdf"
    except ImportError:
        log.warning("reportlab not installed — falling back to plain-text export")
        return _pdf_fallback_text(alerts), "text/plain; charset=utf-8"


def _pdf_reportlab(alerts: list[dict]) -> io.BytesIO:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        rightMargin=10*mm, leftMargin=10*mm,
        topMargin=12*mm, bottomMargin=12*mm,
        title="NIDA Alert Export",
    )
    styles = getSampleStyleSheet()

    # ---- Title ----
    title_style = ParagraphStyle(
        "title", parent=styles["Heading1"],
        fontSize=16, spaceAfter=4,
        textColor=colors.HexColor("#1d4ed8"),
    )
    sub_style = ParagraphStyle(
        "sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#64748b"), spaceAfter=12,
    )
    ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    elements = [
        Paragraph("Network Intrusion Detection Agent — Alert Report", title_style),
        Paragraph(f"Exported {ts_now} · {len(alerts)} alerts", sub_style),
        Spacer(1, 4*mm),
    ]

    # ---- Table ----
    # Short columns for landscape A4
    col_defs = [
        ("Time",       "timestamp",       28*mm),
        ("Category",   "attack_category", 20*mm),
        ("Label",      "attack_label",    20*mm),
        ("Risk",       "risk_score",      12*mm),
        ("MITRE",      "technique_id",    16*mm),
        ("Technique",  "technique_name",  38*mm),
        ("Confidence", "clf_confidence",  16*mm),
        ("Anomaly",    "anomaly_score",   16*mm),
        ("Summary",    "incident_summary",80*mm),
    ]
    headers = [c[0] for c in col_defs]
    col_widths = [c[2] for c in col_defs]

    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=7, leading=9)

    data = [headers]
    for a in alerts:
        row = []
        for _, key, _ in col_defs:
            val = a.get(key) or ""
            if key == "timestamp":
                val = str(val)[:19].replace("T", " ")
            elif key in ("risk_score", "clf_confidence", "anomaly_score"):
                try:
                    val = f"{float(val):.1f}"
                except (TypeError, ValueError):
                    val = str(val)
            elif key == "incident_summary":
                # Truncate long summaries to avoid huge rows
                val = str(val)[:200]
            else:
                val = str(val)
            row.append(Paragraph(val, small))
        data.append(row)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    # Colour-code risk rows
    RISK_COLORS = {
        "critical": colors.HexColor("#fef2f2"),
        "high":     colors.HexColor("#fffbeb"),
    }
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  8),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]
    for row_idx, a in enumerate(alerts, start=1):
        try:
            score = float(a.get("risk_score", 0))
        except (TypeError, ValueError):
            score = 0
        if score >= 80:
            style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), RISK_COLORS["critical"]))
        elif score >= 60:
            style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), RISK_COLORS["high"]))

    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)

    doc.build(elements)
    buf.seek(0)
    return buf


def _pdf_fallback_text(alerts: list[dict]) -> io.BytesIO:
    """Plain-text fallback when reportlab is absent."""
    lines = [
        "NIDA Alert Export (plain-text fallback — install reportlab for PDF)",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Alerts: {len(alerts)}",
        "-" * 120,
    ]
    for a in alerts:
        lines.append(
            f"[{str(a.get('timestamp',''))[:19]}] "
            f"cat={a.get('attack_category','?'):12} "
            f"risk={a.get('risk_score', 0):5.1f}  "
            f"mitre={a.get('technique_id','N/A'):10}  "
            f"{str(a.get('incident_summary',''))[:80]}"
        )
    buf = io.BytesIO("\n".join(lines).encode("utf-8"))
    buf.seek(0)
    return buf

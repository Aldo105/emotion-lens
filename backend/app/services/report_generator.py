"""
EmotionLens — Report Generator Service

Generates professional PDF and CSV reports from session analysis data.

PDF reports include:
  - Title page with session metadata
  - Emotion distribution table
  - Congruence summary statistics
  - Micro-expression event log
  - Interviewer notes table
  - Key moments section

CSV reports provide a flat emotion timeline suitable for external analysis.
"""

import csv
import io
import os
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak,
)


# ── Color Scheme ─────────────────────────────────────────────────────

BRAND_DARK = colors.HexColor("#1a1a2e")
BRAND_PRIMARY = colors.HexColor("#16213e")
BRAND_ACCENT = colors.HexColor("#0f3460")
BRAND_HIGHLIGHT = colors.HexColor("#e94560")
ROW_LIGHT = colors.HexColor("#f5f5f5")
ROW_WHITE = colors.white
HEADER_TEXT = colors.white
BODY_TEXT = colors.HexColor("#333333")


# ═══════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_pdf_report(session_data: dict, output_path: str) -> str:
    """
    Generate a professional PDF report from session analysis data.

    Args:
        session_data: Complete session data dict (same shape as the
            ``GET /api/reports/{session_id}/data`` response).
        output_path: Absolute path for the output PDF file.

    Returns:
        The output_path on success.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=BRAND_DARK,
        spaceAfter=6 * mm,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=BRAND_ACCENT,
        spaceBefore=8 * mm,
        spaceAfter=4 * mm,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["BodyText"],
        fontSize=10,
        textColor=BODY_TEXT,
        leading=14,
    )
    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        textColor=colors.gray,
    )

    elements: list = []
    session = session_data.get("session", {})
    summary = session_data.get("summary")

    # ── Title Page ───────────────────────────────────────────────────
    elements.append(Spacer(1, 20 * mm))
    elements.append(Paragraph("EmotionLens — Session Report", title_style))
    elements.append(Spacer(1, 6 * mm))

    meta_rows = [
        ["Session Name", session.get("name", "—")],
        ["Candidate", session.get("candidate_name") or "—"],
        ["Date", _format_datetime(session.get("created_at"))],
        ["Duration", _format_duration(session.get("duration_seconds"))],
        ["Status", session.get("status", "—")],
        ["Input Type", session.get("input_type", "—")],
    ]
    meta_table = Table(meta_rows, colWidths=[50 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), BRAND_ACCENT),
        ("TEXTCOLOR", (1, 0), (1, -1), BODY_TEXT),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(meta_table)

    # ── Emotion Distribution ─────────────────────────────────────────
    if summary and summary.get("emotion_distribution"):
        elements.append(Paragraph("Emotion Distribution", heading_style))

        dist = summary["emotion_distribution"]
        emo_header = ["Emotion", "Percentage"]
        emo_rows = [emo_header]
        for emotion, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            emo_rows.append([
                emotion.capitalize(),
                f"{pct * 100:.1f}%",
            ])

        if summary.get("dominant_emotion"):
            emo_rows.append(["Dominant Emotion", summary["dominant_emotion"].capitalize()])

        emo_table = Table(emo_rows, colWidths=[80 * mm, 80 * mm])
        emo_table.setStyle(_table_style(len(emo_rows)))
        elements.append(emo_table)

    # ── Congruence Summary ───────────────────────────────────────────
    if summary and summary.get("average_congruence") is not None:
        elements.append(Paragraph("Congruence / Trustworthiness", heading_style))

        cong_rows = [
            ["Metric", "Score"],
            ["Average Congruence", f"{summary['average_congruence']:.1f} / 100"],
            ["Minimum Congruence", f"{summary.get('min_congruence', 0):.1f}"],
            ["Maximum Congruence", f"{summary.get('max_congruence', 0):.1f}"],
        ]

        if summary.get("average_nervousness") is not None:
            cong_rows.append(["Avg Nervousness", f"{summary['average_nervousness']:.3f}"])
        if summary.get("average_confidence") is not None:
            cong_rows.append(["Avg Confidence", f"{summary['average_confidence']:.3f}"])
        if summary.get("nervousness_peaks") is not None:
            cong_rows.append(["Nervousness Peaks", str(summary["nervousness_peaks"])])
        if summary.get("average_model_confidence") is not None:
            cong_rows.append(["Avg Model Confidence", f"{summary['average_model_confidence']:.3f}"])

        cong_table = Table(cong_rows, colWidths=[80 * mm, 80 * mm])
        cong_table.setStyle(_table_style(len(cong_rows)))
        elements.append(cong_table)

    # ── Micro-Expression Log ─────────────────────────────────────────
    micros = session_data.get("micro_expressions", [])
    if micros:
        elements.append(Paragraph("Micro-Expression Log", heading_style))
        elements.append(Paragraph(
            f"Total: {len(micros)} events detected",
            body_style,
        ))
        elements.append(Spacer(1, 3 * mm))

        micro_header = ["Time (s)", "Emotion", "Duration (ms)", "Contradictory", "Description"]
        micro_rows = [micro_header]
        for m in micros:
            micro_rows.append([
                f"{m.get('timestamp', 0):.1f}",
                m.get("detected_emotion", "—"),
                f"{m.get('duration_ms', 0):.0f}",
                "Yes" if m.get("is_contradictory") else "No",
                _truncate(m.get("description", ""), 50),
            ])

        micro_table = Table(
            micro_rows,
            colWidths=[22 * mm, 28 * mm, 28 * mm, 28 * mm, 64 * mm],
        )
        micro_table.setStyle(_table_style(len(micro_rows)))
        elements.append(micro_table)

    # ── Interviewer Notes ────────────────────────────────────────────
    notes = session_data.get("notes", [])
    if notes:
        elements.append(Paragraph("Interviewer Notes", heading_style))

        note_header = ["Time (s)", "Tag", "Content", "Emotion", "Congruence"]
        note_rows = [note_header]
        for n in notes:
            note_rows.append([
                f"{n.get('timestamp', 0):.1f}",
                n.get("tag") or "—",
                _truncate(n.get("content", ""), 45),
                n.get("emotion_at_time") or "—",
                f"{n.get('congruence_at_time', 0):.0f}" if n.get("congruence_at_time") else "—",
            ])

        note_table = Table(
            note_rows,
            colWidths=[22 * mm, 25 * mm, 60 * mm, 28 * mm, 28 * mm],
        )
        note_table.setStyle(_table_style(len(note_rows)))
        elements.append(note_table)

    # ── Key Moments ──────────────────────────────────────────────────
    key_moments = (summary or {}).get("key_moments", [])
    if key_moments:
        elements.append(Paragraph("Key Moments", heading_style))

        km_header = ["Time (s)", "Type", "Detail"]
        km_rows = [km_header]
        for km in key_moments[:30]:  # Cap at 30 rows
            km_rows.append([
                f"{km.get('timestamp', 0):.1f}",
                km.get("type", "—"),
                _truncate(km.get("detail", ""), 65),
            ])

        km_table = Table(km_rows, colWidths=[25 * mm, 35 * mm, 110 * mm])
        km_table.setStyle(_table_style(len(km_rows)))
        elements.append(km_table)

    # ── Footer ───────────────────────────────────────────────────────
    elements.append(Spacer(1, 10 * mm))
    elements.append(Paragraph(
        f"Generated by EmotionLens on {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "This report is an analytical aid and does not constitute a definitive "
        "assessment of character or truthfulness.",
        small_style,
    ))

    # Build PDF
    doc.build(elements)
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# CSV REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_csv_report(session_data: dict, output_path: str) -> str:
    """
    Generate a CSV report with the emotion timeline.

    Columns: timestamp, emotion, confidence, congruence_score, action_units

    Args:
        session_data: Complete session data dict.
        output_path: Absolute path for the output CSV file.

    Returns:
        The output_path on success.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    timeline = session_data.get("emotion_timeline", [])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "emotion", "confidence",
            "congruence_score", "action_units",
        ])

        for record in timeline:
            action_units_str = ""
            if record.get("action_units"):
                action_units_str = "; ".join(
                    f"{k}={v}" for k, v in record["action_units"].items()
                )

            writer.writerow([
                record.get("timestamp", ""),
                record.get("emotion", ""),
                record.get("confidence", ""),
                record.get("congruence_score", ""),
                action_units_str,
            ])

    return output_path


def generate_csv_string(session_data: dict) -> str:
    """
    Generate a CSV string (for streaming responses) instead of writing
    to disk.

    Returns:
        CSV content as a string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "emotion", "confidence",
        "congruence_score", "action_units",
    ])

    timeline = session_data.get("emotion_timeline", [])
    for record in timeline:
        action_units_str = ""
        if record.get("action_units"):
            action_units_str = "; ".join(
                f"{k}={v}" for k, v in record["action_units"].items()
            )

        writer.writerow([
            record.get("timestamp", ""),
            record.get("emotion", ""),
            record.get("confidence", ""),
            record.get("congruence_score", ""),
            action_units_str,
        ])

    return output.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _table_style(row_count: int) -> TableStyle:
    """
    Build a clean, professional table style with dark headers
    and alternating row colors.
    """
    style_commands = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), HEADER_TEXT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),

        # Body rows
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), BODY_TEXT),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 4),

        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LINEBELOW", (0, 0), (-1, 0), 1, BRAND_DARK),

        # Alignment
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    # Alternating row colors
    for i in range(1, row_count):
        bg = ROW_LIGHT if i % 2 == 0 else ROW_WHITE
        style_commands.append(("BACKGROUND", (0, i), (-1, i), bg))

    return TableStyle(style_commands)


def _format_datetime(iso_str: Optional[str]) -> str:
    """Format an ISO datetime string into a human-readable form."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d  %H:%M:%S")
    except (ValueError, TypeError):
        return str(iso_str)


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds into HH:MM:SS."""
    if seconds is None:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, appending '…' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

"""
EmotionLens — Report Generator Service

Generates professional PDF and CSV reports from session analysis data.

PDF reports include:
  - Professional cover page with brand banner, candidate info card,
    key metrics row, executive summary and key findings
  - Emotion distribution table
  - Congruence summary statistics
  - Micro-expression event log
  - Interviewer notes table
  - Key moments section
  - Comparative analysis (emotions vs micro-expressions)
  - Subject validation & feedback
  - Interview analysis (radar chart, patterns, red flags, correlations,
    recommendations, noise filter stats)

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
import math
from reportlab.graphics.shapes import Drawing, Polygon, Line, String, Circle, Group
from reportlab.graphics import renderPDF


# ── Color Scheme ─────────────────────────────────────────────────────

BRAND_DARK      = colors.HexColor("#1a1a2e")
BRAND_PRIMARY   = colors.HexColor("#16213e")
BRAND_ACCENT    = colors.HexColor("#0f3460")
BRAND_HIGHLIGHT = colors.HexColor("#e94560")
ROW_LIGHT       = colors.HexColor("#f5f5f5")
ROW_WHITE       = colors.white
HEADER_TEXT     = colors.white
BODY_TEXT       = colors.HexColor("#333333")


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

    # ── Custom styles ─────────────────────────────────────────────────
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
    label_style = ParagraphStyle(
        "LabelStyle",
        parent=styles["BodyText"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        leading=10,
    )

    elements: list = []
    session  = session_data.get("session", {})
    summary  = session_data.get("summary")
    analysis = session_data.get("interview_analysis")

    page_width = A4[0] - 30 * mm  # usable width: left + right = 15mm each

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COVER PAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Brand banner ──────────────────────────────────────────────────
    banner_data = [[
        Paragraph(
            '<font color="white"><b>EmotionLens</b></font>',
            ParagraphStyle("BannerTitle", parent=styles["Title"],
                           fontSize=28, textColor=colors.white,
                           fontName="Helvetica-Bold", leading=34),
        ),
        Paragraph(
            '<font color="#e94560">Interview Analysis Report</font>',
            ParagraphStyle("BannerSub", parent=styles["BodyText"],
                           fontSize=13, textColor=BRAND_HIGHLIGHT,
                           fontName="Helvetica", leading=16,
                           alignment=2),
        ),
    ]]
    banner_table = Table(banner_data, colWidths=[page_width * 0.55, page_width * 0.45])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BRAND_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(banner_table)
    elements.append(Spacer(1, 8 * mm))

    # ── Candidate info card ───────────────────────────────────────────
    candidate_name = session.get("candidate_name") or "Not specified"
    session_name   = session.get("name", "—")
    session_date   = _format_datetime(session.get("created_at"))
    session_dur    = _format_duration(session.get("duration_seconds"))
    input_type     = session.get("input_type", "webcam").replace("_", " ").title()

    def _info_val(text, size=10):
        return Paragraph(text, ParagraphStyle(
            "InfoV", parent=styles["BodyText"],
            fontSize=size, textColor=BODY_TEXT))

    col_w = page_width / 5
    info_data = [
        [
            Paragraph("CANDIDATE",   label_style),
            Paragraph("SESSION",     label_style),
            Paragraph("DATE",        label_style),
            Paragraph("DURATION",    label_style),
            Paragraph("INPUT",       label_style),
        ],
        [
            Paragraph(f"<b>{candidate_name}</b>",
                      ParagraphStyle("IV0", parent=styles["BodyText"],
                                     fontSize=11, fontName="Helvetica-Bold",
                                     textColor=BRAND_DARK)),
            _info_val(session_name, 9),
            _info_val(session_date, 9),
            _info_val(session_dur),
            _info_val(input_type),
        ],
    ]
    info_table = Table(info_data, colWidths=[col_w] * 5)
    info_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.5, colors.HexColor("#dddddd")),
        ("LINEBELOW",     (0, 1), (-1, 1),  1.5, BRAND_HIGHLIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8 * mm))

    # ── Key metrics row ───────────────────────────────────────────────
    dim_scores = (analysis or {}).get("dimension_scores", {})
    overall_score  = dim_scores.get("overall_score") if dim_scores else None
    avg_congruence = (summary or {}).get("average_congruence")
    dominant_emo   = (summary or {}).get("dominant_emotion", "—")
    micro_count    = len(session_data.get("micro_expressions", []))
    nerv_peaks     = (summary or {}).get("nervousness_peaks", 0)

    def _score_color(score):
        if score is None:
            return BODY_TEXT
        return (colors.HexColor("#27ae60") if score >= 70
                else colors.HexColor("#f39c12") if score >= 40
                else colors.HexColor("#c0392b"))

    def _metric_cell_table(value_str, label_str, value_color=BRAND_DARK):
        return Table(
            [[Paragraph(value_str,
                        ParagraphStyle("MV", parent=styles["BodyText"],
                                       fontSize=22, fontName="Helvetica-Bold",
                                       textColor=value_color, leading=26))],
             [Paragraph(label_str, label_style)]],
            colWidths=[col_w],
        )

    score_str = f"{overall_score:.0f}/100" if overall_score is not None else "N/A"
    cong_str  = f"{avg_congruence:.0f}/100" if avg_congruence is not None else "N/A"

    metrics_data = [[
        _metric_cell_table(score_str, "Overall Interview Score", _score_color(overall_score)),
        _metric_cell_table(cong_str,  "Avg. Congruence",         _score_color(avg_congruence)),
        _metric_cell_table(dominant_emo.capitalize(), "Dominant Emotion"),
        _metric_cell_table(str(micro_count), "Micro-Expressions"),
        _metric_cell_table(
            str(nerv_peaks), "Nervousness Peaks",
            colors.HexColor("#c0392b") if nerv_peaks > 3 else BRAND_DARK
        ),
    ]]
    metrics_table = Table(metrics_data, colWidths=[col_w] * 5)
    metrics_table.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 1,   colors.HexColor("#e0e0e0")),
        ("LINEAFTER",     (0, 0), (3,  0),  0.5, colors.HexColor("#e0e0e0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 8 * mm))

    # ── Executive Summary ────────────────────────────────────────────
    elements.append(Paragraph("Executive Summary", heading_style))

    patterns  = (analysis or {}).get("behavioral_patterns", []) or []
    red_flags = (analysis or {}).get("red_flags",           []) or []
    recs      = (analysis or {}).get("recommendations",     []) or []

    positive_patterns = [p for p in patterns
                         if p.get("type") in ("stress_recovery", "authentic_engagement")]
    concern_patterns  = [p for p in patterns
                         if p.get("type") not in ("stress_recovery", "authentic_engagement")]

    if dim_scores:
        scored_dims = [(k, v) for k, v in dim_scores.items() if k != "overall_score"]
        scored_dims_sorted = sorted(scored_dims, key=lambda x: x[1], reverse=True)
        top_k, top_v = scored_dims_sorted[0]  if scored_dims_sorted else ("—", 0)
        low_k, low_v = scored_dims_sorted[-1] if scored_dims_sorted else ("—", 0)
        top_label = top_k.replace("_", " ").title()
        low_label = low_k.replace("_", " ").title()

        exec_text = (
            f"This report presents the automated behavioral and emotional analysis of candidate "
            f"<b>{candidate_name}</b> during a {session_dur} interview session conducted on "
            f"{session_date}. "
        )
        if overall_score is not None:
            level = ("strong" if overall_score >= 70
                     else "moderate" if overall_score >= 40
                     else "low")
            exec_text += (
                f"The candidate achieved an overall interview score of "
                f"<b>{overall_score:.0f}/100</b>, indicating a <b>{level}</b> overall performance. "
            )
        exec_text += (
            f"The highest-scoring dimension was <b>{top_label}</b> ({top_v:.0f}/100), "
            f"while the area most in need of follow-up was <b>{low_label}</b> ({low_v:.0f}/100). "
        )
        if avg_congruence is not None:
            cong_level = ("high" if avg_congruence >= 70
                          else "moderate" if avg_congruence >= 40
                          else "low")
            exec_text += (
                f"Emotional congruence averaged <b>{avg_congruence:.0f}/100</b> ({cong_level}), "
                f"suggesting {'authentic and consistent' if avg_congruence >= 70 else 'potentially inconsistent'} "
                f"self-presentation throughout the interview. "
            )
        micros_list = session_data.get("micro_expressions", [])
        if micro_count:
            contradictory_count = sum(1 for m in micros_list if m.get("is_contradictory"))
            if contradictory_count:
                exec_text += (
                    f"A total of <b>{micro_count}</b> micro-expression(s) were detected, "
                    f"including <b>{contradictory_count}</b> contradictory event(s) suggesting emotional masking. "
                )
            else:
                exec_text += (
                    f"A total of <b>{micro_count}</b> micro-expression(s) were detected, "
                    f"all consistent with displayed emotions. "
                )
        elements.append(Paragraph(exec_text, body_style))
    else:
        elements.append(Paragraph(
            f"This report presents the automated behavioral analysis of candidate "
            f"<b>{candidate_name}</b> during a {session_dur} interview session "
            f"conducted on {session_date}.",
            body_style
        ))

    elements.append(Spacer(1, 4 * mm))

    # Key findings bullets
    findings_rows = []
    if positive_patterns:
        pnames = ", ".join(p["type"].replace("_", " ").title() for p in positive_patterns)
        findings_rows.append([
            Paragraph('<font color="#27ae60">✔</font>', body_style),
            Paragraph(f"<b>Strengths detected:</b> {pnames}.", body_style),
        ])
    if concern_patterns:
        cnames = ", ".join(p["type"].replace("_", " ").title() for p in concern_patterns[:3])
        findings_rows.append([
            Paragraph('<font color="#f39c12">⚠</font>', body_style),
            Paragraph(f"<b>Areas of concern:</b> {cnames}.", body_style),
        ])
    if red_flags:
        findings_rows.append([
            Paragraph('<font color="#c0392b">✗</font>', body_style),
            Paragraph(
                f"<b>{len(red_flags)} red flag(s)</b> detected — see the Red Flags section for details.",
                body_style
            ),
        ])
    if recs:
        high_recs = [r for r in recs if r.get("priority") == "high"]
        if high_recs:
            findings_rows.append([
                Paragraph('<font color="#0f3460">→</font>', body_style),
                Paragraph(
                    f"<b>Top recommendation:</b> {high_recs[0].get('text', '')}",
                    body_style
                ),
            ])

    if findings_rows:
        findings_table = Table(
            findings_rows, colWidths=[8 * mm, page_width - 8 * mm]
        )
        findings_table.setStyle(TableStyle([
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 2),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("ALIGN",         (0, 0), (0,  -1), "CENTER"),
            ("LINEAFTER",     (0, 0), (0,  -1), 0.5, colors.HexColor("#dddddd")),
        ]))
        elements.append(findings_table)

    elements.append(Spacer(1, 4 * mm))

    # Disclaimer strip
    disclaimer_data = [[
        Paragraph(
            "⚠  This report is an analytical aid generated by EmotionLens. It is not a "
            "definitive assessment of character, truthfulness, or suitability. All results "
            "must be interpreted by a qualified professional within an appropriate ethical framework.",
            ParagraphStyle("Disclaimer", parent=small_style,
                           textColor=colors.HexColor("#666666"), leading=11)
        )
    ]]
    disc_table = Table(disclaimer_data, colWidths=[page_width])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fff8e1")),
        ("LINEABOVE",     (0, 0), (-1,  0), 1, colors.HexColor("#f39c12")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    elements.append(disc_table)

    # ── Page break: end of cover, begin detailed sections ────────────
    elements.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DETAILED ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    # ── Comparative Analysis: Emotions vs Micro-Expressions ──────────
    if micros:
        elements.append(PageBreak())
        elements.append(Paragraph(
            "Comparative Analysis: Emotions vs Micro-Expressions",
            heading_style,
        ))
        elements.append(Paragraph(
            "This table correlates each detected micro-expression with the "
            "emotion being displayed at that moment, revealing potential "
            "emotional incongruences during the session.",
            body_style,
        ))
        elements.append(Spacer(1, 3 * mm))

        comp_header = [
            "Time", "Displayed\nEmotion", "Micro-Expression\nDetected",
            "AUs Involved", "Duration", "Contradiction", "Interpretation",
        ]
        comp_rows = [comp_header]

        for m in micros:
            ts = m.get("timestamp", 0)
            mins = int(ts // 60)
            secs = int(ts % 60)
            time_str = f"{mins:02d}:{secs:02d}"

            dominant = (m.get("dominant_emotion_at_time") or "neutral").capitalize()
            detected = (m.get("detected_emotion") or "unknown").capitalize()
            aus = ", ".join(m.get("action_units_involved", []))
            duration = f"{m.get('duration_ms', 0):.0f}ms"
            is_contra = m.get("is_contradictory", False)
            contradiction_str = "YES" if is_contra else "No"

            if is_contra:
                interpretation = (
                    f"Subject showed {dominant.lower()} but briefly "
                    f"revealed {detected.lower()} — possible suppression"
                )
            else:
                interpretation = (
                    f"Congruent — {detected.lower()} reinforces "
                    f"the displayed {dominant.lower()}"
                )

            comp_rows.append([
                time_str, dominant, detected,
                _truncate(aus, 20), duration,
                contradiction_str, _truncate(interpretation, 45),
            ])

        comp_table = Table(
            comp_rows,
            colWidths=[14 * mm, 22 * mm, 22 * mm, 22 * mm, 16 * mm, 20 * mm, 54 * mm],
        )

        comp_style_cmds = _table_style(len(comp_rows)).getCommands()
        for i in range(1, len(comp_rows)):
            is_yes = comp_rows[i][5] == "YES"
            if is_yes:
                comp_style_cmds.append(("TEXTCOLOR", (5, i), (5, i), colors.HexColor("#c0392b")))
                comp_style_cmds.append(("FONTNAME",  (5, i), (5, i), "Helvetica-Bold"))
            else:
                comp_style_cmds.append(("TEXTCOLOR", (5, i), (5, i), colors.HexColor("#27ae60")))

        comp_table.setStyle(TableStyle(comp_style_cmds))
        elements.append(comp_table)

        # ── Contradiction summary sub-table ──────────────────────────
        contra_pairs: dict[str, int] = {}
        congruent_count = 0
        for m in micros:
            if m.get("is_contradictory"):
                dom = (m.get("dominant_emotion_at_time") or "?").capitalize()
                det = (m.get("detected_emotion") or "?").capitalize()
                pair_key = f"{dom} -> {det}"
                contra_pairs[pair_key] = contra_pairs.get(pair_key, 0) + 1
            else:
                congruent_count += 1

        if contra_pairs:
            elements.append(Spacer(1, 6 * mm))
            elements.append(Paragraph(
                "Contradiction Summary",
                ParagraphStyle(
                    "SubHeading", parent=heading_style,
                    fontSize=12, spaceBefore=2 * mm,
                ),
            ))
            elements.append(Paragraph(
                "Frequency of emotion contradictions detected during the session. "
                "A higher count suggests more emotional suppression in that pattern.",
                body_style,
            ))
            elements.append(Spacer(1, 2 * mm))

            sum_header = ["Displayed Emotion -> Hidden Emotion", "Occurrences", "Interpretation"]
            sum_rows = [sum_header]

            interpretations = {
                "Happiness": "social masking",
                "Neutral":   "emotional suppression",
                "Sadness":   "concealed frustration",
                "Fear":      "hidden anxiety",
                "Anger":     "suppressed aggression",
                "Surprise":  "concealed reaction",
                "Disgust":   "hidden aversion",
            }

            for pair, count in sorted(contra_pairs.items(), key=lambda x: x[1], reverse=True):
                displayed = pair.split(" -> ")[0]
                interp = interpretations.get(displayed, "emotional incongruence")
                sum_rows.append([pair, str(count), f"Possible {interp}"])

            sum_rows.append(["Congruent (non-contradictory)", str(congruent_count), "Authentic expression"])

            sum_table = Table(sum_rows, colWidths=[65 * mm, 25 * mm, 80 * mm])
            sum_table.setStyle(_table_style(len(sum_rows)))
            elements.append(sum_table)

    # ── Subject Validation & Feedback ───────────────────────────────
    feedback = session_data.get("feedback")
    if feedback and feedback.get("overall_accuracy_rating") is not None:
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph("Subject Validation & Feedback", heading_style))
        elements.append(Paragraph(
            "This section presents post-session self-report feedback completed by the subject, "
            "allowing for validation of the automated analysis.",
            body_style,
        ))
        elements.append(Spacer(1, 3 * mm))

        overall_rating = feedback["overall_accuracy_rating"]
        self_reported  = feedback.get("self_reported_emotion") or "—"
        dom_emo        = (summary or {}).get("dominant_emotion") or "—"
        match_str      = "—"
        if self_reported != "—" and dom_emo != "—":
            match_str = "YES" if self_reported.lower() == dom_emo.lower() else "No"

        suppression = "Yes" if feedback.get("attempted_suppression") else "No"

        moment_vals  = feedback.get("moment_validations") or []
        total_vals   = len(moment_vals)
        correct_vals = sum(1 for mv in moment_vals if mv.get("verdict") == "correct")
        val_str = (
            f"{correct_vals} / {total_vals} correct ({correct_vals / total_vals * 100:.0f}%)"
            if total_vals > 0 else "No events validated"
        )

        feedback_rows = [
            ["Metric", "Value"],
            ["Overall Self-Reported Accuracy",     f"{overall_rating * 100:.0f}%"],
            ["Self-Reported Predominant Emotion",   self_reported.capitalize()],
            ["Displayed Dominant Emotion",          dom_emo.capitalize()],
            ["Emotion Match",                       match_str],
            ["Attempted Emotional Suppression",     suppression],
            ["Moment Validations",                  val_str],
        ]

        feedback_table = Table(feedback_rows, colWidths=[80 * mm, 80 * mm])
        feedback_style_cmds = _table_style(len(feedback_rows)).getCommands()

        for i in range(1, len(feedback_rows)):
            if feedback_rows[i][0] == "Emotion Match":
                if match_str == "YES":
                    feedback_style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#27ae60")))
                    feedback_style_cmds.append(("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"))
                elif match_str == "No":
                    feedback_style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#c0392b")))
                    feedback_style_cmds.append(("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"))

        feedback_table.setStyle(TableStyle(feedback_style_cmds))
        elements.append(feedback_table)

        if feedback.get("free_text_comments"):
            elements.append(Spacer(1, 4 * mm))
            elements.append(Paragraph("Subject Comments:",
                                       ParagraphStyle("CommentSub", parent=small_style,
                                                      fontName="Helvetica-Bold", fontSize=9)))
            elements.append(Paragraph(feedback["free_text_comments"], body_style))

    # ── Interview Analysis ───────────────────────────────────────────
    if analysis:
        elements.append(PageBreak())

        # Section 1: Candidate Profile (Radar Chart)
        elements.append(Paragraph("Candidate Profile", heading_style))
        dimensions = [
            "technical_mastery", "emotional_stability", "authenticity",
            "self_confidence", "communication",
        ]

        d = Drawing(300, 250)
        center_x, center_y = 150, 125
        radius = 80

        # Background pentagon + axes + labels
        bg_points = []
        for i in range(5):
            angle = math.pi / 2 + (2 * math.pi * i / 5)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            bg_points.extend([x, y])
            d.add(Line(center_x, center_y, x, y, strokeColor=colors.lightgrey))
            label = dimensions[i].replace("_", " ").title()
            d.add(String(
                center_x + (radius + 20) * math.cos(angle) - 30,
                center_y + (radius + 15) * math.sin(angle),
                label, fontSize=8, fillColor=BODY_TEXT
            ))

        d.add(Polygon(bg_points, strokeColor=colors.lightgrey, fillColor=None, strokeWidth=1))

        # Score pentagon
        score_points = []
        for i, dim in enumerate(dimensions):
            angle = math.pi / 2 + (2 * math.pi * i / 5)
            score = dim_scores.get(dim, 0)
            r = radius * (score / 100.0)
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            score_points.extend([x, y])
            # Score label on vertex
            d.add(String(
                center_x + (r + 5) * math.cos(angle) - 8,
                center_y + (r + 5) * math.sin(angle),
                f"{score:.0f}", fontSize=7,
                fillColor=BRAND_HIGHLIGHT, fontName="Helvetica-Bold"
            ))

        d.add(Polygon(score_points,
                      strokeColor=BRAND_HIGHLIGHT,
                      fillColor=colors.Color(0.91, 0.27, 0.38, alpha=0.3),
                      strokeWidth=2))

        overall = dim_scores.get("overall_score", 0)
        d.add(String(center_x - 45, 10,
                     f"Overall Score: {overall:.1f}/100",
                     fontSize=10, fillColor=BRAND_ACCENT, fontName="Helvetica-Bold"))

        elements.append(d)
        elements.append(Spacer(1, 6 * mm))

        # Section 2: Behavioral Patterns
        patterns = analysis.get("behavioral_patterns", [])
        if patterns:
            elements.append(Paragraph("Behavioral Patterns", heading_style))
            pat_header = ["Time", "Pattern", "Severity", "Description", "Context"]
            pat_rows   = [pat_header]
            for p in patterns:
                sev      = p.get("severity", "low").lower()
                time_str = f"{p.get('timestamp', 0):.1f}s"
                pat_type = p.get("type", "—").replace("_", " ").title()
                pat_rows.append([
                    time_str,
                    pat_type,
                    sev.capitalize(),
                    _truncate(p.get("description", ""), 45),
                    _truncate(p.get("context_question", "") or "—", 30)
                ])

            pat_table = Table(pat_rows, colWidths=[15*mm, 35*mm, 20*mm, 60*mm, 40*mm])
            pat_style_cmds = _table_style(len(pat_rows)).getCommands()
            for i in range(1, len(pat_rows)):
                s = pat_rows[i][2].lower()
                c = (colors.green if s == "low"
                     else colors.orange if s == "medium"
                     else colors.red)
                if pat_rows[i][1] in ["Stress Recovery", "Authentic Engagement"]:
                    c = colors.green
                pat_style_cmds.append(("TEXTCOLOR", (2, i), (2, i), c))
                pat_style_cmds.append(("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"))
            pat_table.setStyle(TableStyle(pat_style_cmds))
            elements.append(pat_table)
            elements.append(Spacer(1, 6 * mm))

        # Section 3: Red Flags
        red_flags = analysis.get("red_flags", [])
        if red_flags:
            elements.append(Paragraph("Red Flags", heading_style))
            for rf in red_flags:
                sev     = rf.get("severity", "low").lower()
                rf_type = rf.get("type", "").replace("_", " ").title()
                ev      = rf.get("evidence", "")
                color_hex = ("#c0392b" if sev == "high"
                             else "#d35400" if sev == "medium"
                             else "#f1c40f")
                text = f'<font color="{color_hex}">⬤</font> <b>{rf_type}</b> ({sev.capitalize()}): {ev}'
                elements.append(Paragraph(text, body_style))
            elements.append(Spacer(1, 6 * mm))

        # Section 4: Question-Emotion Correlation
        correlations = analysis.get("question_correlations", [])
        if correlations:
            elements.append(Paragraph("Question-Emotion Correlation", heading_style))
            corr_header = ["Time", "Question", "Pre-Emotion", "Post-Emotion",
                           "Nervous Δ", "Congruence Δ", "Insight"]
            corr_rows = [corr_header]
            for c in correlations:
                n_change = c.get("nervousness_change", 0)
                c_change = c.get("congruence_change",  0)
                corr_rows.append([
                    f"{c.get('question_timestamp', 0):.1f}s",
                    _truncate(c.get("question_text", ""), 30),
                    c.get("pre_emotion",  ""),
                    c.get("post_emotion", ""),
                    f"{n_change:+.2f}",
                    f"{c_change:+.1f}",
                    _truncate(c.get("insight", ""), 40)
                ])

            corr_table = Table(corr_rows,
                               colWidths=[15*mm, 35*mm, 20*mm, 20*mm, 20*mm, 20*mm, 40*mm])
            corr_style_cmds = _table_style(len(corr_rows)).getCommands()
            for i in range(1, len(corr_rows)):
                try:
                    n_val = float(corr_rows[i][4])
                    n_color = colors.red if n_val > 0 else colors.green
                    corr_style_cmds.append(("TEXTCOLOR", (4, i), (4, i), n_color))
                except ValueError:
                    pass
            corr_table.setStyle(TableStyle(corr_style_cmds))
            elements.append(corr_table)
            elements.append(Spacer(1, 6 * mm))

        # Section 5: Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            elements.append(Paragraph("Recommendations", heading_style))
            priority_map = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(
                key=lambda x: priority_map.get(x.get("priority", "low").lower(), 3)
            )
            for idx, rec in enumerate(recommendations, 1):
                prio = rec.get("priority", "low").lower()
                cat  = rec.get("category", "").title()
                txt  = rec.get("text", "")
                color_hex = ("#c0392b" if prio == "high"
                             else "#d35400" if prio == "medium"
                             else "#7f8c8d")
                bullet = f'<font color="{color_hex}">•</font>'
                if prio == "high":
                    content = f"{idx}. {bullet} <b>[{cat}]</b> <b>{txt}</b>"
                else:
                    content = f"{idx}. {bullet} <b>[{cat}]</b> {txt}"
                elements.append(Paragraph(content, body_style))
            elements.append(Spacer(1, 6 * mm))

        # Section 6: Noise Filter Stats
        noise_stats = analysis.get("noise_stats")
        if noise_stats:
            elements.append(Paragraph("Noise Filter Statistics", heading_style))
            n_rows = [
                ["Metric", "Value"],
                ["Total Frames Analyzed",
                 str(noise_stats.get("total_frames_analyzed", 0))],
                ["Speaking Time Ratio",
                 f"{noise_stats.get('speaking_time_ratio', 0) * 100:.1f}%"],
                ["Total Noise Events Filtered",
                 str(noise_stats.get("total_noise_events_filtered", 0))],
            ]
            for k, v in noise_stats.get("noise_type_breakdown", {}).items():
                n_rows.append([f"Filtered: {k.title()}", str(v)])

            n_table = Table(n_rows, colWidths=[80*mm, 80*mm])
            n_table.setStyle(_table_style(len(n_rows)))
            elements.append(n_table)

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
        ("TEXTCOLOR",  (0, 0), (-1, 0), HEADER_TEXT),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING",    (0, 0), (-1, 0), 6),

        # Body rows
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 1), (-1, -1), BODY_TEXT),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING",    (0, 1), (-1, -1), 4),

        # Grid
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LINEBELOW", (0, 0), (-1, 0),  1,   BRAND_DARK),

        # Alignment
        ("ALIGN",  (0, 0), (-1, -1), "LEFT"),
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

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

# Slug -> Spanish label for values that come from the backend as English
# identifiers (dimension keys, pattern/flag "type", recommendation
# "category") and would otherwise print as raw snake_case in the report.
DIMENSION_LABELS = {
    "technical_mastery": "Dominio Técnico",
    "emotional_stability": "Estabilidad Emocional",
    "authenticity": "Autenticidad",
    "self_confidence": "Autoconfianza",
    "communication": "Comunicación",
}
EMOTION_LABELS = {
    "happy": "Feliz", "sad": "Triste", "angry": "Enojado",
    "surprise": "Sorpresa", "disgust": "Disgusto", "fear": "Miedo",
    "neutral": "Neutral", "nervousness": "Nervioso", "confidence": "Confiado",
}
TYPE_LABELS = {
    "rapid_confusion": "Confusión Rápida",
    "nervous_spiral": "Espiral de Nerviosismo",
    "social_masking": "Enmascaramiento Social",
    "stress_recovery": "Recuperación de Estrés",
    "emotional_flatline": "Aplanamiento Emocional",
    "confidence_decline": "Baja de Confianza",
    "authentic_engagement": "Compromiso Auténtico",
    "emotion_transition": "Transición Emocional",
    "technical": "Técnico",
    "behavioral": "Conductual",
    "communication": "Comunicación",
    "resilience": "Resiliencia",
    "confidence": "Confianza",
    "usability": "Usabilidad",
    "task_start": "Inicio de tarea",
    "task_end": "Fin de tarea",
    "error": "Error",
    "confusion": "Confusión",
    "key_question": "Pregunta clave",
}


def _type_label(slug: str) -> str:
    if not slug:
        return ""
    return TYPE_LABELS.get(slug, slug.replace("_", " "))


def _dimension_label(slug: str) -> str:
    return DIMENSION_LABELS.get(slug, slug.replace("_", " ").title())


def _emotion_label(slug: str) -> str:
    if not slug or slug == "—":
        return "—"
    return EMOTION_LABELS.get(slug.lower(), slug.capitalize())


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
            '<font color="#e94560">Reporte de Análisis de Entrevista</font>',
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
    candidate_name = session.get("candidate_name") or "No especificado"
    session_name   = session.get("name", "—")
    session_date   = _format_datetime(session.get("created_at"))
    session_dur    = _format_duration(session.get("duration_seconds"))
    input_type     = {"webcam": "Cámara Web", "video_upload": "Video Subido"}.get(
        session.get("input_type", "webcam"), session.get("input_type", "webcam")
    )

    def _info_val(text, size=10):
        return Paragraph(text, ParagraphStyle(
            "InfoV", parent=styles["BodyText"],
            fontSize=size, textColor=BODY_TEXT))

    col_w = page_width / 5
    info_data = [
        [
            Paragraph("CANDIDATO",   label_style),
            Paragraph("SESIÓN",      label_style),
            Paragraph("FECHA",       label_style),
            Paragraph("DURACIÓN",    label_style),
            Paragraph("ENTRADA",     label_style),
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
        _metric_cell_table(score_str, "Puntuación General", _score_color(overall_score)),
        _metric_cell_table(cong_str,  "Congruencia Prom.",  _score_color(avg_congruence)),
        _metric_cell_table(_emotion_label(dominant_emo), "Emoción Dominante"),
        _metric_cell_table(str(micro_count), "Microexpresiones"),
        _metric_cell_table(
            str(nerv_peaks), "Picos de Nerviosismo",
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
    elements.append(Paragraph("Resumen Ejecutivo", heading_style))

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
        top_label = _dimension_label(top_k)
        low_label = _dimension_label(low_k)

        exec_text = (
            f"Este reporte presenta el análisis automatizado de comportamiento y emociones del candidato "
            f"<b>{candidate_name}</b> durante una sesión de entrevista de {session_dur} realizada el "
            f"{session_date}. "
        )
        if overall_score is not None:
            level = ("sólido" if overall_score >= 70
                     else "moderado" if overall_score >= 40
                     else "bajo")
            exec_text += (
                f"El candidato obtuvo una puntuación general de entrevista de "
                f"<b>{overall_score:.0f}/100</b>, lo que indica un desempeño general <b>{level}</b>. "
            )
        exec_text += (
            f"La dimensión con mayor puntuación fue <b>{top_label}</b> ({top_v:.0f}/100), "
            f"mientras que el área que más requiere seguimiento fue <b>{low_label}</b> ({low_v:.0f}/100). "
        )
        if avg_congruence is not None:
            cong_level = ("alta" if avg_congruence >= 70
                          else "moderada" if avg_congruence >= 40
                          else "baja")
            exec_text += (
                f"La congruencia emocional promedió <b>{avg_congruence:.0f}/100</b> ({cong_level}), "
                f"lo que sugiere una autopresentación {'auténtica y consistente' if avg_congruence >= 70 else 'potencialmente inconsistente'} "
                f"a lo largo de la entrevista. "
            )
        micros_list = session_data.get("micro_expressions", [])
        if micro_count:
            contradictory_count = sum(1 for m in micros_list if m.get("is_contradictory"))
            if contradictory_count:
                exec_text += (
                    f"Se detectó un total de <b>{micro_count}</b> microexpresión(es), "
                    f"incluyendo <b>{contradictory_count}</b> evento(s) contradictorio(s) que sugieren enmascaramiento emocional. "
                )
            else:
                exec_text += (
                    f"Se detectó un total de <b>{micro_count}</b> microexpresión(es), "
                    f"todas consistentes con las emociones mostradas. "
                )
        elements.append(Paragraph(exec_text, body_style))
    else:
        elements.append(Paragraph(
            f"Este reporte presenta el análisis automatizado de comportamiento del candidato "
            f"<b>{candidate_name}</b> durante una sesión de entrevista de {session_dur} "
            f"realizada el {session_date}.",
            body_style
        ))

    elements.append(Spacer(1, 4 * mm))

    # Key findings bullets
    findings_rows = []
    if positive_patterns:
        pnames = ", ".join(_type_label(p["type"]) for p in positive_patterns)
        findings_rows.append([
            Paragraph('<font color="#27ae60">✔</font>', body_style),
            Paragraph(f"<b>Fortalezas detectadas:</b> {pnames}.", body_style),
        ])
    if concern_patterns:
        cnames = ", ".join(_type_label(p["type"]) for p in concern_patterns[:3])
        findings_rows.append([
            Paragraph('<font color="#f39c12">⚠</font>', body_style),
            Paragraph(f"<b>Áreas de atención:</b> {cnames}.", body_style),
        ])
    if red_flags:
        findings_rows.append([
            Paragraph('<font color="#c0392b">✗</font>', body_style),
            Paragraph(
                f"<b>{len(red_flags)} señal(es) de alerta</b> detectada(s) — ver la sección de Señales de Alerta para más detalles.",
                body_style
            ),
        ])
    if recs:
        high_recs = [r for r in recs if r.get("priority") == "high"]
        if high_recs:
            findings_rows.append([
                Paragraph('<font color="#0f3460">→</font>', body_style),
                Paragraph(
                    f"<b>Recomendación principal:</b> {high_recs[0].get('text', '')}",
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
            "⚠  Este reporte es una herramienta analítica generada por EmotionLens. No es una "
            "evaluación definitiva de carácter, veracidad o idoneidad. Todos los resultados "
            "deben ser interpretados por un profesional calificado dentro de un marco ético apropiado.",
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
        elements.append(Paragraph("Distribución de Emociones", heading_style))

        dist = summary["emotion_distribution"]
        emo_header = ["Emoción", "Porcentaje"]
        emo_rows = [emo_header]
        for emotion, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            emo_rows.append([
                _emotion_label(emotion),
                f"{pct * 100:.1f}%",
            ])

        if summary.get("dominant_emotion"):
            emo_rows.append(["Emoción Dominante", _emotion_label(summary["dominant_emotion"])])

        emo_table = Table(emo_rows, colWidths=[80 * mm, 80 * mm])
        emo_table.setStyle(_table_style(len(emo_rows)))
        elements.append(emo_table)

    # ── Congruence Summary ───────────────────────────────────────────
    if summary and summary.get("average_congruence") is not None:
        elements.append(Paragraph("Congruencia / Confiabilidad", heading_style))

        cong_rows = [
            ["Métrica", "Puntuación"],
            ["Congruencia Promedio", f"{summary['average_congruence']:.1f} / 100"],
            ["Congruencia Mínima", f"{summary.get('min_congruence', 0):.1f}"],
            ["Congruencia Máxima", f"{summary.get('max_congruence', 0):.1f}"],
        ]

        if summary.get("average_nervousness") is not None:
            cong_rows.append(["Nerviosismo Prom.", f"{summary['average_nervousness']:.3f}"])
        if summary.get("average_confidence") is not None:
            cong_rows.append(["Confianza Prom.", f"{summary['average_confidence']:.3f}"])
        if summary.get("nervousness_peaks") is not None:
            cong_rows.append(["Picos de Nerviosismo", str(summary["nervousness_peaks"])])
        if summary.get("average_model_confidence") is not None:
            cong_rows.append(["Confianza del Modelo Prom.", f"{summary['average_model_confidence']:.3f}"])

        cong_table = Table(cong_rows, colWidths=[80 * mm, 80 * mm])
        cong_table.setStyle(_table_style(len(cong_rows)))
        elements.append(cong_table)

    # ── Heart Rate (rPPG) ────────────────────────────────────────────
    hr_stats = _heart_rate_stats(session_data.get("emotion_timeline", []) or [])
    if hr_stats:
        elements.append(Paragraph("Ritmo Cardíaco (rPPG)", heading_style))

        hr_rows = [
            ["Métrica", "Valor"],
            ["Promedio", f"{hr_stats['mean']:.0f} BPM"],
            ["Rango habitual", f"{hr_stats['p25']:.0f} – {hr_stats['p75']:.0f} BPM"],
            ["Mínimo / Máximo", f"{hr_stats['min']:.0f} / {hr_stats['max']:.0f} BPM"],
            ["Lecturas válidas", f"{hr_stats['count']} ({hr_stats['coverage'] * 100:.0f}% de la sesión)"],
            ["Calidad media de señal", f"{hr_stats['mean_confidence']:.2f}"],
        ]

        hr_table = Table(hr_rows, colWidths=[80 * mm, 80 * mm])
        hr_table.setStyle(_table_style(len(hr_rows)))
        elements.append(hr_table)

        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph(
            "Estimado por fotopletismografía remota (rPPG) a partir del color de "
            "la piel de la frente. El <b>rango habitual</b> es el intervalo "
            "intercuartílico: descarta los extremos, que suelen venir de una "
            "sola ventana ruidosa. Solo se registran lecturas cuando la señal "
            "es utilizable, por eso la cobertura puede ser parcial. "
            "<b>No es un dispositivo médico</b>: sirve para analizar activación "
            "fisiológica, no para diagnóstico clínico.",
            small_style,
        ))
    elif session_data.get("emotion_timeline"):
        elements.append(Paragraph("Ritmo Cardíaco (rPPG)", heading_style))
        elements.append(Paragraph(
            "No se registraron lecturas de pulso en esta sesión. El estimador "
            "solo guarda un valor cuando la señal es utilizable; una sesión sin "
            "lecturas indica que el pulso no pudo recuperarse del video "
            "(iluminación, movimiento o calidad de cámara insuficientes).",
            body_style,
        ))

    # ── Micro-Expression Log ─────────────────────────────────────────
    micros = session_data.get("micro_expressions", [])
    if micros:
        elements.append(Paragraph("Registro de Microexpresiones", heading_style))
        elements.append(Paragraph(
            f"Total: {len(micros)} eventos detectados",
            body_style,
        ))
        elements.append(Spacer(1, 3 * mm))

        micro_header = ["Tiempo (s)", "Emoción", "Duración (ms)", "Contradictoria", "Descripción"]
        micro_rows = [micro_header]
        for m in micros:
            micro_rows.append([
                f"{m.get('timestamp', 0):.1f}",
                _emotion_label(m.get("detected_emotion", "—")),
                f"{m.get('duration_ms', 0):.0f}",
                "Sí" if m.get("is_contradictory") else "No",
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
        elements.append(Paragraph("Notas del Entrevistador", heading_style))

        note_header = ["Tiempo (s)", "Etiqueta", "Contenido", "Emoción", "Congruencia"]
        note_rows = [note_header]
        for n in notes:
            note_rows.append([
                f"{n.get('timestamp', 0):.1f}",
                _type_label(n.get("tag")) or "—",
                _truncate(n.get("content", ""), 45),
                _emotion_label(n.get("emotion_at_time")) if n.get("emotion_at_time") else "—",
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
        elements.append(Paragraph("Momentos Clave", heading_style))

        km_header = ["Tiempo (s)", "Tipo", "Detalle"]
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
            "Análisis Comparativo: Emociones vs Microexpresiones",
            heading_style,
        ))
        elements.append(Paragraph(
            "Esta tabla correlaciona cada microexpresión detectada con la "
            "emoción mostrada en ese momento, revelando posibles "
            "incongruencias emocionales durante la sesión.",
            body_style,
        ))
        elements.append(Spacer(1, 3 * mm))

        comp_header = [
            "Tiempo", "Emoción\nMostrada", "Microexpresión\nDetectada",
            "AUs Involucradas", "Duración", "Contradicción", "Interpretación",
        ]
        comp_rows = [comp_header]

        for m in micros:
            ts = m.get("timestamp", 0)
            mins = int(ts // 60)
            secs = int(ts % 60)
            time_str = f"{mins:02d}:{secs:02d}"

            dominant = _emotion_label(m.get("dominant_emotion_at_time") or "neutral")
            detected = _emotion_label(m.get("detected_emotion") or "unknown")
            aus = ", ".join(m.get("action_units_involved", []))
            duration = f"{m.get('duration_ms', 0):.0f}ms"
            is_contra = m.get("is_contradictory", False)
            contradiction_str = "SÍ" if is_contra else "No"

            if is_contra:
                interpretation = (
                    f"El sujeto mostró {dominant.lower()} pero reveló brevemente "
                    f"{detected.lower()} — posible supresión"
                )
            else:
                interpretation = (
                    f"Congruente — {detected.lower()} refuerza "
                    f"la emoción mostrada de {dominant.lower()}"
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
            is_yes = comp_rows[i][5] == "SÍ"
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
                # Keep the raw slug as the dict key (not the display label) so
                # the lookup below matches reliably regardless of capitalization.
                dom_slug = (m.get("dominant_emotion_at_time") or "?").lower()
                det_slug = (m.get("detected_emotion") or "?").lower()
                pair_key = f"{dom_slug}|{det_slug}"
                contra_pairs[pair_key] = contra_pairs.get(pair_key, 0) + 1
            else:
                congruent_count += 1

        if contra_pairs:
            elements.append(Spacer(1, 6 * mm))
            elements.append(Paragraph(
                "Resumen de Contradicciones",
                ParagraphStyle(
                    "SubHeading", parent=heading_style,
                    fontSize=12, spaceBefore=2 * mm,
                ),
            ))
            elements.append(Paragraph(
                "Frecuencia de contradicciones emocionales detectadas durante la sesión. "
                "Un conteo más alto sugiere mayor supresión emocional en ese patrón.",
                body_style,
            ))
            elements.append(Spacer(1, 2 * mm))

            sum_header = ["Emoción Mostrada -> Emoción Oculta", "Ocurrencias", "Interpretación"]
            sum_rows = [sum_header]

            interpretations = {
                "happy":       "enmascaramiento social",
                "neutral":     "supresión emocional",
                "sad":         "frustración oculta",
                "fear":        "ansiedad oculta",
                "angry":       "agresión suprimida",
                "surprise":    "reacción oculta",
                "disgust":     "aversión oculta",
                "nervousness": "nerviosismo oculto",
                "confidence":  "inseguridad oculta",
            }

            for pair, count in sorted(contra_pairs.items(), key=lambda x: x[1], reverse=True):
                dom_slug, det_slug = pair.split("|")
                pair_label = f"{_emotion_label(dom_slug)} -> {_emotion_label(det_slug)}"
                interp = interpretations.get(dom_slug, "incongruencia emocional")
                sum_rows.append([pair_label, str(count), f"Posible {interp}"])

            sum_rows.append(["Congruente (no contradictorio)", str(congruent_count), "Expresión auténtica"])

            sum_table = Table(sum_rows, colWidths=[65 * mm, 25 * mm, 80 * mm])
            sum_table.setStyle(_table_style(len(sum_rows)))
            elements.append(sum_table)

    # ── Subject Validation & Feedback ───────────────────────────────
    feedback = session_data.get("feedback")
    if feedback and feedback.get("overall_accuracy_rating") is not None:
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph("Validación y Retroalimentación del Sujeto", heading_style))
        elements.append(Paragraph(
            "Esta sección presenta la retroalimentación post-sesión completada por el sujeto, "
            "permitiendo validar el análisis automatizado.",
            body_style,
        ))
        elements.append(Spacer(1, 3 * mm))

        overall_rating = feedback["overall_accuracy_rating"]
        self_reported  = feedback.get("self_reported_emotion") or "—"
        dom_emo        = (summary or {}).get("dominant_emotion") or "—"
        match_str      = "—"
        if self_reported != "—" and dom_emo != "—":
            match_str = "SÍ" if self_reported.lower() == dom_emo.lower() else "No"

        suppression = "Sí" if feedback.get("attempted_suppression") else "No"

        # Agreement counts only moments the reviewer actually ruled on --
        # skipped and "unsure" moments are excluded from the rate rather
        # than silently counted as disagreement.
        val_metrics = feedback.get("validation_metrics") or {}
        confirmed = val_metrics.get("confirmed", 0)
        rejected = val_metrics.get("rejected", 0)
        agreement = val_metrics.get("agreement_rate")
        val_str = (
            f"{confirmed} / {confirmed + rejected} confirmadas ({agreement * 100:.0f}%)"
            if agreement is not None else "Ningún evento validado"
        )

        feedback_rows = [
            ["Métrica", "Valor"],
            ["Precisión Autorreportada General",    f"{overall_rating * 100:.0f}%"],
            ["Emoción Predominante Autorreportada", _emotion_label(self_reported)],
            ["Emoción Dominante Mostrada",          _emotion_label(dom_emo)],
            ["Coincidencia de Emoción",              match_str],
            ["Intento de Supresión Emocional",      suppression],
            ["Validaciones de Momentos",             val_str],
        ]

        feedback_table = Table(feedback_rows, colWidths=[80 * mm, 80 * mm])
        feedback_style_cmds = _table_style(len(feedback_rows)).getCommands()

        for i in range(1, len(feedback_rows)):
            if feedback_rows[i][0] == "Coincidencia de Emoción":
                if match_str == "SÍ":
                    feedback_style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#27ae60")))
                    feedback_style_cmds.append(("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"))
                elif match_str == "No":
                    feedback_style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#c0392b")))
                    feedback_style_cmds.append(("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"))

        feedback_table.setStyle(TableStyle(feedback_style_cmds))
        elements.append(feedback_table)

        # ── Measured agreement breakdown ─────────────────────────────
        # Whether the engine's own relevance score predicts which flags a
        # human accepts. If high-relevance flags aren't confirmed more
        # often than medium ones, the scoring isn't earning its keep.
        bands = val_metrics.get("by_relevance_band") or {}
        band_rows = [["Banda de Relevancia", "Confirmadas", "Rechazadas", "Inciertas", "Concordancia"]]
        for band_key, band_label in (("high", "Alta (80-100)"),
                                     ("medium", "Media (60-79)"),
                                     ("low", "Baja (<60)"),
                                     ("unknown", "Sin registrar")):
            band = bands.get(band_key)
            if not band:
                continue
            rate = band.get("agreement_rate")
            band_rows.append([
                band_label,
                str(band.get("confirmed", 0)),
                str(band.get("rejected", 0)),
                str(band.get("unsure", 0)),
                f"{rate * 100:.0f}%" if rate is not None else "—",
            ])

        if len(band_rows) > 1:
            elements.append(Spacer(1, 4 * mm))
            elements.append(Paragraph(
                "Concordancia medida entre las detecciones del sistema y la revisión humana. "
                "Los momentos que el revisor omitió o marcó como inciertos se excluyen de la tasa.",
                small_style,
            ))
            elements.append(Spacer(1, 2 * mm))
            band_table = Table(band_rows, colWidths=[45 * mm, 28 * mm, 28 * mm, 25 * mm, 34 * mm])
            band_table.setStyle(_table_style(len(band_rows)))
            elements.append(band_table)

        unanswered = val_metrics.get("unanswered", 0)
        if unanswered:
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph(
                f"{unanswered} momento(s) marcado(s) no fueron revisados.", small_style))

        if feedback.get("free_text_comments"):
            elements.append(Spacer(1, 4 * mm))
            elements.append(Paragraph("Comentarios del Sujeto:",
                                       ParagraphStyle("CommentSub", parent=small_style,
                                                      fontName="Helvetica-Bold", fontSize=9)))
            elements.append(Paragraph(feedback["free_text_comments"], body_style))

    # ── Interview Analysis ───────────────────────────────────────────
    if analysis:
        elements.append(PageBreak())

        # Section 1: Candidate Profile (Radar Chart)
        elements.append(Paragraph("Perfil del Candidato", heading_style))
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
            label = _dimension_label(dimensions[i])
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
                     f"Puntuación General: {overall:.1f}/100",
                     fontSize=10, fillColor=BRAND_ACCENT, fontName="Helvetica-Bold"))

        elements.append(d)
        elements.append(Spacer(1, 6 * mm))

        # Section 2: Behavioral Patterns
        patterns = analysis.get("behavioral_patterns", [])
        if patterns:
            elements.append(Paragraph("Patrones de Comportamiento", heading_style))
            pat_header = ["Tiempo", "Patrón", "Severidad", "Descripción", "Contexto"]
            pat_rows   = [pat_header]
            for p in patterns:
                sev      = p.get("severity", "low").lower()
                time_str = _format_video_time(p.get("timestamp", 0))
                pat_type = _type_label(p.get("type", "—"))
                sev_label = {"low": "Baja", "medium": "Media", "high": "Alta"}.get(sev, sev.capitalize())
                pat_rows.append([
                    time_str,
                    pat_type,
                    sev_label,
                    _truncate(p.get("description", ""), 45),
                    _truncate(p.get("context_question", "") or "—", 30)
                ])

            pat_table = Table(pat_rows, colWidths=[15*mm, 35*mm, 20*mm, 60*mm, 40*mm])
            pat_style_cmds = _table_style(len(pat_rows)).getCommands()
            for i in range(1, len(pat_rows)):
                s = pat_rows[i][2].lower()
                c = (colors.green if s == "baja"
                     else colors.orange if s == "media"
                     else colors.red)
                if pat_rows[i][1] in ["Recuperación de Estrés", "Compromiso Auténtico"]:
                    c = colors.green
                pat_style_cmds.append(("TEXTCOLOR", (2, i), (2, i), c))
                pat_style_cmds.append(("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"))
            pat_table.setStyle(TableStyle(pat_style_cmds))
            elements.append(pat_table)
            elements.append(Spacer(1, 6 * mm))

        # Section 3: Red Flags
        red_flags = analysis.get("red_flags", [])
        if red_flags:
            elements.append(Paragraph("Señales de Alerta", heading_style))
            for rf in red_flags:
                sev     = rf.get("severity", "low").lower()
                rf_type = _type_label(rf.get("type", ""))
                ev      = rf.get("evidence", "")
                sev_label = {"low": "Baja", "medium": "Media", "high": "Alta"}.get(sev, sev.capitalize())
                color_hex = ("#c0392b" if sev == "high"
                             else "#d35400" if sev == "medium"
                             else "#f1c40f")
                text = f'<font color="{color_hex}">⬤</font> <b>{rf_type}</b> ({sev_label}): {ev}'
                elements.append(Paragraph(text, body_style))
            elements.append(Spacer(1, 6 * mm))

        # Section 4: Question-Emotion Correlation
        correlations = analysis.get("question_correlations", [])
        if correlations:
            elements.append(Paragraph("Correlación Pregunta-Emoción", heading_style))
            corr_header = ["Tiempo", "Pregunta", "Emoción Previa", "Emoción Posterior",
                           "Δ Nervios.", "Δ Congr.", "Perspectiva"]
            corr_rows = [corr_header]
            for c in correlations:
                n_change = c.get("nervousness_change", 0)
                c_change = c.get("congruence_change",  0)
                corr_rows.append([
                    f"{c.get('question_timestamp', 0):.1f}s",
                    _truncate(c.get("question_text", ""), 30),
                    _emotion_label(c.get("pre_emotion",  "")),
                    _emotion_label(c.get("post_emotion", "")),
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

        # Section 4b: Task Friction — ranked worst-first, since the top row
        # is the actionable output of a usability review.
        tasks = analysis.get("task_analysis") or []
        if tasks:
            elements.append(Paragraph("Fricción por Tarea", heading_style))
            elements.append(Paragraph(
                "La fricción combina lo que mostró el rostro (nerviosismo promedio y pico) "
                "con lo que marcó el moderador (errores, confusión) durante cada tarea.",
                small_style,
            ))
            elements.append(Spacer(1, 2 * mm))

            task_rows = [["Tarea", "Ventana", "Duración", "Errores", "Confusión", "Fricción"]]
            for t in sorted(tasks, key=lambda x: x.get("friction_score", 0), reverse=True):
                window = f"{t.get('start_video_time', '')}-{t.get('end_video_time', '')}"
                if not t.get("completed", True):
                    window += " *"
                task_rows.append([
                    _truncate(str(t.get("task_name", "")), 32),
                    window,
                    f"{t.get('duration_seconds', 0):.0f}s",
                    str(t.get("error_count", 0)),
                    str(t.get("confusion_count", 0)),
                    f"{t.get('friction_score', 0):.0f}/100",
                ])

            task_table = Table(task_rows,
                               colWidths=[52*mm, 28*mm, 20*mm, 18*mm, 24*mm, 24*mm])
            task_cmds = _table_style(len(task_rows)).getCommands()
            for i in range(1, len(task_rows)):
                try:
                    score = float(task_rows[i][5].split("/")[0])
                except ValueError:
                    continue
                color = (colors.red if score >= 60
                         else colors.orange if score >= 35
                         else colors.green)
                task_cmds.append(("TEXTCOLOR", (5, i), (5, i), color))
                task_cmds.append(("FONTNAME",  (5, i), (5, i), "Helvetica-Bold"))
            task_table.setStyle(TableStyle(task_cmds))
            elements.append(task_table)

            if any(not t.get("completed", True) for t in tasks):
                elements.append(Spacer(1, 2 * mm))
                elements.append(Paragraph(
                    "* Tarea nunca marcada como completada.", small_style))
            elements.append(Spacer(1, 6 * mm))

        # Section 4c: Marked friction points
        friction_events = [e for e in (analysis.get("event_timeline") or [])
                           if e.get("is_friction_point")]
        if friction_events:
            elements.append(Paragraph("Puntos de Fricción Marcados", heading_style))
            ev_rows = [["Tiempo", "Marcador", "Nota", "Emoción", "Nervios.", "Congr."]]
            for e in friction_events:
                emo_before = _emotion_label(e.get("emotion_before", ""))
                emo_after = _emotion_label(e.get("emotion_after", ""))
                ev_rows.append([
                    e.get("video_time", ""),
                    _truncate(str(e.get("label", "")), 18),
                    _truncate(str(e.get("content", "")), 30),
                    f"{emo_before}->{emo_after}",
                    f"{e.get('nervousness_change', 0):+.2f}",
                    f"{e.get('congruence_change', 0):+.1f}",
                ])

            ev_table = Table(ev_rows,
                             colWidths=[16*mm, 30*mm, 48*mm, 36*mm, 20*mm, 20*mm])
            ev_table.setStyle(_table_style(len(ev_rows)))
            elements.append(ev_table)
            elements.append(Spacer(1, 6 * mm))

        # Section 4d: Emotion changes, each with a reading offered as a
        # hypothesis. Observation and interpretation are printed on separate
        # lines on purpose — a reader must be able to take the first without
        # the second, since a facial configuration does not establish what the
        # person felt (Barrett et al., 2019).
        transitions = [e for e in (analysis.get("event_timeline") or [])
                       if e.get("type") == "emotion_transition" and e.get("suggestion")]
        if transitions:
            elements.append(Paragraph("Cambios Observados y Lecturas Posibles", heading_style))
            elements.append(Paragraph(
                "Cada cambio se acompana de una lectura posible, no de una "
                "conclusion. Son hipotesis para orientar la revision de la "
                "grabacion: el sistema no determina que sintio la persona.",
                small_style))
            elements.append(Spacer(1, 3 * mm))

            for e in transitions[:12]:
                elements.append(Paragraph(
                    f"<b>{e.get('description', '')}</b>", body_style))
                elements.append(Paragraph(
                    f"<i>Lectura posible:</i> {e.get('suggestion', '')}", body_style))
                elements.append(Paragraph(
                    f"Base: {e.get('basis', '')}", small_style))
                elements.append(Spacer(1, 3 * mm))
            elements.append(Spacer(1, 4 * mm))

        # Section 5: Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            elements.append(Paragraph("Recomendaciones", heading_style))
            priority_map = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(
                key=lambda x: priority_map.get(x.get("priority", "low").lower(), 3)
            )
            for idx, rec in enumerate(recommendations, 1):
                prio = rec.get("priority", "low").lower()
                cat  = _type_label(rec.get("category", ""))
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
            elements.append(Paragraph("Estadísticas del Filtro de Ruido", heading_style))
            n_rows = [
                ["Métrica", "Valor"],
                ["Total de Cuadros Analizados",
                 str(noise_stats.get("total_frames_analyzed", 0))],
                ["Proporción de Tiempo Hablando",
                 f"{noise_stats.get('speaking_time_ratio', 0) * 100:.1f}%"],
                ["Total de Eventos de Ruido Filtrados",
                 str(noise_stats.get("total_noise_events_filtered", 0))],
            ]
            noise_type_labels = {"speaking": "Hablando", "yawning": "Bostezando", "tic": "Tic"}
            for k, v in noise_stats.get("noise_type_breakdown", {}).items():
                n_rows.append([f"Filtrado: {noise_type_labels.get(k, k.title())}", str(v)])

            n_table = Table(n_rows, colWidths=[80*mm, 80*mm])
            n_table.setStyle(_table_style(len(n_rows)))
            elements.append(n_table)

    # ── Footer ───────────────────────────────────────────────────────
    elements.append(Spacer(1, 10 * mm))
    elements.append(Paragraph(
        f"Generado por EmotionLens el {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "Este reporte es una herramienta analítica y no constituye una evaluación "
        "definitiva de carácter o veracidad.",
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
    Generate a CSV report with the emotion timeline, one row per second.

    Columns: tiempo, segundo, emocion, confianza, congruencia, bpm,
    bpm_confianza, frames, action_units

    Args:
        session_data: Complete session data dict.
        output_path: Absolute path for the output CSV file.

    Returns:
        The output_path on success.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(_csv_rows(session_data))

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
    writer.writerow(CSV_HEADER)
    writer.writerows(_csv_rows(session_data))
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


def _format_video_time(seconds) -> str:
    """Format elapsed session time as MM:SS for human review."""
    seconds = max(0, int(round(seconds or 0)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ═══════════════════════════════════════════════════════════════════════
# HEART RATE (rPPG)
# ═══════════════════════════════════════════════════════════════════════

def _heart_rate_series(timeline: list) -> list[tuple[float, float, float]]:
    """
    (timestamp, bpm, confidence) for every record that carries a reading.

    The live pipeline stores the estimate alongside the Action Units and only
    while the estimator reports a usable signal, so a session can legitimately
    have none: a gap here means the pulse was not recoverable from the video,
    not that the field is missing.
    """
    series = []
    for record in timeline:
        aus = record.get("action_units") or {}
        bpm = aus.get("hr_bpm")
        if bpm is None:
            continue
        try:
            bpm = float(bpm)
        except (TypeError, ValueError):
            continue
        if bpm <= 0:
            continue
        try:
            confidence = float(aus.get("hr_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        series.append((float(record.get("timestamp") or 0.0), bpm, confidence))
    return series


def _heart_rate_stats(timeline: list) -> Optional[dict]:
    """Summary of the session's pulse, or None if nothing was measured."""
    series = _heart_rate_series(timeline)
    if not series:
        return None

    values = sorted(bpm for _, bpm, _ in series)
    confidences = [c for _, _, c in series]

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        position = p * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {
        "count": len(values),
        "coverage": len(series) / len(timeline) if timeline else 0.0,
        "mean": sum(values) / len(values),
        "min": values[0],
        "max": values[-1],
        # Typical band rather than the extremes: a single bad window sets the
        # min and max, so on its own the full range overstates the spread.
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "mean_confidence": sum(confidences) / len(confidences),
    }


# ═══════════════════════════════════════════════════════════════════════
# CSV ROW BUILDING
# ═══════════════════════════════════════════════════════════════════════

CSV_HEADER = [
    "tiempo", "segundo", "emocion", "confianza",
    "congruencia", "bpm", "bpm_confianza", "frames", "action_units",
]


def _dominant(labels: list[str]) -> str:
    """The most frequent label, ties broken by first appearance."""
    if not labels:
        return ""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best = max(counts.values())
    for label in labels:
        if counts[label] == best:
            return label
    return labels[0]


def _mean(values: list) -> Optional[float]:
    numbers = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(numbers) / len(numbers) if numbers else None


def _csv_rows(session_data: dict) -> list[list]:
    """
    One row per second of session, not one per frame.

    The pipeline writes a record per processed frame (~20 a second), so the
    raw timeline repeats every second twenty times over and a few minutes of
    interview become thousands of near-identical rows — unreadable as a
    timeline, which is what this export is for. Each second is collapsed to
    its dominant emotion and the mean of the numeric columns; `frames` keeps
    how many readings went into it, so a second thinned by dropped frames is
    still visible as such.
    """
    timeline = session_data.get("emotion_timeline", []) or []

    buckets: dict[int, list[dict]] = {}
    for record in timeline:
        try:
            second = int(float(record.get("timestamp") or 0.0))
        except (TypeError, ValueError):
            continue
        buckets.setdefault(second, []).append(record)

    rows = []
    for second in sorted(buckets):
        records = buckets[second]
        aus_list = [r.get("action_units") or {} for r in records]

        bpm_values = [a.get("hr_bpm") for a in aus_list if a.get("hr_bpm")]
        bpm = _mean(bpm_values)
        bpm_confidence = _mean([a.get("hr_confidence") for a in aus_list if a.get("hr_bpm")])

        confidence = _mean([r.get("confidence") for r in records])
        congruence = _mean([r.get("congruence_score") for r in records])

        # Mean of each Action Unit across the second, keeping the existing
        # "name=value" shape so anything already parsing this column still can.
        au_keys = sorted({k for a in aus_list for k in a if not k.startswith("hr_")})
        action_units_str = "; ".join(
            f"{key}={value:.4f}"
            for key in au_keys
            if (value := _mean([a.get(key) for a in aus_list])) is not None
        )

        rows.append([
            _format_video_time(second),
            second,
            _dominant([r.get("emotion", "") for r in records]),
            f"{confidence:.3f}" if confidence is not None else "",
            f"{congruence:.1f}" if congruence is not None else "",
            f"{bpm:.1f}" if bpm is not None else "",
            f"{bpm_confidence:.3f}" if bpm_confidence is not None else "",
            len(records),
            action_units_str,
        ])

    return rows

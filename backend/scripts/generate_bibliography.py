"""
Genera docs/bibliografia-cientifica.pdf a partir de backend/app/references.py.

El PDF se construye desde el registro de procedencia, no a mano, para que no
pueda desincronizarse del código.

Uso:
    python -m backend.scripts.generate_bibliography
"""

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.references import (  # noqa: E402
    BIBLIOGRAPHY, CONSTANTS, Tier, constants_by_tier, coverage,
)

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT = BASE_DIR / "docs" / "bibliografia-cientifica.pdf"

INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f3460")
GREEN = colors.HexColor("#1b7a4b")
AMBER = colors.HexColor("#b8860b")
RED = colors.HexColor("#9b2226")
GREY = colors.HexColor("#5a5a6e")

TIER_COLOR = {Tier.VALIDATED: GREEN, Tier.CALIBRATED: AMBER, Tier.HEURISTIC: RED}

# Helvetica (WinAnsi) carece de glifos como "Č" en los nombres de autores, que
# saldrían como cajas. Se busca una fuente TTF Unicode del sistema.
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def register_fonts() -> tuple[str, str]:
    """Registra una fuente Unicode si hay alguna disponible; si no, Helvetica."""
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("Doc", regular))
            pdfmetrics.registerFont(TTFont("Doc-Bold", bold))
            pdfmetrics.registerFontFamily("Doc", normal="Doc", bold="Doc-Bold",
                                          italic="Doc", boldItalic="Doc-Bold")
            return "Doc", "Doc-Bold"
    return "Helvetica", "Helvetica-Bold"


def styles(font: str, font_bold: str):
    s = getSampleStyleSheet()
    for style in s.byName.values():
        if getattr(style, "fontName", "").startswith("Helvetica"):
            style.fontName = font_bold if "Bold" in style.fontName else font
    return {
        "title": ParagraphStyle("T", parent=s["Title"], fontSize=22, leading=27,
                                textColor=INK, spaceAfter=6),
        "subtitle": ParagraphStyle("S", parent=s["Normal"], fontSize=11.5, leading=16,
                                   textColor=GREY, alignment=TA_JUSTIFY, spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=s["Heading1"], fontSize=15, leading=19,
                             textColor=ACCENT, spaceBefore=18, spaceAfter=9),
        "h2": ParagraphStyle("H2", parent=s["Heading2"], fontSize=12, leading=15,
                             textColor=INK, spaceBefore=13, spaceAfter=6),
        "body": ParagraphStyle("B", parent=s["Normal"], fontSize=9.8, leading=14,
                               alignment=TA_JUSTIFY, spaceAfter=7),
        "ref": ParagraphStyle("R", parent=s["Normal"], fontSize=9.3, leading=13,
                              leftIndent=16, firstLineIndent=-16, spaceAfter=9),
        "cell": ParagraphStyle("C", parent=s["Normal"], fontSize=8.2, leading=11),
        "cellb": ParagraphStyle("CB", parent=s["Normal"], fontSize=8.2, leading=11,
                                fontName=font_bold),
    }


def tier_badge(tier: Tier, st) -> Paragraph:
    return Paragraph(
        f'<font color="{TIER_COLOR[tier].hexval()}"><b>{tier.value}</b></font>', st["cell"]
    )


def build_story(st) -> list:
    cov = coverage()
    total = len(CONSTANTS)
    story = []

    # ── Portada ──────────────────────────────────────────────────────
    story.append(Paragraph("EmotionLens", st["title"]))
    story.append(Paragraph("Bibliografía científica y procedencia de constantes", st["h2"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "Este documento registra el respaldo científico de cada constante numérica que "
        "gobierna el comportamiento de EmotionLens, y — con igual detalle — identifica "
        "las que no lo tienen. Se genera automáticamente a partir de "
        "<i>backend/app/references.py</i>, de modo que no puede quedar desactualizado "
        "respecto al código.",
        st["subtitle"]))

    story.append(Paragraph("Cómo leer este documento", st["h1"]))
    story.append(Paragraph(
        "Cada constante está clasificada en uno de tres niveles de evidencia. La "
        "distinción central es entre <b>usar un concepto respaldado</b> y <b>usar un valor "
        "numérico publicado</b>: el proyecto hace mucho de lo primero y poco de lo segundo, "
        "y este documento existe para hacer esa diferencia explícita y auditable.",
        st["body"]))

    tier_rows = [[
        Paragraph("Nivel", st["cellb"]),
        Paragraph("Significado", st["cellb"]),
        Paragraph("N.º", st["cellb"]),
    ]]
    tier_desc = {
        Tier.VALIDATED: "El valor proviene de una fuente publicada y revisada por pares. "
                        "Se puede citar el número exacto.",
        Tier.CALIBRATED: "El valor fue ajustado empíricamente sobre un dataset público con "
                         "codificación FACS certificada, reportando métricas de error.",
        Tier.HEURISTIC: "No existe evidencia publicada para este valor. Fue elegido por "
                        "prueba y error y debe presentarse como tal ante el usuario.",
    }
    for t in Tier:
        tier_rows.append([
            tier_badge(t, st),
            Paragraph(tier_desc[t], st["cell"]),
            Paragraph(str(cov[t.value]), st["cell"]),
        ])

    tbl = Table(tier_rows, colWidths=[2.6 * cm, 11.9 * cm, 1.4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    n_heur = cov[Tier.HEURISTIC.value]
    story.append(Paragraph(
        f"<b>Estado actual: {n_heur} de {total} grupos de constantes carecen de respaldo "
        f"publicado.</b> Entre ellas están las que definen la semántica visible del "
        f"sistema — los pesos del score de congruencia y los cortes verde/amarillo/rojo — "
        f"es decir, precisamente aquellas cuya salida interpreta un entrevistador humano.",
        st["body"]))

    # ── Bibliografía ─────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Bibliografía", st["h1"]))

    groups = {
        "Base anatómica: FACS": ["ekman1978facs", "ekman2002facs"],
        "Micro-expresiones: temporalidad": ["yan2013duration", "matsumoto2018microintent"],
        "Datasets con codificación FACS certificada": [
            "yan2014casme2", "davison2018samm", "mavadati2013disfa"],
        "Parpadeo": ["bentivoglio1997blink", "soukupova2016ear", "plos2025blink"],
        "Ritmo cardíaco remoto (rPPG)": ["dehaan2013chrom", "wu2012evm"],
        "Límites de la inferencia facial": [
            "barrett2019reconsidered", "bond2006accuracy", "nrc2003polygraph"],
    }

    for group, keys in groups.items():
        story.append(Paragraph(group, st["h2"]))
        for k in keys:
            r = BIBLIOGRAPHY[k]
            ident = (f'DOI: {r.identifier}' if not r.identifier.startswith("http")
                     else r.identifier)
            story.append(Paragraph(
                f"{r.authors} ({r.year}). <i>{r.title}</i>. {r.venue}. "
                f'<font color="{GREY.hexval()}">{ident}</font>',
                st["ref"]))

    # ── Inventario de constantes ─────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Inventario de constantes", st["h1"]))
    story.append(Paragraph(
        "Cada fila vincula una constante del código con su nivel de evidencia y sus "
        "fuentes. La columna de notas indica qué respalda exactamente la cita — o, "
        "cuando no hay respaldo, por qué.",
        st["body"]))

    rows = [[
        Paragraph("Constante / Ubicación", st["cellb"]),
        Paragraph("Valor", st["cellb"]),
        Paragraph("Nivel", st["cellb"]),
        Paragraph("Notas y fuentes", st["cellb"]),
    ]]
    for c in CONSTANTS:
        cites = ""
        if c.refs:
            cites = "<br/><i>" + "; ".join(
                f"{BIBLIOGRAPHY[k].authors.split(',')[0]} et al. ({BIBLIOGRAPHY[k].year})"
                for k in c.refs
            ) + "</i>"
        rows.append([
            Paragraph(f"<b>{c.name}</b><br/>"
                      f'<font size="7" color="{GREY.hexval()}">{c.location}</font>',
                      st["cell"]),
            Paragraph(c.current_value, st["cell"]),
            tier_badge(c.tier, st),
            Paragraph(c.note + cites, st["cell"]),
        ])

    inv = Table(rows, colWidths=[3.9 * cm, 2.3 * cm, 2.4 * cm, 7.3 * cm], repeatRows=1)
    inv.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fc")]),
    ]))
    story.append(inv)

    # ── Advertencia ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Qué no puede afirmar este sistema", st["h1"]))
    story.append(Paragraph(
        "Tres resultados de la literatura acotan directamente lo que EmotionLens puede "
        "sostener sobre una persona, y conviene tenerlos presentes al leer cualquier "
        "reporte que genere:",
        st["body"]))
    for k, claim in [
        ("bond2006accuracy",
         "Sobre 206 estudios y 24.483 jueces, la precisión humana para distinguir "
         "verdad de mentira es del 54%, apenas por encima del azar. Los meta-análisis "
         "no encuentran que las señales faciales se relacionen de forma fiable con la "
         "veracidad."),
        ("barrett2019reconsidered",
         "La correspondencia entre configuraciones faciales y estados emocionales "
         "internos es altamente variable entre personas, situaciones y culturas. "
         "Inferir la emoción de una persona a partir de su movimiento facial no está "
         "respaldado por la evidencia disponible."),
        ("nrc2003polygraph",
         "La base científica del Comparison Question Technique — comparar el "
         "comportamiento contra un basal propio para inferir ocultamiento — fue "
         "evaluada como débil, con tasa de error desconocida. El componente de "
         "'desviación del baseline' de EmotionLens replica esa misma premisa."),
    ]:
        r = BIBLIOGRAPHY[k]
        story.append(Paragraph(f"<b>{r.authors.split(',')[0]} et al. ({r.year})</b>", st["h2"]))
        story.append(Paragraph(claim, st["body"]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "En consecuencia, el score de congruencia debe entenderse como un <b>índice "
        "descriptivo de variabilidad expresiva</b>, no como una medida de honestidad, "
        "confiabilidad ni idoneidad laboral. Ninguna decisión de contratación debería "
        "apoyarse en él.",
        st["body"]))

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles(*register_fonts())
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=LETTER,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="EmotionLens — Bibliografía científica",
        author="EmotionLens",
    )
    doc.build(build_story(st))
    print(f"[OK] PDF generado: {OUTPUT}")
    for tier, n in coverage().items():
        print(f"     {tier}: {n}")


if __name__ == "__main__":
    main()

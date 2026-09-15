"""
Genera docs/plan-evidencia-cientifica.pdf a partir de docs/plan-evidencia-cientifica.md.

Conversor Markdown -> PDF minimalista (encabezados, tablas, listas, negrita/
cursiva, código). No es un motor Markdown completo: cubre exactamente la
sintaxis usada en el plan, nada más.

Uso:
    python -m backend.scripts.generate_plan_pdf
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE = BASE_DIR / "docs" / "plan-evidencia-cientifica.md"
OUTPUT = BASE_DIR / "docs" / "plan-evidencia-cientifica.pdf"

INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f3460")
GREY = colors.HexColor("#5a5a6e")
CODE_BG = colors.HexColor("#f2f2f7")

_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
]


def register_fonts() -> dict[str, str]:
    """Registra una fuente Unicode. Si no hay itálica real en disco, usa la
    regular como sustituta: importa más la cobertura de glifos (acentos,
    Č, emoji básicos) que el ángulo del trazo."""
    for regular, bold, italic, mono in _FONT_CANDIDATES:
        if not all(Path(p).exists() for p in (regular, bold, mono)):
            continue
        pdfmetrics.registerFont(TTFont("Doc", regular))
        pdfmetrics.registerFont(TTFont("Doc-Bold", bold))
        italic_path = italic if Path(italic).exists() else regular
        pdfmetrics.registerFont(TTFont("Doc-Italic", italic_path))
        pdfmetrics.registerFont(TTFont("Doc-Mono", mono))
        pdfmetrics.registerFontFamily(
            "Doc", normal="Doc", bold="Doc-Bold",
            italic="Doc-Italic", boldItalic="Doc-Bold")
        pdfmetrics.registerFontFamily(
            "Doc-Mono", normal="Doc-Mono", bold="Doc-Mono",
            italic="Doc-Mono", boldItalic="Doc-Mono")
        return {"reg": "Doc", "bold": "Doc-Bold", "italic": "Doc-Italic",
                "mono": "Doc-Mono"}
    return {"reg": "Helvetica", "bold": "Helvetica-Bold",
            "italic": "Helvetica-Oblique", "mono": "Courier"}


def styles(font: dict[str, str]):
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=s["Title"], fontName=font["bold"],
                                fontSize=20, leading=25, textColor=INK,
                                spaceAfter=10),
        "h1": ParagraphStyle("H1", parent=s["Heading1"], fontName=font["bold"],
                             fontSize=14.5, leading=18, textColor=ACCENT,
                             spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=s["Heading2"], fontName=font["bold"],
                             fontSize=11.5, leading=15, textColor=INK,
                             spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("H3", parent=s["Heading3"], fontName=font["bold"],
                             fontSize=10.3, leading=13, textColor=INK,
                             spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("B", parent=s["Normal"], fontName=font["reg"],
                               fontSize=9.6, leading=13.6, spaceAfter=7),
        "li": ParagraphStyle("LI", parent=s["Normal"], fontName=font["reg"],
                             fontSize=9.6, leading=13.6, spaceAfter=3),
        "cell": ParagraphStyle("C", parent=s["Normal"], fontName=font["reg"],
                               fontSize=8.4, leading=11.6),
        "cellb": ParagraphStyle("CB", parent=s["Normal"], fontName=font["bold"],
                                fontSize=8.4, leading=11.6),
        "code": ParagraphStyle("Code", parent=s["Normal"], fontName=font["mono"],
                               fontSize=8.2, leading=11, textColor=INK,
                               backColor=CODE_BG, borderPadding=6,
                               spaceAfter=8, spaceBefore=2),
    }


def inline(text: str) -> str:
    """Traduce negrita/cursiva/código Markdown a etiquetas ReportLab."""
    text = text.strip()
    # DejaVuSans no tiene glifo para "✅" (U+2705); "✓" (U+2713) sí.
    text = text.replace("✅", "✓")
    text = re.sub(r"`([^`]+)`",
                  rf'<font color="{ACCENT.hexval()}"><i>\1</i></font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    return text


def parse_table(lines: list[str], i: int):
    """Parsea una tabla Markdown a partir de la línea i. Devuelve (rows, next_i)."""
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    rows = [header]
    i += 2  # saltar encabezado y separador ---|---
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return rows, i


def build_story(st, font: dict[str, str]) -> list:
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.split("\n")
    story = []
    i = 0
    first_h1 = True

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 0.15 * cm))
            story.append(HRFlowable(width="100%", color=colors.HexColor("#c8cdd8"),
                                    thickness=0.6))
            story.append(Spacer(1, 0.25 * cm))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), st["title"]))
            i += 1
            continue

        if stripped.startswith("## "):
            if not first_h1:
                pass
            first_h1 = False
            story.append(Paragraph(inline(stripped[3:]), st["h1"]))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), st["h2"]))
            i += 1
            continue

        if stripped.startswith("#### "):
            story.append(Paragraph(inline(stripped[5:]), st["h3"]))
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # saltar el cierre ```
            story.append(Preformatted("\n".join(code_lines), st["code"]))
            continue

        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            n_cols = len(rows[0])
            col_width = (17.0 * cm) / n_cols
            data = [[Paragraph(inline(c), st["cellb"] if r == 0 else st["cell"])
                     for c in row] for r, row in enumerate(rows)]
            tbl = Table(data, colWidths=[col_width] * n_cols)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cdd8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f8f9fc")]),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))
            continue

        if re.match(r"^(-|\d+\.)\s", stripped):
            items = []
            ordered = bool(re.match(r"^\d+\.\s", stripped))
            while i < len(lines) and re.match(r"^(-|\d+\.)\s", lines[i].strip()):
                item_lines = [re.sub(r"^(-|\d+\.)\s", "", lines[i].strip())]
                i += 1
                # líneas de continuación: texto envuelto del mismo ítem
                while i < len(lines) and lines[i].strip() and not re.match(
                    r"^(-|\d+\.)\s", lines[i].strip()
                ):
                    item_lines.append(lines[i].strip())
                    i += 1
                items.append(ListItem(Paragraph(inline(" ".join(item_lines)), st["li"]),
                                      leftIndent=6))
            list_kwargs = dict(
                bulletType="1" if ordered else "bullet",
                leftIndent=14, bulletFontName=font["reg"],
                bulletFontSize=8.6, spaceBefore=2, spaceAfter=8)
            if ordered:
                list_kwargs["start"] = "1"
            story.append(ListFlowable(items, **list_kwargs))
            continue

        # párrafo normal (puede continuar en líneas siguientes hasta línea en blanco)
        para_lines = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            lines[i].strip().startswith(("#", "|", "-", "```")) or
            re.match(r"^\d+\.\s", lines[i].strip()) or lines[i].strip() == "---"
        ):
            para_lines.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(para_lines)), st["body"]))

    return story


def main() -> None:
    if not SOURCE.exists():
        print(f"[ERROR] No se encontró {SOURCE}", file=sys.stderr)
        sys.exit(1)

    font = register_fonts()
    st = styles(font)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=LETTER,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title="EmotionLens — Plan de evidencia científica",
        author="EmotionLens",
    )
    doc.build(build_story(st, font))
    print(f"[OK] PDF generado: {OUTPUT}")


if __name__ == "__main__":
    main()

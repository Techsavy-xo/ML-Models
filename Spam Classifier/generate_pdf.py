"""
PDF Generator for SentinelSpam AI Interview Guide
Uses ReportLab to convert Markdown-structured study guide into a clean PDF.
"""

import os
import re
import html
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(PROJECT_ROOT, "PROJECT_INTERVIEW_GUIDE.md")
PDF_PATH = os.path.join(PROJECT_ROOT, "PROJECT_INTERVIEW_GUIDE.pdf")

def create_pdf():
    if not os.path.exists(MD_PATH):
        print(f"Error: {MD_PATH} not found.")
        return

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("SentinelSpam AI — Master Technical Study & Interview Guide", title_style))
    story.append(Paragraph("<b>End-to-End Spam Email Classifier & Threat Intelligence Platform</b>", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=14))

    lines = md_text.split("\n")
    in_code_block = False
    code_lines = []

    for line in lines:
        stripped = line.strip()

        # Handle Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                # Close code block
                code_content = "<br/>".join([html.escape(c) for c in code_lines])
                story.append(Paragraph(code_content, code_style))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            continue

        # Skip main markdown title as we already placed it
        if stripped.startswith("# SentinelSpam AI"):
            continue

        # Headings
        if stripped.startswith("# "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(html.escape(stripped[2:]), h1_style))
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#93c5fd"), spaceAfter=6))
            continue
        elif stripped.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(html.escape(stripped[3:]), h2_style))
            continue
        elif stripped.startswith("### "):
            story.append(Paragraph(f"<b>{html.escape(stripped[4:])}</b>", h2_style))
            continue

        # Table rows or delimiters (basic table rendering)
        if stripped.startswith("|"):
            # Skip horizontal separator
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            formatted_cells = [Paragraph(f"<b>{html.escape(c)}</b>", body_style) if "Algorithm" in line or "Technology" in line else Paragraph(html.escape(c), body_style) for c in cells]
            t = Table([formatted_cells], colWidths=[520 / len(cells)] * len(cells))
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(t)
            continue

        # Bullets
        if stripped.startswith("- ") or stripped.startswith("* "):
            content = html.escape(stripped[2:])
            # bold tag translation
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
            story.append(Paragraph(f"&bull; {content}", bullet_style))
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", stripped):
            num, rest = stripped.split(".", 1)
            content = html.escape(rest.strip())
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
            story.append(Paragraph(f"<b>{num}.</b> {content}", bullet_style))
            continue

        # Standard Paragraph
        content = html.escape(stripped)
        content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
        content = re.sub(r"\*(.+?)\*", r"<i>\1</i>", content)
        story.append(Paragraph(content, body_style))

    doc.build(story)
    print(f"PDF generated successfully at: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)")

if __name__ == "__main__":
    create_pdf()

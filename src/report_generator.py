
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_RIGHT, TA_LEFT
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REPORT_DIR = "outputs/reports"

FONT_PATH = r"C:\Windows\Fonts\tahoma.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\tahomabd.ttf"


# --------------------------------------------------
# REGISTER FONTS
# --------------------------------------------------

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(
        f"Tahoma font not found: {FONT_PATH}"
    )

pdfmetrics.registerFont(
    TTFont("Tahoma", FONT_PATH)
)

if os.path.exists(FONT_BOLD_PATH):
    pdfmetrics.registerFont(
        TTFont("Tahoma-Bold", FONT_BOLD_PATH)
    )
else:
    pdfmetrics.registerFont(
        TTFont("Tahoma-Bold", FONT_PATH)
    )


# --------------------------------------------------
# LANGUAGE DETECTION
# --------------------------------------------------

def detect_language(text):
    """Detect Persian or English based on text content."""

    if re.search(r"[\u0600-\u06FF]", str(text)):
        return "fa"

    return "en"


def resolve_language(language, question):
    """Resolve report language."""

    if language == "auto":
        return detect_language(question)

    if language in ("fa", "en"):
        return language

    return detect_language(question)


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def contains_persian(text):
    return bool(
        re.search(r"[\u0600-\u06FF]", str(text))
    )


def prepare_text(text):
    """Prepare Persian text for correct PDF rendering."""

    text = str(text)

    # Remove unsupported control characters
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
        "",
        text
    )

    if contains_persian(text):
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    return text


def escape_text(text):
    """Escape text for ReportLab Paragraph."""

    return escape(prepare_text(text))


# --------------------------------------------------
# TRANSLATIONS
# --------------------------------------------------

TRANSLATIONS = {
    "fa": {
        "title": "گزارش تحلیل داده",
        "date": "تاریخ تهیه گزارش",
        "question": "سؤال تحلیل",
        "answer": "نتیجه تحلیل",
        "charts": "نمودارهای تحلیل",
        "footer": "دستیار تحلیل داده",
    },

    "en": {
        "title": "Data Analysis Report",
        "date": "Report Generation Date",
        "question": "Analysis Question",
        "answer": "Analysis Results",
        "charts": "Analysis Charts",
        "footer": "AI Data Analyst",
    }
}


# --------------------------------------------------
# PDF REPORT GENERATOR
# --------------------------------------------------

def create_pdf_report(
    question,
    answer,
    chart_paths=None,
    language="auto"
):
    """
    Create a bilingual PDF analysis report.

    Parameters:
        question: Original user question.
        answer: Generated analysis answer.
        chart_paths: Optional list of chart file paths.
        language: "fa", "en", or "auto".

    Returns:
        Generated PDF file path.
    """

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Resolve language
    language = resolve_language(
        language,
        question
    )

    labels = TRANSLATIONS[language]

    is_persian = language == "fa"

    # --------------------------------------------------
    # FILE NAME
    # --------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"analysis_report_{language}_{timestamp}.pdf"
    )

    filepath = os.path.join(
        REPORT_DIR,
        filename
    )

    # --------------------------------------------------
    # DOCUMENT SETTINGS
    # --------------------------------------------------

    document = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    alignment = TA_RIGHT if is_persian else TA_LEFT

    # --------------------------------------------------
    # TEXT STYLES
    # --------------------------------------------------

    title_style = ParagraphStyle(
        "ReportTitle",
        fontName="Tahoma-Bold",
        fontSize=18,
        leading=25,
        textColor=colors.HexColor("#1F4E79"),
        alignment=alignment,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        fontName="Tahoma-Bold",
        fontSize=12,
        leading=20,
        textColor=colors.HexColor("#2F75B5"),
        alignment=alignment,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style_settings = {
        "name": "ReportBody",
        "fontName": "Tahoma",
        "fontSize": 9,
        "leading": 16,
        "alignment": alignment,
        "spaceAfter": 7
    }

    if is_persian:
        body_style_settings["wordWrap"] = "RTL"

    body_style = ParagraphStyle(
        **body_style_settings
    )

    # --------------------------------------------------
    # STORY
    # --------------------------------------------------

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            escape_text(labels["title"]),
            title_style
        )
    )

    report_date = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    date_text = (
        f"{labels['date']}: {report_date}"
    )

    story.append(
        Paragraph(
            escape_text(date_text),
            body_style
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------
    # QUESTION
    # --------------------------------------------------

    story.append(
        Paragraph(
            escape_text(labels["question"]),
            heading_style
        )
    )

    story.append(
        Paragraph(
            escape_text(question),
            body_style
        )
    )

    # --------------------------------------------------
    # ANSWER
    # --------------------------------------------------

    story.append(
        Paragraph(
            escape_text(labels["answer"]),
            heading_style
        )
    )

    for line in str(answer).splitlines():

        line = line.strip()

        if not line:
            story.append(
                Spacer(1, 5)
            )
            continue

        story.append(
            Paragraph(
                escape_text(line),
                body_style
            )
        )

    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    if chart_paths:

        valid_charts = [
            path
            for path in chart_paths
            if path and os.path.exists(path)
        ]

        if valid_charts:

            story.append(
                Paragraph(
                    escape_text(labels["charts"]),
                    heading_style
                )
            )

            for chart_path in valid_charts:

                chart = Image(chart_path)

                chart._restrictSize(
                    165 * mm,
                    110 * mm
                )

                story.append(
                    Spacer(1, 8)
                )

                story.append(chart)

                story.append(
                    Spacer(1, 12)
                )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    def add_page_number(canvas, doc):

        canvas.saveState()

        canvas.setFont(
            "Tahoma",
            8
        )

        footer_text = prepare_text(
            labels["footer"]
        )

        page_number = str(doc.page)

        if is_persian:

            canvas.drawRightString(
                190 * mm,
                12 * mm,
                footer_text
            )

            canvas.drawString(
                20 * mm,
                12 * mm,
                page_number
            )

        else:

            canvas.drawString(
                20 * mm,
                12 * mm,
                footer_text
            )

            canvas.drawRightString(
                190 * mm,
                12 * mm,
                page_number
            )

        canvas.restoreState()

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    return filepath

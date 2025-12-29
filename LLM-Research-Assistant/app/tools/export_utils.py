"""
export_utils.py
Utility to convert plain text into a valid PDF using ReportLab Platypus.
This method produces clean PDFs that Streamlit can download correctly.
"""

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


def convert_to_pdf(text: str) -> bytes:
    """
    Convert a text string into a proper PDF byte stream.

    Returns:
        PDF binary bytes suitable for Streamlit download_button.
    """
    buffer = BytesIO()

    # Document template
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title="Research Assistant Output"
    )

    styles = getSampleStyleSheet()
    story = []

    # ReportLab Paragraph requires HTML-safe text
    safe_text = text.replace("\n", "<br/>")

    story.append(Paragraph(safe_text, styles["Normal"]))

    # Build PDF
    pdf.build(story)

    # Retrieve PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

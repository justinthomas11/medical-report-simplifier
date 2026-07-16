"""
Module 12 — Output Delivery

Handles PDF report generation and export.
Uses fpdf2 to write clean structured PDFs.
"""

import io
from fpdf import FPDF
from modules.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class SimplifiedReportPDF(FPDF):
    """Custom FPDF class for formatting the simplified medical report PDF."""
    
    def header(self):
        # Logo/Title
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(26, 54, 93) # Deep Blue
        self.cell(0, 10, 'Medical Report Simplifier', border=False, new_x="LMARGIN", new_y="NEXT", align='L')
        self.set_draw_color(26, 54, 93)
        self.set_line_width(0.5)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        
        # Disclaimer
        disclaimer_text = "Disclaimer: This document is an automated simplification for informational purposes only. It is not medical advice."
        self.cell(0, 5, disclaimer_text, border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        
        # Page number
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', border=False, new_x="LMARGIN", new_y="NEXT", align='C')


def _break_long_words(pdf: FPDF, text: str) -> str:
    """
    Ensure no individual word/token is wider than the printable area.

    fpdf2 with default WORD wrap raises 'Not enough horizontal space'
    when a single token (no spaces) is wider than the available width.
    This helper splits such tokens by inserting spaces so the layout
    engine can always make progress.
    """
    printable_w = pdf.w - pdf.l_margin - pdf.r_margin  # mm
    words = text.split(" ")
    safe_words = []
    for word in words:
        # Measure word width at current font size
        word_w = pdf.get_string_width(word)
        if word_w <= printable_w or len(word) <= 1:
            safe_words.append(word)
            continue
        # Binary-split the word into chunks that fit
        chunk = ""
        for ch in word:
            if pdf.get_string_width(chunk + ch) > printable_w:
                safe_words.append(chunk)
                chunk = ch
            else:
                chunk += ch
        if chunk:
            safe_words.append(chunk)
    return " ".join(safe_words)


@log_execution_time
def generate_pdf_report(simplified_markdown: str) -> bytes:
    """
    Generate a formatted PDF document from markdown text.
    
    Args:
        simplified_markdown: The markdown content of the simplified report.
        
    Returns:
        Bytes of the generated PDF file.
    """
    pdf = SimplifiedReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)
    pdf.set_text_color(45, 55, 72)  # Charcoal dark grey
    
    # Replace characters that fpdf2 cannot render natively in helvetica
    text_to_render = simplified_markdown
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2022": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "*": "",  # Remove markdown bold/italic asterisks for simple PDF rendering
    }
    for orig, repl in replacements.items():
        text_to_render = text_to_render.replace(orig, repl)
        
    # Strip emojis and other characters unsupported by default Latin-1 Helvetica
    text_to_render = text_to_render.encode('latin-1', 'ignore').decode('latin-1')
        
    lines = text_to_render.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue
            
        # Formatting headings simple logic
        if line.startswith("###"):
            pdf.ln(3)
            pdf.set_font("helvetica", "B", 12)
            pdf.set_text_color(43, 108, 176) # Secondary blue
            heading = _break_long_words(pdf, line.replace("###", "").strip())
            pdf.multi_cell(0, 6, heading)
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(45, 55, 72)
        elif line.startswith("##"):
            pdf.ln(4)
            pdf.set_font("helvetica", "B", 14)
            pdf.set_text_color(26, 54, 93) # Primary blue
            heading = _break_long_words(pdf, line.replace("##", "").strip())
            pdf.multi_cell(0, 8, heading)
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(45, 55, 72)
        elif line.startswith("#"):
            pdf.ln(5)
            pdf.set_font("helvetica", "B", 16)
            pdf.set_text_color(26, 54, 93)
            heading = _break_long_words(pdf, line.replace("#", "").strip())
            pdf.multi_cell(0, 10, heading)
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(45, 55, 72)
        else:
            # Regular text
            # Handle horizontal rules/separator lines elegantly
            if len(line) >= 3 and set(line) <= {'-', '_', '='}:
                pdf.ln(2)
                pdf.set_x(pdf.l_margin)  # Reset x before drawing line
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(2)
            else:
                safe_line = _break_long_words(pdf, line)
                pdf.multi_cell(0, 5, safe_line)
            
    # Return PDF as bytes
    pdf_output = pdf.output()
    if isinstance(pdf_output, bytes):
        return pdf_output
    else:
        # Older versions or fallback
        return bytes(pdf_output)

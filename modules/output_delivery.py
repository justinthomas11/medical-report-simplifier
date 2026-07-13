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


@log_execution_time
def generate_pdf_report(simplified_markdown: str) -> bytes:
    """
    Generate a formatted PDF document from markdown text.
    
    Args:
        simplified_markdown: The markdown content of the simplified report.
        
    Returns:
        Bytes of the generated PDF file.
    """
    try:
        pdf = SimplifiedReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)
        pdf.set_text_color(45, 55, 72)  # Charcoal dark grey
        
        # Replace characters that fpdf2 cannot render natively in helvetica
        text_to_render = simplified_markdown
        replacements = {
            "—": "-",
            "–": "-",
            "•": "-",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
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
                pdf.multi_cell(0, 6, line.replace("###", "").strip())
                pdf.set_font("helvetica", "", 10)
                pdf.set_text_color(45, 55, 72)
            elif line.startswith("##"):
                pdf.ln(4)
                pdf.set_font("helvetica", "B", 14)
                pdf.set_text_color(26, 54, 93) # Primary blue
                pdf.multi_cell(0, 8, line.replace("##", "").strip())
                pdf.set_font("helvetica", "", 10)
                pdf.set_text_color(45, 55, 72)
            elif line.startswith("#"):
                pdf.ln(5)
                pdf.set_font("helvetica", "B", 16)
                pdf.set_text_color(26, 54, 93)
                pdf.multi_cell(0, 10, line.replace("#", "").strip())
                pdf.set_font("helvetica", "", 10)
                pdf.set_text_color(45, 55, 72)
            else:
                # Regular text
                pdf.multi_cell(0, 5, line)
                
        # Return PDF as bytes
        pdf_output = pdf.output()
        if isinstance(pdf_output, bytes):
            return pdf_output
        else:
            # Older versions or fallback
            return bytes(pdf_output)
            
    except Exception as e:
        logger.error(f"Failed to generate PDF: {str(e)}")
        # Return a simple error PDF bytes or raise
        raise

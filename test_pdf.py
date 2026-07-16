import os
import sys

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.output_delivery import generate_pdf_report

def main():
    sample_markdown = """# Understanding Your Blood Test Results

Patient Name: John Doe
Date of Test: July 10, 2026

You recently had a Complete Blood Count (CBC). Below is a simplified guide.

---

## Your Blood Test Results

Your blood contains several components:
* **White Blood Cell (WBC):** 12.4 x10^3/uL (High) - Indicates infection or inflammation.
* **Red Blood Cell (RBC):** 3.8 x10^6/uL (Low) - Indicates anemia.
* **Hemoglobin (Hgb):** 11.2 g/dL (Low) - Low oxygen carrying capacity.

---

### Medical Glossary
- Diseases: anemia
- Tests: CBC
"""
    print("Generating PDF...")
    pdf_bytes = generate_pdf_report(sample_markdown)
    base_dir = r"C:\Users\Admin\Desktop\My stuff\Christ\3rd Sem\NLP\MedicalReportSimplifier"
    output_filepath = os.path.join(base_dir, "test_output.pdf")
    with open(output_filepath, "wb") as f:
        f.write(pdf_bytes)
    print("Done! PDF written to test_output.pdf")

if __name__ == "__main__":
    main()

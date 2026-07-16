import os
import sys
from dotenv import load_dotenv

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.input_handler import validate_file, get_file_type
from modules.text_extractor import extract_text
from modules.text_preprocessor import preprocess
from modules.info_extractor import extract_medical_entities
from modules.retrieval_engine import compare_retrieval_models, get_best_context
from modules.simplification_engine import simplify_report

def main():
    load_dotenv()
    print("API Key loaded:", bool(os.getenv("GEMINI_API_KEY")))
    
    # Read sample file (absolute path)
    base_dir = r"C:\Users\Admin\Desktop\My stuff\Christ\3rd Sem\NLP\MedicalReportSimplifier"
    filepath = os.path.join(base_dir, "tests", "sample_reports", "sample_cbc.txt")
    class MockUploadedFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self.path = path
        def read(self):
            with open(self.path, 'rb') as f:
                return f.read()
    
    uploaded_file = MockUploadedFile(filepath)
    file_type = get_file_type(uploaded_file)
    extracted_text = extract_text(uploaded_file, file_type)
    
    print("\n--- Extracted Text ---")
    print(extracted_text)
    
    preprocessed = preprocess(extracted_text)
    entities = extract_medical_entities(extracted_text)
    
    query_terms = " ".join(entities.diseases + entities.tests_procedures)
    if not query_terms.strip():
        query_terms = " ".join(extracted_text.split()[:20])
        
    retrieval_comparison = compare_retrieval_models(query_terms)
    context, _ = get_best_context(retrieval_comparison, "SBERT")
    
    print("\n--- Simplification ---")
    raw_simplification = simplify_report(extracted_text, context, entities.to_dict())
    print(raw_simplification)

if __name__ == "__main__":
    main()

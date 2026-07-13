"""
Global configuration and constants for the Medical Report Simplifier.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "documents"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
LOGS_DIR = BASE_DIR / "data" / "logs"
FEEDBACK_DIR = BASE_DIR / "data" / "feedback"

# Create directories if they don't exist
for dir_path in [VECTOR_STORE_DIR, LOGS_DIR, FEEDBACK_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# API KEYS
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============================================================
# FILE VALIDATION
# ============================================================
ALLOWED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "text/plain": ".txt",
}
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
MAX_FILE_SIZE_MB = 10
MAX_PAGE_COUNT = 50

# ============================================================
# TEXT PREPROCESSING
# ============================================================
SPACY_MODEL = "en_core_web_sm"
MIN_EXTRACTED_TEXT_LENGTH = 50  # Fallback to OCR if PDF yields less

# ============================================================
# MEDICAL NER
# ============================================================
SCISPACY_MODEL = "en_ner_bc5cdr_md"

# ============================================================
# EMBEDDING MODELS
# ============================================================
MINILM_MODEL = "all-MiniLM-L6-v2"
SBERT_MODEL = "all-mpnet-base-v2"

# ============================================================
# DOCUMENT INDEXING
# ============================================================
CHUNK_SIZE = 512  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K_RESULTS = 5  # number of retrieval results

# ============================================================
# LLM CONFIGURATION
# ============================================================
GEMINI_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# ============================================================
# SIMPLIFICATION PROMPT
# ============================================================
SYSTEM_PROMPT = """You are a medical report simplifier. Your job is to take complex 
medical reports and explain them in simple, easy-to-understand language that a 
non-medical person can understand.

Guidelines:
- Use simple, everyday language (8th grade reading level)
- Explain all medical terms in parentheses when first used
- Organize information clearly with sections
- Highlight any critical or abnormal findings
- Include what the findings might mean for the patient
- Add a note that this is not medical advice and they should consult their doctor
- Be accurate — do not make up or assume information not in the report"""

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

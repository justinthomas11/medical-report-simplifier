# Medical Report Simplifier

A full-stack NLP system designed to translate complex medical reports (PDF/Image/Text) into plain, easy-to-understand language. It leverages a Multi-Model Retrieval-Augmented Generation (RAG) pipeline to compare retrieval metrics across TF-IDF, MiniLM, and SBERT models, extracting medical entities (NER) and querying a Gemini 1.5 Flash LLM for high-quality, patient-friendly simplifications.

---

## Key Features

1. **Robust Input Handling & Validation**: File type, size, and page count validation.
2. **Hybrid Text Extraction**: PyPDF2/pdfplumber for text PDFs, with automatic fallback to EasyOCR for scanned reports or image uploads.
3. **Advanced Medical NER**: Entity extraction using SciSpacy (`en_ner_bc5cdr_md`) supplemented by structured rule-based pattern matching for medical measurements, values, symptoms, tests, and body parts.
4. **Curated Reference Corpus**: Built-in library of ~25 detailed reference documents covering common lab panels (CBC, kidney, liver, lipids, thyroid, urinalysis), diseases, medications, and abbreviations.
5. **Multi-Model Retrieval comparison**: Side-by-side performance evaluation of TF-IDF, MiniLM, and SBERT (using FAISS vector indexes) based on precision, recall, and search latency.
6. **LLM Simplification Engine**: Powered by Google Gemini 1.5 Flash (free tier) with optimized medical communication prompts.
7. **Streamlit Web Dashboard**: Intuitive, tabbed user interface for report uploads, entity visualization, retrieval model charts, and feedback compilation.

---

## Project Architecture

```
MedicalReportSimplifier/
├── app.py                          # Streamlit web interface dashboard
├── requirements.txt                # Project dependencies
├── .env                            # Gemini API key (gitignored)
├── README.md                       # Setup and project documentation
│
├── config/
│   └── settings.py                 # Centralized settings & configurations
│
├── modules/
│   ├── input_handler.py            # File validation & uploads (Modules 1 & 2)
│   ├── text_extractor.py           # pdfplumber / EasyOCR extraction (Module 3)
│   ├── text_preprocessor.py        # Tokenization & lemmatization (Module 4)
│   ├── info_extractor.py           # SciSpacy + rule-based Medical NER (Module 5)
│   ├── knowledge_base.py           # Medical corpus file management (Module 6)
│   ├── document_indexer.py         # smart chunking & FAISS index creation (Module 7)
│   ├── retrieval_engine.py         # Multi-model retrieval & comparison (Module 8)
│   ├── simplification_engine.py    # Gemini LLM translation (Module 9)
│   ├── evaluation.py               # Precision, recall, & latency metrics (Module 10)
│   ├── output_generator.py         # Plain language summaries formatting (Module 11)
│   ├── output_delivery.py          # PDF export & delivery (Module 12)
│   ├── feedback.py                 # Feedback collection system (Module 13)
│   └── logger.py                   # Central logging & tracking (Module 14)
│
├── knowledge_base/
│   └── documents/                  # Reference medical texts (by category)
│
└── data/
    ├── vector_store/               # Serialized FAISS indices & TF-IDF matrix
    ├── feedback/                   # User ratings & review logs
    └── logs/                       # Application run logs
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- Get a free **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Installation
Clone the repository, navigate into the directory, and install the dependencies:
```bash
# Clone the repository
git clone git@github.com:justinthomas11/medical-report-simplifier.git
cd MedicalReportSimplifier

# Install base dependencies
pip install -r requirements.txt

# Download SciSpacy model (Required for Medical NER)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### 3. API Key Setup
Create a `.env` file in the root directory (or update the existing template):
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 4. Build the Vector Indices
Before running the application for the first time, build the FAISS indexes from the knowledge base documents:
```bash
python -c "from modules.document_indexer import initialize_indices; initialize_indices()"
```

### 5. Run the Application
Start the Streamlit dashboard server:
```bash
streamlit run app.py
```
Open the provided URL (typically `http://localhost:8501`) in your browser.

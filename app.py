"""
Medical Report Simplifier — Streamlit Web Application Dashboard

This dashboard coordinates the pipeline:
1. File Upload & Validation
2. Text Extraction (PDF / OCR)
3. NLP Preprocessing (spaCy / NLTK)
4. Medical NER (SciSpacy / Rules)
5. Multi-Model Retrieval (TF-IDF vs MiniLM vs SBERT)
6. Gemini LLM Simplification
7. PDF Output Export & Feedback
"""

import time
import os
import pandas as pd
import streamlit as st

# Custom module imports
from config.settings import VECTOR_STORE_DIR, GEMINI_API_KEY
from modules.input_handler import validate_file, get_file_type
from modules.text_extractor import extract_text
from modules.text_preprocessor import preprocess, get_clean_text_for_llm
from modules.info_extractor import extract_medical_entities
from modules.document_indexer import initialize_indices
from modules.retrieval_engine import compare_retrieval_models, get_best_context
from modules.simplification_engine import simplify_report
from modules.output_generator import format_simplified_report
from modules.output_delivery import generate_pdf_report
from modules.evaluation import evaluate_retrieval
from modules.feedback import save_user_feedback, get_feedback_statistics

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Medical Report Simplifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    /* Gradient headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #4A5568;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    /* Section containers */
    .report-card {
        background-color: #F8FAFC;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #2B6CB0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    /* Metrics panel */
    .metric-box {
        background-color: #EDF2F7;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check if vector indices exist
def indices_built() -> bool:
    required_files = [
        "tfidf_vectorizer.pkl", "tfidf_matrix.pkl", "tfidf_chunks.pkl",
        "minilm.index", "minilm_chunks.pkl",
        "sbert.index", "sbert_chunks.pkl"
    ]
    return all((VECTOR_STORE_DIR / f).exists() for f in required_files)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=70)
    st.markdown("### Settings & Controls")
    
    # Check Gemini API Key Status
    if not GEMINI_API_KEY:
        st.error("🔑 Gemini API Key: Missing")
        st.info("Please set the GEMINI_API_KEY variable in your `.env` file to enable simplification.")
    else:
        st.success("🔑 Gemini API Key: Connected")
        
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Medical Report",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        help="Upload a PDF (scanned/text), Image, or Text file (< 10MB)"
    )
    
    # Model Selection (Preferred for output context)
    preferred_retrieval_model = st.selectbox(
        "Preferred RAG Retrieval Model",
        options=["sbert", "minilm", "tfidf"],
        format_func=lambda x: {
            "sbert": "SBERT (all-mpnet-base-v2)",
            "minilm": "MiniLM (all-MiniLM-L6-v2)",
            "tfidf": "TF-IDF Baseline"
        }[x],
        help="The RAG model used to select reference context for the simplification prompt."
    )
    
    st.divider()
    
    # Index Status / Rebuilding
    st.markdown("### Reference Index Status")
    if indices_built():
        st.success("✅ Reference Vector Store: Ready")
    else:
        st.warning("⚠️ Reference Vector Store: Not Built")
        
    if st.button("Rebuild All Vector Indices", help="Recompile the indexes using current knowledge base documents"):
        with st.spinner("Rebuilding indices... This may take a few seconds."):
            try:
                initialize_indices()
                st.success("Indices built successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error rebuilding indices: {str(e)}")

# ============================================================
# MAIN APPLICATION LAYOUT
# ============================================================
st.markdown('<div class="main-title">🏥 Medical Report Simplifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-assisted medical report explanation system utilizing Multi-Model Retrieval-Augmented Generation (RAG).</div>', unsafe_allow_html=True)

if not indices_built():
    st.info("👉 Welcome! Before analyzing any reports, please build the vector store by clicking the **'Rebuild All Vector Indices'** button in the sidebar.")
    st.stop()

# Initialize session state variables
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.original_text = ""
    st.session_state.cleaned_text = ""
    st.session_state.entities = None
    st.session_state.retrieval_comparison = None
    st.session_state.simplified_output = ""
    st.session_state.elapsed_times = {}

if uploaded_file is not None:
    # Trigger processing pipeline when a new file is uploaded
    file_name = uploaded_file.name
    
    # Build a process button to start the analysis
    if st.button("Process & Simplify Report", type="primary"):
        st.session_state.analysis_complete = False

        with st.status("⚙️ Running analysis pipeline...", expanded=True) as status:

            # 1. Validate Upload
            st.write("✅ Validating uploaded file...")
            val_result = validate_file(uploaded_file)
            if not val_result.is_valid:
                status.update(label="❌ Validation failed", state="error", expanded=True)
                st.error("\n".join(val_result.errors))
                st.stop()

            # 2. Extract Text
            st.write("📄 Extracting text...")
            start = time.time()
            file_type = get_file_type(uploaded_file)
            extracted_text = extract_text(uploaded_file, file_type)
            st.session_state.original_text = extracted_text
            st.session_state.elapsed_times["Extraction"] = time.time() - start

            # 3. Preprocess Text
            st.write("🔤 Running NLP preprocessing & token cleaning...")
            start = time.time()
            preprocessed = preprocess(extracted_text)
            st.session_state.cleaned_text = preprocessed.cleaned_text
            st.session_state.elapsed_times["Preprocessing"] = time.time() - start

            # 4. Medical NER
            st.write("🏷️ Identifying medical terms & measurements (SciSpacy)...")
            start = time.time()
            entities = extract_medical_entities(extracted_text)
            st.session_state.entities = entities
            st.session_state.elapsed_times["NER"] = time.time() - start

            # 5. Multi-Model Retrieval comparison
            st.write("🔍 Running retrieval models (TF-IDF · MiniLM · SBERT)...")
            start = time.time()
            query_terms = " ".join(entities.diseases + entities.tests_procedures)
            if not query_terms.strip():
                query_terms = " ".join(extracted_text.split()[:20])
            retrieval_comparison = compare_retrieval_models(query_terms)
            st.session_state.retrieval_comparison = retrieval_comparison
            st.session_state.elapsed_times["Retrieval"] = time.time() - start

            # 6. Gemini Simplification
            st.write("🤖 Querying Gemini to generate plain-language report...")
            start = time.time()
            context, _ = get_best_context(retrieval_comparison, preferred_retrieval_model)
            clean_llm_input = get_clean_text_for_llm(st.session_state.original_text)
            raw_simplification = simplify_report(clean_llm_input, context, entities.to_dict())
            st.session_state.elapsed_times["LLM Simplification"] = time.time() - start

            # Detect error sentinel returned by simplification engine
            if raw_simplification.startswith("ERROR:"):
                status.update(label="❌ Gemini API error", state="error", expanded=True)
                st.error(f"⚠️ Gemini API Error: {raw_simplification[6:].strip()}")
                st.info("💡 This is usually a quota or rate-limit issue. Wait a minute and try again, or check your API key at https://aistudio.google.com/apikey")
                st.stop()

            # Format and append reference glossary index
            formatted_output = format_simplified_report(raw_simplification, entities.to_dict())
            st.session_state.simplified_output = formatted_output

            status.update(label="🎉 Report processing complete!", state="complete", expanded=False)

        st.session_state.analysis_complete = True
        st.rerun()


else:
    # Clear state when file is removed
    st.session_state.analysis_complete = False
    st.info("Please upload a medical report file in the sidebar to begin.")

# ============================================================
# DISPLAY TABS
# ============================================================
if st.session_state.analysis_complete:
    tab1, tab2, tab3 = st.tabs([
        "📄 Simplified Report", 
        "🔍 Medical NER Explorer", 
        "📊 RAG Model Comparison"
    ])
    
    # --------------------------------------------------------
    # TAB 1: SIMPLIFIED REPORT
    # --------------------------------------------------------
    with tab1:
        st.markdown("### Plain-Language Summary Output")
        
        # Download PDF button
        pdf_bytes = generate_pdf_report(st.session_state.simplified_output)
        st.download_button(
            label="⬇️ Download Simplified Report (PDF)",
            data=pdf_bytes,
            file_name=f"simplified_{uploaded_file.name.replace('.pdf', '')}.pdf",
            mime="application/pdf"
        )
        
        # Render markdown content
        st.markdown(st.session_state.simplified_output)
        
        # Display performance times
        with st.expander("⏱️ Pipeline Performance Details"):
            st.write("Processing times for individual pipeline modules:")
            for stage, elapsed in st.session_state.elapsed_times.items():
                st.write(f"- **{stage}**: {elapsed:.2f} seconds")
            total_time = sum(st.session_state.elapsed_times.values())
            st.write(f"**Total Elapsed Pipeline Time**: {total_time:.2f} seconds")

    # --------------------------------------------------------
    # TAB 2: MEDICAL NER EXPLORER
    # --------------------------------------------------------
    with tab2:
        st.markdown("### Discovered Medical Entities")
        st.write("The system identified the following medical items using SciSpacy models and rule patterns:")
        
        entities_map = st.session_state.entities.to_dict()
        
        # Create columns for cards
        cols = st.columns(3)
        cat_keys = list(entities_map.keys())
        
        for i, col in enumerate(cols):
            with col:
                # Column 1: Diseases and Symptoms
                cat_1 = cat_keys[i * 2]
                st.markdown(f"#### 🏷️ {cat_1}")
                if entities_map[cat_1]:
                    st.write(", ".join([f"`{t}`" for t in entities_map[cat_1]]))
                else:
                    st.write("*None identified*")
                    
                # Column 2: Medications and Tests
                cat_2 = cat_keys[i * 2 + 1]
                st.markdown(f"#### 🏷️ {cat_2}")
                if entities_map[cat_2]:
                    st.write(", ".join([f"`{t}`" for t in entities_map[cat_2]]))
                else:
                    st.write("*None identified*")
                    
        st.divider()
        with st.expander("📄 View Extracted Raw Text"):
            st.text_area("Extracted text snippet:", st.session_state.original_text, height=300, disabled=True)

    # --------------------------------------------------------
    # TAB 3: RAG MODEL COMPARISON
    # --------------------------------------------------------
    with tab3:
        st.markdown("### Retrieval Model Performance side-by-side")
        st.write("We evaluate the medical reference guidelines retrieved across three models:")
        
        eval_report = evaluate_retrieval(
            " ".join(st.session_state.entities.diseases), 
            st.session_state.retrieval_comparison
        )
        
        df_eval = pd.DataFrame(eval_report)
        st.dataframe(df_eval, use_container_width=True)
        
        # Metrics Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Search Latency (ms) — Lower is Better")
            st.bar_chart(data=df_eval, x="Model Name", y="Latency (ms)")
            
        with col2:
            st.markdown("#### Maximum Match Score — Higher is Better")
            st.bar_chart(data=df_eval, x="Model Name", y="Max Match Score")
            
        # Display top retrieved texts side-by-side
        st.markdown("#### Top Retrieved Chunks Per Model")
        cols_models = st.columns(3)
        model_keys = ["tfidf", "minilm", "sbert"]
        for idx, key in enumerate(model_keys):
            with cols_models[idx]:
                output = st.session_state.retrieval_comparison[key]
                st.markdown(f"**{output.model_name}**")
                if output.results:
                    top_res = output.results[0]
                    st.info(f"**Score**: {top_res.score:.4f}\n\n**Source**: {top_res.title}\n\n{top_res.text[:250]}...")
                else:
                    st.write("No matching documents found.")


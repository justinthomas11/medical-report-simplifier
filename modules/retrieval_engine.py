"""
Module 8 — Multi-Model Retrieval Engine

Retrieves relevant medical guidelines and context using three different models:
1. TF-IDF + Cosine Similarity (Baseline)
2. MiniLM + FAISS (Dense Embeddings)
3. SBERT + FAISS (Dense Embeddings)

Allows side-by-side comparison of results, similarity scores, and execution latency.
"""

import time
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from config.settings import (
    VECTOR_STORE_DIR, MINILM_MODEL, SBERT_MODEL, TOP_K_RESULTS
)
from modules.logger import get_logger, log_execution_time
from modules.document_indexer import ChunkedText

logger = get_logger(__name__)


class RetrievalResult:
    """Standardized retrieval result for comparison."""
    def __init__(self, chunk: ChunkedText, score: float):
        self.text = chunk.text
        self.source_path = chunk.source_path
        self.category = chunk.category
        self.title = chunk.title
        self.score = float(score)


class ModelRetrievalOutput:
    """Output bundle for a single retrieval model."""
    def __init__(self, model_name: str, results: List[RetrievalResult], execution_time: float):
        self.model_name = model_name
        self.results = results
        self.execution_time = execution_time


# Global Cache for Models and Indices to prevent reloading every time
_MODELS_CACHE = {}


def _get_or_load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODELS_CACHE:
        logger.info(f"Loading sentence transformer model: {model_name}...")
        _MODELS_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODELS_CACHE[model_name]


def load_tfidf_resources() -> Tuple[Any, Any, List[ChunkedText]]:
    """Load pre-saved TF-IDF files."""
    vectorizer_path = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"
    matrix_path = VECTOR_STORE_DIR / "tfidf_matrix.pkl"
    chunks_path = VECTOR_STORE_DIR / "tfidf_chunks.pkl"
    
    if not (vectorizer_path.exists() and matrix_path.exists() and chunks_path.exists()):
        raise FileNotFoundError("TF-IDF index files not found. Build index first.")
        
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(matrix_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    return vectorizer, tfidf_matrix, chunks


def load_faiss_resources(suffix: str) -> Tuple[faiss.IndexFlatIP, List[ChunkedText]]:
    """Load FAISS index and chunks."""
    index_path = VECTOR_STORE_DIR / f"{suffix}.index"
    chunks_path = VECTOR_STORE_DIR / f"{suffix}_chunks.pkl"
    
    if not (index_path.exists() and chunks_path.exists()):
        raise FileNotFoundError(f"FAISS index files for {suffix} not found. Build index first.")
        
    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    return index, chunks


@log_execution_time
def retrieve_tfidf(query: str, top_k: int = TOP_K_RESULTS) -> ModelRetrievalOutput:
    """Retrieve relevant chunks using TF-IDF + Cosine Similarity."""
    start_time = time.time()
    try:
        vectorizer, tfidf_matrix, chunks = load_tfidf_resources()
        query_vec = vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            # Only include chunks with positive similarity or top 1 anyway
            if score > 0.0 or len(results) == 0:
                results.append(RetrievalResult(chunks[idx], score))
                
        elapsed = time.time() - start_time
        return ModelRetrievalOutput("TF-IDF Baseline", results, elapsed)
    except Exception as e:
        logger.error(f"TF-IDF retrieval error: {str(e)}")
        return ModelRetrievalOutput("TF-IDF Baseline", [], 0.0)


@log_execution_time
def retrieve_dense(query: str, model_name: str, index_suffix: str, top_k: int = TOP_K_RESULTS) -> ModelRetrievalOutput:
    """Retrieve relevant chunks using dense vector embeddings + FAISS."""
    start_time = time.time()
    try:
        model = _get_or_load_model(model_name)
        index, chunks = load_faiss_resources(index_suffix)
        
        # Generate and normalize query embedding
        query_emb = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search index
        scores, indices = index.search(query_emb, top_k)
        
        results = []
        # scores and indices are arrays of shape (1, top_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 represents no match/out of bounds in FAISS
                results.append(RetrievalResult(chunks[idx], score))
                
        elapsed = time.time() - start_time
        return ModelRetrievalOutput(f"Dense FAISS ({index_suffix.upper()})", results, elapsed)
    except Exception as e:
        logger.error(f"Dense FAISS ({index_suffix}) retrieval error: {str(e)}")
        return ModelRetrievalOutput(f"Dense FAISS ({index_suffix.upper()})", [], 0.0)


def compare_retrieval_models(query: str, top_k: int = TOP_K_RESULTS) -> Dict[str, ModelRetrievalOutput]:
    """
    Run all three retrieval models and return side-by-side results.
    """
    comparison = {}
    
    # 1. TF-IDF
    comparison["tfidf"] = retrieve_tfidf(query, top_k)
    
    # 2. MiniLM
    comparison["minilm"] = retrieve_dense(query, MINILM_MODEL, "minilm", top_k)
    
    # 3. SBERT
    comparison["sbert"] = retrieve_dense(query, SBERT_MODEL, "sbert", top_k)
    
    return comparison


def get_best_context(comparison: Dict[str, ModelRetrievalOutput], preferred_model: str = "sbert") -> Tuple[str, List[RetrievalResult]]:
    """
    Selects the best retrieved context from the comparison.
    Defaults to the preferred_model if it has results, otherwise falls back.
    
    Returns:
        Tuple of (merged_context_string, list_of_retrieved_results)
    """
    model_key = preferred_model if preferred_model in comparison else "sbert"
    output = comparison.get(model_key)
    
    # Fallback cascade: sbert -> minilm -> tfidf
    if not output or not output.results:
        output = comparison.get("minilm")
    if not output or not output.results:
        output = comparison.get("tfidf")
        
    if not output or not output.results:
        return "No reference guidelines found.", []
        
    # Merge retrieved contents into a coherent context block
    unique_texts = []
    seen = set()
    for res in output.results:
        clean_text = res.text.strip()
        if clean_text not in seen:
            seen.add(clean_text)
            unique_texts.append(f"Source: [{res.title}]\n{clean_text}")
            
    merged_context = "\n\n---\n\n".join(unique_texts)
    return merged_context, output.results

"""
Module 7 — Document Processing & Indexing

Handles text chunking, embedding generation, and FAISS vector index creation/storage.
Supports:
1. TF-IDF indexing (scikit-learn)
2. MiniLM indexing (all-MiniLM-L6-v2 + FAISS)
3. SBERT indexing (all-mpnet-base-v2 + FAISS)
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR, 
    MINILM_MODEL, SBERT_MODEL
)
from modules.logger import get_logger, log_execution_time
from modules.knowledge_base import ReferenceDocument, load_knowledge_base

logger = get_logger(__name__)


class ChunkedText:
    """Represents a chunk of text with its source document metadata."""
    def __init__(self, text: str, source_doc: ReferenceDocument, chunk_index: int):
        self.text = text
        self.source_path = str(source_doc.file_path)
        self.category = source_doc.category
        self.title = source_doc.title
        self.chunk_index = chunk_index

# Explicitly override the module namespace for pickle serialization
ChunkedText.__module__ = "modules.document_indexer"



def smart_chunk(documents: List[ReferenceDocument]) -> List[ChunkedText]:
    """
    Split document contents into overlapping chunks of CHUNK_SIZE characters.
    
    Args:
        documents: List of ReferenceDocument.
    
    Returns:
        List of ChunkedText.
    """
    chunks = []
    
    for doc in documents:
        content = doc.content
        if len(content) <= CHUNK_SIZE:
            chunks.append(ChunkedText(content, doc, 0))
            continue
            
        start = 0
        chunk_idx = 0
        while start < len(content):
            end = start + CHUNK_SIZE
            
            # Try to align end to a sentence boundary or word boundary
            if end < len(content):
                # Search backwards for a period/newline to end chunk cleanly
                boundary = -1
                for offset in range(0, min(100, CHUNK_SIZE // 2)):
                    char_pos = end - offset
                    if content[char_pos] in ['.', '\n']:
                        boundary = char_pos + 1
                        break
                
                if boundary != -1:
                    end = boundary
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(ChunkedText(chunk_text, doc, chunk_idx))
                chunk_idx += 1
                
            start = end - CHUNK_OVERLAP
            # Safety checks to prevent infinite loop
            if CHUNK_OVERLAP >= CHUNK_SIZE or end <= start:
                start = end
                
    logger.info(f"Chunked {len(documents)} documents into {len(chunks)} text chunks.")
    return chunks


@log_execution_time
def build_tfidf_index(chunks: List[ChunkedText]) -> Tuple[TfidfVectorizer, Any, Path]:
    """Build and save TF-IDF Vectorizer and representation matrix."""
    texts = [c.text for c in chunks]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Save objects
    vectorizer_path = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"
    matrix_path = VECTOR_STORE_DIR / "tfidf_matrix.pkl"
    chunks_path = VECTOR_STORE_DIR / "tfidf_chunks.pkl"
    
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
        
    logger.info("TF-IDF Index saved successfully.")
    return vectorizer, tfidf_matrix, chunks_path


@log_execution_time
def build_faiss_index(chunks: List[ChunkedText], model_name: str, index_suffix: str) -> Tuple[SentenceTransformer, faiss.IndexFlatIP, Path]:
    """
    Build and save FAISS index for a sentence transformer model.
    Uses Inner Product (Cosine Similarity on normalized embeddings).
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks using {model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Normalize embeddings for Cosine Similarity (Inner Product)
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save index and chunks
    index_path = VECTOR_STORE_DIR / f"{index_suffix}.index"
    chunks_path = VECTOR_STORE_DIR / f"{index_suffix}_chunks.pkl"
    
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
        
    logger.info(f"FAISS index {index_suffix} saved successfully.")
    return model, index, chunks_path


def initialize_indices():
    """
    Wrapper function to rebuild all indices. Run when KB is loaded or changes.
    """
    docs = load_knowledge_base()
    if not docs:
        logger.error("No documents to index.")
        return
        
    chunks = smart_chunk(docs)
    
    # 1. TF-IDF
    build_tfidf_index(chunks)
    
    # 2. MiniLM
    build_faiss_index(chunks, MINILM_MODEL, "minilm")
    
    # 3. SBERT
    build_faiss_index(chunks, SBERT_MODEL, "sbert")
    
    logger.info("All indices successfully generated.")


if __name__ == "__main__":
    initialize_indices()


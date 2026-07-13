"""
Module 4 — Text Preprocessing

Cleans and normalizes extracted text:
- Lowercasing
- Remove special characters (preserve medical terms)
- Remove extra spaces
- Sentence segmentation
- Tokenization
- Stopword removal
- Lemmatization
"""

import re
from dataclasses import dataclass, field
from typing import List

import spacy
import nltk

from modules.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Download required NLTK data (once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@dataclass
class PreprocessedText:
    """Result of text preprocessing."""
    original_text: str = ""
    cleaned_text: str = ""
    sentences: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    lemmatized_tokens: List[str] = field(default_factory=list)
    num_sentences: int = 0
    num_tokens: int = 0


@log_execution_time
def preprocess(text: str) -> PreprocessedText:
    """
    Full text preprocessing pipeline.
    
    Pipeline:
    1. Remove extra whitespace
    2. Clean special characters (preserve medical-relevant ones)
    3. Sentence segmentation
    4. Tokenization
    5. Stopword removal
    6. Lemmatization
    
    Args:
        text: Raw extracted text.
    
    Returns:
        PreprocessedText with all preprocessing stages.
    """
    result = PreprocessedText(original_text=text)
    
    if not text or not text.strip():
        logger.warning("Empty text received for preprocessing")
        return result
    
    # Step 1: Remove extra whitespace
    cleaned = _normalize_whitespace(text)
    
    # Step 2: Clean special characters (preserve medical terms)
    cleaned = _clean_special_characters(cleaned)
    
    result.cleaned_text = cleaned
    
    # Step 3: Sentence segmentation
    result.sentences = _segment_sentences(cleaned)
    result.num_sentences = len(result.sentences)
    
    # Step 4 & 5 & 6: Tokenize, remove stopwords, lemmatize using spaCy
    tokens, lemmatized = _tokenize_and_lemmatize(cleaned)
    result.tokens = tokens
    result.lemmatized_tokens = lemmatized
    result.num_tokens = len(tokens)
    
    logger.info(
        f"Preprocessing complete: {result.num_sentences} sentences, "
        f"{result.num_tokens} tokens"
    )
    
    return result


def _normalize_whitespace(text: str) -> str:
    """Remove extra whitespace, tabs, and normalize line breaks."""
    # Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Remove multiple consecutive spaces
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def _clean_special_characters(text: str) -> str:
    """
    Remove special characters while preserving:
    - Medical units (mg/dL, mmol/L, etc.)
    - Numbers and decimals (3.5, 120/80)
    - Common medical punctuation (-, /, .)
    - Parentheses (used in medical terms)
    """
    # Preserve alphanumeric, spaces, and medical-relevant characters
    # Keep: letters, digits, spaces, periods, commas, hyphens, slashes,
    #        parentheses, colons, semicolons, percent, plus, equals, newlines
    text = re.sub(r'[^\w\s\.\,\-\/\(\)\:\;\%\+\=\n\>\<]', ' ', text)
    # Clean up any resulting double spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences using NLTK.
    
    Args:
        text: Cleaned text.
    
    Returns:
        List of sentences.
    """
    from nltk.tokenize import sent_tokenize
    
    sentences = sent_tokenize(text)
    # Filter out very short "sentences" (likely fragments)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences


def _tokenize_and_lemmatize(text: str):
    """
    Tokenize text, remove stopwords, and lemmatize using spaCy.
    
    Args:
        text: Cleaned text.
    
    Returns:
        Tuple of (tokens, lemmatized_tokens).
    """
    doc = nlp(text)
    
    tokens = []
    lemmatized = []
    
    for token in doc:
        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue
        
        tokens.append(token.text.lower())
        
        # Skip stopwords for lemmatized output
        if not token.is_stop:
            lemmatized.append(token.lemma_.lower())
    
    return tokens, lemmatized


def get_clean_text_for_llm(text: str) -> str:
    """
    Light cleaning for LLM input — preserves readability and structure,
    just removes noise. Used before sending to the simplification engine.
    
    Args:
        text: Raw extracted text.
    
    Returns:
        Lightly cleaned text suitable for LLM processing.
    """
    # Normalize whitespace
    text = _normalize_whitespace(text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()

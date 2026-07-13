"""
Modules 1 & 2 — Input Handling & Validation

Handles file upload validation:
- File type check (PDF, PNG, JPG, TXT)
- File size check (< 10MB)
- Page count check (< 50 pages for PDFs)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from config.settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_PAGE_COUNT
from modules.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool = True
    file_name: str = ""
    file_size_mb: float = 0.0
    file_extension: str = ""
    page_count: Optional[int] = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_file(uploaded_file) -> ValidationResult:
    """
    Validate an uploaded file (Streamlit UploadedFile object).
    
    Checks:
    1. File is not None
    2. File extension is allowed
    3. File size is within limits
    4. Page count is within limits (PDF only)
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
    
    Returns:
        ValidationResult with validation status and any errors.
    """
    result = ValidationResult()
    
    # Check if file exists
    if uploaded_file is None:
        result.is_valid = False
        result.errors.append("No file uploaded.")
        logger.warning("Validation failed: No file uploaded")
        return result
    
    result.file_name = uploaded_file.name
    
    # Check file extension
    file_ext = Path(uploaded_file.name).suffix.lower()
    result.file_extension = file_ext
    
    if file_ext not in ALLOWED_EXTENSIONS:
        result.is_valid = False
        result.errors.append(
            f"Unsupported file type: '{file_ext}'. "
            f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        logger.warning(f"Validation failed: Unsupported file type '{file_ext}'")
        return result
    
    # Check file size
    file_size_bytes = uploaded_file.size
    file_size_mb = file_size_bytes / (1024 * 1024)
    result.file_size_mb = round(file_size_mb, 2)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        result.is_valid = False
        result.errors.append(
            f"File too large: {result.file_size_mb}MB. "
            f"Maximum allowed: {MAX_FILE_SIZE_MB}MB."
        )
        logger.warning(f"Validation failed: File too large ({result.file_size_mb}MB)")
        return result
    
    # Check page count for PDFs
    if file_ext == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            page_count = len(reader.pages)
            result.page_count = page_count
            
            if page_count > MAX_PAGE_COUNT:
                result.is_valid = False
                result.errors.append(
                    f"Too many pages: {page_count}. "
                    f"Maximum allowed: {MAX_PAGE_COUNT} pages."
                )
                logger.warning(f"Validation failed: Too many pages ({page_count})")
                return result
            
            if page_count == 0:
                result.is_valid = False
                result.errors.append("PDF has no readable pages.")
                logger.warning("Validation failed: PDF has 0 pages")
                return result
            
            # Reset file pointer after reading
            uploaded_file.seek(0)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Could not read PDF: {str(e)}")
            logger.error(f"PDF reading error: {str(e)}")
            return result
    
    # Validation passed
    logger.info(
        f"Validation passed: {result.file_name} "
        f"({result.file_size_mb}MB, {result.file_extension})"
    )
    return result


def get_file_type(uploaded_file) -> str:
    """
    Determine the category of the uploaded file.
    
    Returns:
        'pdf', 'image', or 'text'
    """
    ext = Path(uploaded_file.name).suffix.lower()
    
    if ext == ".pdf":
        return "pdf"
    elif ext in {".png", ".jpg", ".jpeg"}:
        return "image"
    elif ext == ".txt":
        return "text"
    else:
        return "unknown"

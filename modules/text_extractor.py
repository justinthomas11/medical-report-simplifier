"""
Module 3 — Text Extraction

Extracts text from uploaded files:
- PDF: Uses pdfplumber for text-based PDFs
- Images: Uses EasyOCR for scanned documents
- Text: Direct read
- Fallback: If PDF yields insufficient text, switches to OCR
"""

import io
from typing import Optional

import pdfplumber
from PIL import Image

from config.settings import MIN_EXTRACTED_TEXT_LENGTH
from modules.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@log_execution_time
def extract_text(uploaded_file, file_type: str) -> str:
    """
    Extract text from an uploaded file based on its type.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        file_type: One of 'pdf', 'image', or 'text'.
    
    Returns:
        Extracted text as a string.
    """
    if file_type == "pdf":
        return extract_from_pdf(uploaded_file)
    elif file_type == "image":
        return extract_from_image(uploaded_file)
    elif file_type == "text":
        return extract_from_text(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def extract_from_pdf(uploaded_file) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    Falls back to OCR if extracted text is too short.
    
    Args:
        uploaded_file: PDF file (Streamlit UploadedFile).
    
    Returns:
        Extracted text string.
    """
    logger.info(f"Extracting text from PDF: {uploaded_file.name}")
    
    text_parts = []
    
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    logger.debug(f"Page {i+1}: extracted {len(page_text)} chars")
                else:
                    logger.debug(f"Page {i+1}: no text found")
        
        full_text = "\n\n".join(text_parts).strip()
        
        # Fallback to OCR if insufficient text
        if len(full_text) < MIN_EXTRACTED_TEXT_LENGTH:
            logger.info(
                f"PDF text too short ({len(full_text)} chars), "
                f"falling back to OCR..."
            )
            uploaded_file.seek(0)
            return _pdf_to_ocr(uploaded_file)
        
        logger.info(f"PDF extraction successful: {len(full_text)} chars")
        return full_text
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        # Try OCR as fallback
        try:
            uploaded_file.seek(0)
            return _pdf_to_ocr(uploaded_file)
        except Exception as ocr_error:
            logger.error(f"OCR fallback also failed: {str(ocr_error)}")
            raise RuntimeError(
                f"Could not extract text from PDF: {str(e)}"
            ) from e


def _pdf_to_ocr(uploaded_file) -> str:
    """
    Convert PDF pages to images and run OCR.
    
    Args:
        uploaded_file: PDF file object.
    
    Returns:
        OCR-extracted text.
    """
    logger.info("Converting PDF to images for OCR...")
    
    try:
        # Use PyPDF2 to get page count, then convert each page
        import PyPDF2
        
        # For PDF OCR, we need pdf2image or similar
        # Alternative: extract images embedded in PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            # Try to extract images from PDF pages
            if hasattr(page, 'images') and page.images:
                for img in page.images:
                    image = Image.open(io.BytesIO(img.data))
                    ocr_text = _run_ocr_on_image(image)
                    if ocr_text:
                        text_parts.append(ocr_text)
        
        if text_parts:
            return "\n\n".join(text_parts).strip()
        
        # If no images found, return what we can
        logger.warning("No extractable images found in PDF for OCR")
        return "Unable to extract text from this PDF. The document may be encrypted or corrupted."
        
    except Exception as e:
        logger.error(f"PDF-to-OCR conversion failed: {str(e)}")
        raise


def extract_from_image(uploaded_file) -> str:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        uploaded_file: Image file (Streamlit UploadedFile).
    
    Returns:
        OCR-extracted text string.
    """
    logger.info(f"Extracting text from image: {uploaded_file.name}")
    
    try:
        image = Image.open(uploaded_file)
        text = _run_ocr_on_image(image)
        
        if not text or len(text.strip()) < 10:
            logger.warning("OCR produced very little text from image")
            return "Unable to extract meaningful text from this image. Please ensure the image is clear and contains readable text."
        
        logger.info(f"Image OCR successful: {len(text)} chars")
        return text
        
    except Exception as e:
        logger.error(f"Image OCR failed: {str(e)}")
        raise RuntimeError(f"Could not extract text from image: {str(e)}") from e


def _run_ocr_on_image(image: Image.Image) -> str:
    """
    Run EasyOCR on a PIL Image.
    
    Args:
        image: PIL Image object.
    
    Returns:
        Extracted text.
    """
    import numpy as np
    import easyocr
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Initialize EasyOCR reader (English)
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Run OCR
    results = reader.readtext(img_array, detail=0, paragraph=True)
    
    return "\n".join(results)


def extract_from_text(uploaded_file) -> str:
    """
    Read text directly from a .txt file.
    
    Args:
        uploaded_file: Text file (Streamlit UploadedFile).
    
    Returns:
        File contents as string.
    """
    logger.info(f"Reading text file: {uploaded_file.name}")
    
    try:
        content = uploaded_file.read()
        
        # Handle bytes vs string
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = content
        
        logger.info(f"Text file read: {len(text)} chars")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Text file reading failed: {str(e)}")
        raise RuntimeError(f"Could not read text file: {str(e)}") from e

"""
Module 6 — Knowledge Base

Loads and manages the reference corpus of medical documents.
"""

from pathlib import Path
from typing import List, Dict
from config.settings import KNOWLEDGE_BASE_DIR
from modules.logger import get_logger, log_execution_time

logger = get_logger(__name__)


from dataclasses import dataclass

@dataclass
class ReferenceDocument:
    """Represents a medical reference document."""
    file_path: Path
    category: str
    title: str
    content: str


@log_execution_time
def load_knowledge_base() -> List[ReferenceDocument]:
    """
    Scan the KNOWLEDGE_BASE_DIR and load all text files recursively.
    
    Returns:
        List of ReferenceDocument objects.
    """
    documents = []
    
    if not KNOWLEDGE_BASE_DIR.exists():
        logger.warning(f"Knowledge base directory does not exist: {KNOWLEDGE_BASE_DIR}")
        return documents
        
    # Traverse directory recursively
    for txt_file in KNOWLEDGE_BASE_DIR.glob("**/*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8")
            
            # Extract category from parent directory name
            category = txt_file.parent.name
            
            # Simple title extraction: first non-empty line
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            title = lines[0] if lines else txt_file.stem.replace("_", " ").title()
            
            doc = ReferenceDocument(
                file_path=txt_file,
                category=category,
                title=title,
                content=content
            )
            documents.append(doc)
            logger.debug(f"Loaded reference doc: {txt_file.relative_to(KNOWLEDGE_BASE_DIR.parent)}")
        except Exception as e:
            logger.error(f"Failed to load reference doc {txt_file}: {str(e)}")
            
    logger.info(f"Loaded {len(documents)} reference documents from knowledge base.")
    return documents

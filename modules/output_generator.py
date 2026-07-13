"""
Module 11 — Output Generation

Structures and formats the simplified medical report output.
"""

from typing import Dict, List
from modules.logger import get_logger

logger = get_logger(__name__)


def format_simplified_report(simplified_text: str, entities_dict: Dict[str, List[str]]) -> str:
    """
    Format the simplified text returned by the LLM. Ensures formatting standards are met.
    
    Args:
        simplified_text: Markdown output from LLM.
        entities_dict: Extracted medical entities list.
        
    Returns:
        Structured output string.
    """
    # The Gemini prompt enforces a clean structure, so we mostly return it clean.
    # However, we can append a glossary/extracted terms key index at the bottom for patient reference.
    
    glossary_lines = []
    for category, terms in entities_dict.items():
        if terms:
            glossary_lines.append(f"**{category}**: {', '.join(terms)}")
            
    glossary_section = ""
    if glossary_lines:
        glossary_section = (
            "\n\n---\n\n"
            "### 📚 Medical Glossary (Terms Found in Your Report)\n"
            "Here are the medical terms identified in your report and simplified above:\n\n"
        ) + "\n".join([f"- {line}" for line in glossary_lines])
        
    return simplified_text + glossary_section

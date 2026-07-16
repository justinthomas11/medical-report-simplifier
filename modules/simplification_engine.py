"""
Module 9 — LLM Simplification Engine

Integrates with the Google Gemini API (gemini-1.5-flash) to simplify 
medical reports using retrieved context and extracted NER terms.
"""

import google.generativeai as genai
from config.settings import (
    GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE, 
    LLM_MAX_TOKENS, SYSTEM_PROMPT
)
from modules.logger import get_logger, log_execution_time

logger = get_logger(__name__)

# Configure the Gemini API client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY is not set in configuration settings.")


def build_simplification_prompt(report_text: str, retrieved_context: str, extracted_entities: dict) -> str:
    """
    Construct the full prompt for the Gemini LLM.
    
    Args:
        report_text: The cleaned, raw text of the medical report.
        retrieved_context: Medical guideline reference text retrieved from the knowledge base.
        extracted_entities: Dictionary of categorized medical terms found in the text.
        
    Returns:
        Formated prompt string.
    """
    # Build entity list string
    entity_str = ""
    for category, entities in extracted_entities.items():
        if entities:
            entity_str += f"- {category}: {', '.join(entities)}\n"
            
    prompt = f"""
ORIGINAL MEDICAL REPORT:
{report_text}

---
RETRIEVED MEDICAL GUIDELINES & REFERENCE CONTEXT:
{retrieved_context}

---
EXTRACTED MEDICAL TERMS FOR REFERENCE:
{entity_str if entity_str else "None detected."}

---
TASK:
Simplify the "ORIGINAL MEDICAL REPORT" using the "RETRIEVED MEDICAL GUIDELINES & REFERENCE CONTEXT" to explain terms and provide accurate context. Follow the system rules exactly. Produce a plain language explanation that is easy to read, comforting, and clear.
"""
    return prompt


def get_supported_model(preferred_model: str) -> str:
    """Find a supported model name, falling back if the preferred one is not available."""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        pref_full = f"models/{preferred_model}" if not preferred_model.startswith("models/") else preferred_model
        if pref_full in models:
            return preferred_model
        
        # Strip models/ prefix to compare
        model_names = [m.replace("models/", "") for m in models]
        
        # Check standard models in order of preference
        fallbacks = [preferred_model, "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"]
        for fb in fallbacks:
            if fb in model_names:
                logger.info(f"Using fallback model: {fb}")
                return fb
                
        # If none of fallbacks are in list, return first available or default to preferred
        if model_names:
            logger.info(f"Model {preferred_model} not found. Using first available: {model_names[0]}")
            return model_names[0]
    except Exception as e:
        logger.warning(f"Failed to list models: {str(e)}. Defaulting to {preferred_model}")
    return preferred_model


@log_execution_time
def simplify_report(report_text: str, retrieved_context: str, extracted_entities: dict) -> str:
    """
    Send the report, retrieved context, and terms to Gemini for simplification.
    
    Args:
        report_text: Raw extracted report text.
        retrieved_context: Context string retrieved from KB.
        extracted_entities: Categorized entity dictionary.
        
    Returns:
        Plain-language simplified text.
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key is missing. Cannot proceed with simplification.")
        return (
            "Error: Gemini API key is missing. "
            "Please add your GEMINI_API_KEY to the .env file in the root folder."
        )
        
    try:
        # Determine the best model to use
        actual_model_name = get_supported_model(GEMINI_MODEL)
        
        # Build prompt
        prompt = build_simplification_prompt(report_text, retrieved_context, extracted_entities)
        
        # Configure model parameters
        generation_config = genai.types.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        )
        
        # Instantiate model
        model = genai.GenerativeModel(
            model_name=actual_model_name,
            system_instruction=SYSTEM_PROMPT
        )
        
        logger.info("Calling Gemini API for simplification...")
        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )
        
        if not response.text:
            logger.warning("Gemini API returned an empty response.")
            return "Unable to simplify report. The model returned an empty response."
            
        logger.info("Successfully received simplified output from Gemini.")
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini simplification failed: {str(e)}")
        return f"An error occurred while simplifying the report using Gemini: {str(e)}"

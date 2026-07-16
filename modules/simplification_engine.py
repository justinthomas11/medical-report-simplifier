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


def _get_candidate_models(preferred_model: str):
    """
    Return an ordered list of model names to attempt, starting with the
    preferred model, then known fallbacks.  Always yields at least one entry.
    """
    _FALLBACK_ORDER = [
        preferred_model,
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-pro",
    ]
    try:
        available = {
            m.name.replace("models/", "")
            for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        }
        logger.info(f"Available Gemini models: {sorted(available)}")
        # Yield candidates that exist in the API, preserving fallback order
        seen = set()
        for name in _FALLBACK_ORDER:
            if name not in seen and name in available:
                seen.add(name)
                yield name
    except Exception as e:
        logger.warning(f"Could not list models: {e}. Yielding preferred model only.")
        yield preferred_model


@log_execution_time
def simplify_report(report_text: str, retrieved_context: str, extracted_entities: dict) -> str:
    """
    Send the report, retrieved context, and terms to Gemini for simplification.
    Tries each available model in fallback order, skipping those that return
    404 (not found) or 429 (quota exceeded) errors.

    Returns:
        Plain-language simplified text, or an error sentinel string prefixed
        with 'ERROR:' so the caller can detect and surface it properly.
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key is missing. Cannot proceed with simplification.")
        return "ERROR: Gemini API key is missing. Please add GEMINI_API_KEY to your .env file."

    prompt = build_simplification_prompt(report_text, retrieved_context, extracted_entities)
    generation_config = genai.types.GenerationConfig(
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
    )

    last_error = "No models were available to try."
    for model_name in _get_candidate_models(GEMINI_MODEL):
        try:
            logger.info(f"Trying model: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT
            )
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config
            )
            if not response.text:
                logger.warning(f"Model {model_name} returned empty response.")
                last_error = f"Model {model_name} returned an empty response."
                continue

            logger.info(f"Successfully received output from model: {model_name}")
            return response.text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                logger.warning(f"Model {model_name} quota/rate-limit exceeded. Trying next fallback.")
                last_error = f"Model {model_name} quota exceeded."
            elif "404" in err_str or "not found" in err_str.lower() or "no longer available" in err_str.lower():
                logger.warning(f"Model {model_name} not found/deprecated. Trying next fallback.")
                last_error = f"Model {model_name} is not available."
            else:
                # Unexpected error — don't retry, surface immediately
                logger.error(f"Unexpected error with model {model_name}: {err_str}")
                return f"ERROR: {err_str}"

    logger.error(f"All Gemini models exhausted. Last error: {last_error}")
    return f"ERROR: All available Gemini models are currently rate-limited or unavailable. {last_error}"


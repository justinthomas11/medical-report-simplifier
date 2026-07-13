"""
Module 5 — Information Extraction (Medical NER)

Extracts medical entities from text using SciSpacy:
- Diseases / Conditions
- Symptoms
- Medications
- Tests / Procedures
- Body Parts
- Values / Measurements

Falls back to rule-based extraction if SciSpacy model is unavailable.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict

from modules.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class MedicalEntity:
    """A single extracted medical entity."""
    text: str
    label: str
    start: int = 0
    end: int = 0
    confidence: float = 0.0


@dataclass
class MedicalInfoMap:
    """Structured map of all extracted medical information."""
    diseases: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    tests_procedures: List[str] = field(default_factory=list)
    body_parts: List[str] = field(default_factory=list)
    values_measurements: List[str] = field(default_factory=list)
    raw_entities: List[MedicalEntity] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to a dictionary for display."""
        return {
            "Diseases / Conditions": self.diseases,
            "Symptoms": self.symptoms,
            "Medications": self.medications,
            "Tests / Procedures": self.tests_procedures,
            "Body Parts": self.body_parts,
            "Values / Measurements": self.values_measurements,
        }
    
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return any([
            self.diseases, self.symptoms, self.medications,
            self.tests_procedures, self.body_parts, self.values_measurements
        ])


# ============================================================
# SciSpacy-based extraction
# ============================================================

_scispacy_nlp = None


def _load_scispacy_model():
    """Load SciSpacy NER model (lazy loading)."""
    global _scispacy_nlp
    if _scispacy_nlp is not None:
        return _scispacy_nlp
    
    try:
        import spacy
        _scispacy_nlp = spacy.load("en_ner_bc5cdr_md")
        logger.info("SciSpacy model 'en_ner_bc5cdr_md' loaded successfully")
        return _scispacy_nlp
    except OSError:
        logger.warning(
            "SciSpacy model 'en_ner_bc5cdr_md' not found. "
            "Install with: pip install "
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
            "v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz"
        )
        return None


@log_execution_time
def extract_medical_entities(text: str) -> MedicalInfoMap:
    """
    Extract medical entities from text.
    
    Uses SciSpacy NER if available, falls back to rule-based extraction.
    
    Args:
        text: Input text (can be raw or preprocessed).
    
    Returns:
        MedicalInfoMap with categorized entities.
    """
    info_map = MedicalInfoMap()
    
    if not text or not text.strip():
        logger.warning("Empty text received for entity extraction")
        return info_map
    
    # Try SciSpacy first
    scispacy_model = _load_scispacy_model()
    
    if scispacy_model is not None:
        info_map = _extract_with_scispacy(text, scispacy_model)
    
    # Always supplement with rule-based extraction
    _extract_with_rules(text, info_map)
    
    # Deduplicate all lists
    info_map.diseases = _deduplicate(info_map.diseases)
    info_map.symptoms = _deduplicate(info_map.symptoms)
    info_map.medications = _deduplicate(info_map.medications)
    info_map.tests_procedures = _deduplicate(info_map.tests_procedures)
    info_map.body_parts = _deduplicate(info_map.body_parts)
    info_map.values_measurements = _deduplicate(info_map.values_measurements)
    
    total = sum(len(v) for v in info_map.to_dict().values())
    logger.info(f"Entity extraction complete: {total} entities found")
    
    return info_map


def _extract_with_scispacy(text: str, model) -> MedicalInfoMap:
    """
    Extract entities using SciSpacy NER model.
    
    The en_ner_bc5cdr_md model recognizes:
    - DISEASE: diseases and conditions
    - CHEMICAL: drugs and chemicals
    """
    info_map = MedicalInfoMap()
    
    # Process in chunks if text is long (spaCy has a max length)
    max_length = 100000
    if len(text) > max_length:
        text = text[:max_length]
    
    doc = model(text)
    
    for ent in doc.ents:
        entity = MedicalEntity(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char
        )
        info_map.raw_entities.append(entity)
        
        if ent.label_ == "DISEASE":
            info_map.diseases.append(ent.text)
        elif ent.label_ == "CHEMICAL":
            info_map.medications.append(ent.text)
    
    return info_map


def _extract_with_rules(text: str, info_map: MedicalInfoMap):
    """
    Rule-based extraction to supplement NER results.
    Catches values, measurements, tests, body parts, and symptoms
    that NER models often miss.
    """
    text_lower = text.lower()
    
    # ---- Values & Measurements ----
    # Match patterns like "120/80 mmHg", "5.6 mg/dL", "98.6°F"
    value_patterns = [
        r'\d+\.?\d*\s*(?:mg/dl|mg/dL|mmol/L|mmol/l|g/dL|g/dl|mEq/L|meq/l)',
        r'\d+\.?\d*\s*(?:IU/L|iu/l|U/L|u/l|mIU/mL)',
        r'\d+\.?\d*\s*(?:cells/mcL|cells/μL|x10\^?\d*/[uμ]L)',
        r'\d+\.?\d*\s*(?:mm/hr|mm/h|sec|seconds)',
        r'\d+\.?\d*\s*(?:%|percent)',
        r'\d+/\d+\s*(?:mmHg|mm\s*Hg)',
        r'\d+\.?\d*\s*(?:°[FC]|degrees)',
        r'\d+\.?\d*\s*(?:kg|lbs?|pounds?)',
        r'\d+\.?\d*\s*(?:bpm|beats per minute)',
        r'\d+\.?\d*\s*(?:fL|pg|g/dL)',
    ]
    for pattern in value_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info_map.values_measurements.extend(matches)
    
    # ---- Common Lab Tests ----
    test_keywords = [
        "complete blood count", "cbc", "hemoglobin", "hematocrit",
        "white blood cell", "wbc", "red blood cell", "rbc", "platelet",
        "lipid panel", "cholesterol", "triglycerides", "hdl", "ldl",
        "blood glucose", "fasting glucose", "hba1c", "a1c",
        "creatinine", "bun", "blood urea nitrogen", "gfr",
        "alt", "ast", "alkaline phosphatase", "bilirubin",
        "tsh", "t3", "t4", "thyroid",
        "urinalysis", "urine analysis",
        "ecg", "ekg", "electrocardiogram",
        "x-ray", "xray", "mri", "ct scan", "ultrasound",
        "blood pressure", "bp", "heart rate", "pulse",
        "bmi", "body mass index",
    ]
    for keyword in test_keywords:
        if keyword in text_lower:
            info_map.tests_procedures.append(keyword.upper() if len(keyword) <= 4 else keyword.title())
    
    # ---- Body Parts ----
    body_keywords = [
        "heart", "liver", "kidney", "lung", "brain", "stomach",
        "intestine", "pancreas", "thyroid", "spleen", "bladder",
        "bone", "blood", "chest", "abdomen", "head", "spine",
        "artery", "vein", "colon", "breast", "prostate",
    ]
    for keyword in body_keywords:
        if keyword in text_lower:
            info_map.body_parts.append(keyword.title())
    
    # ---- Common Symptoms ----
    symptom_keywords = [
        "pain", "fever", "fatigue", "nausea", "vomiting",
        "headache", "dizziness", "shortness of breath", "cough",
        "swelling", "inflammation", "bleeding", "weakness",
        "chest pain", "abdominal pain", "back pain",
        "weight loss", "weight gain", "loss of appetite",
        "frequent urination", "blurred vision",
    ]
    for keyword in symptom_keywords:
        if keyword in text_lower:
            info_map.symptoms.append(keyword.title())


def _deduplicate(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        normalized = item.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(item.strip())
    return result

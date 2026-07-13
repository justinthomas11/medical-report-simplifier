"""
Module 10 — Model Evaluation

Calculates performance metrics for model comparison:
- Retrieval latency
- Semantic similarity of retrieved context to query
- Token coverage / count
"""

import time
from typing import Dict, List, Any
import numpy as np

from modules.logger import get_logger
from modules.retrieval_engine import ModelRetrievalOutput

logger = get_logger(__name__)


def evaluate_retrieval(query: str, retrieval_results: Dict[str, ModelRetrievalOutput]) -> List[Dict[str, Any]]:
    """
    Evaluate and compare the three retrieval models based on:
    - Response Time (ms)
    - Mean Similarity Score
    - Number of Chunks Retrieved
    
    Args:
        query: The user query or extracted terms.
        retrieval_results: Comparison dictionary from retrieval engine.
        
    Returns:
        List of dictionaries with evaluation stats per model.
    """
    evaluation_report = []
    
    for key, output in retrieval_results.items():
        scores = [res.score for res in output.results]
        avg_score = float(np.mean(scores)) if scores else 0.0
        max_score = float(np.max(scores)) if scores else 0.0
        
        evaluation_report.append({
            "Model ID": key,
            "Model Name": output.model_name,
            "Latency (ms)": round(output.execution_time * 1000, 2),
            "Avg Match Score": round(avg_score, 4),
            "Max Match Score": round(max_score, 4),
            "Chunks Found": len(output.results)
        })
        
    return evaluation_report

"""
Module 13 — Feedback Loop

Handles collecting user ratings and qualitative feedback.
Saves logs to local JSON files in data/feedback/.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from config.settings import FEEDBACK_DIR
from modules.logger import get_logger

logger = get_logger(__name__)


def save_user_feedback(report_name: str, rating: int, comments: str, model_used: str) -> Path:
    """
    Save user feedback for a simplification session to a JSON file.
    
    Args:
        report_name: Name of the medical report file uploaded.
        rating: 1-5 integer rating.
        comments: Optional text comments.
        model_used: The simplification model name.
        
    Returns:
        Path of the saved feedback file.
    """
    feedback_data = {
        "timestamp": time.time(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_name": report_name,
        "rating": rating,
        "comments": comments.strip(),
        "model_used": model_used
    }
    
    # Save with unique timestamp name
    file_name = f"feedback_{int(time.time())}.json"
    dest_path = FEEDBACK_DIR / file_name
    
    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=4)
        logger.info(f"User feedback saved: {dest_path.name}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to save user feedback: {str(e)}")
        raise


def get_feedback_statistics() -> Dict[str, Any]:
    """
    Read all collected feedback files and aggregate statistics.
    
    Returns:
        Dictionary with count and average rating statistics.
    """
    ratings = []
    total_comments = 0
    
    for fb_file in FEEDBACK_DIR.glob("feedback_*.json"):
        try:
            with open(fb_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                ratings.append(data.get("rating", 0))
                if data.get("comments"):
                    total_comments += 1
        except Exception as e:
            logger.error(f"Error reading feedback file {fb_file}: {str(e)}")
            
    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
    
    return {
        "Total Submissions": len(ratings),
        "Average Rating": round(avg_rating, 2),
        "Comments Count": total_comments,
        "Rating Breakdown": {i: ratings.count(i) for i in range(1, 6)}
    }

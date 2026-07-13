"""
Module 14 — Monitoring & Logging

Provides centralized logging for all modules with file and console output.
Tracks processing times and errors.
"""

import logging
import time
from functools import wraps
from config.settings import LOGS_DIR, LOG_LEVEL, LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger instance.
    
    Args:
        name: Name for the logger (typically __name__ of the calling module).
    
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # File handler
    log_file = LOGS_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger


def log_execution_time(func):
    """
    Decorator that logs the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        logger.info(f"Starting: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Completed: {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed: {func.__name__} after {elapsed:.2f}s — {str(e)}")
            raise
    
    return wrapper


class PerformanceTracker:
    """Tracks and stores performance metrics for each pipeline stage."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, stage: str, duration: float, details: dict = None):
        """Record a performance metric."""
        self.metrics[stage] = {
            "duration_seconds": round(duration, 4),
            "details": details or {}
        }
    
    def get_summary(self) -> dict:
        """Return all recorded metrics."""
        return self.metrics
    
    def get_total_time(self) -> float:
        """Return total processing time across all stages."""
        return sum(m["duration_seconds"] for m in self.metrics.values())

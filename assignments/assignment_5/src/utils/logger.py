"""
Logging config with centralized logger for all modules.
"""
import logging
import sys
from typing import Optional
from ..settings import settings


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger
        level: Optional logging level, defaults to setting from settings
        
    Returns:
        Configured logger instance
    """
    # Use provided level or default from settings
    log_level = level if level is not None else settings.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Only add handler if not already present
    if not logger.handlers:
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Setup formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(console_handler)
    
    return logger


# Default logger for import
logger = get_logger("imdb_sentiment_analysis")

__all__ = ["logger", "get_logger"]
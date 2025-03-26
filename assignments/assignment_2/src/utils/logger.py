"""
Logger utility module for the project.
Provides a configured logger that can be imported and used throughout the project.
"""

import logging

from settings import settings


def setup_logger(log_level=None):
    """
    Set up and configure the logger for the project.
    
    Parameters:
    -----------
    log_level : int, optional
        The logging level to use. If None, uses the level from settings.
    
    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    # Use settings for configuration
    log_level = log_level if log_level is not None else settings.log_level
    
    # Create logger
    logger = logging.getLogger("fake_news_classifier")
    logger.setLevel(log_level)
    
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create and configure the default logger
logger = setup_logger() 
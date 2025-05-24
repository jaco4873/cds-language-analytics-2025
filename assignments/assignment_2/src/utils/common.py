"""
Common utility functions for the text classification project.
This module contains general purpose helper functions used across the project.
"""

import os
from settings import settings
from utils.logger import logger


def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Parameters:
    -----------
    directory : str
        Path to the directory to ensure exists
    """
    if not os.path.exists(directory):
        logger.debug(f"Creating directory: {directory}")
        os.makedirs(directory)


def ensure_all_dirs() -> list[str]:
    """
    Ensure all necessary directories exist based on configuration.

    Returns:
    --------
    List[str]
        List of created/verified directories
    """
    dirs = [
        "data",
        settings.vectorization.vectorized_dir,
        settings.output.output_dir,
        settings.output.models_dir,
        settings.output.reports_dir,
        settings.output.figures_dir,
    ]

    for directory in dirs:
        ensure_dir(directory)

    logger.debug("All required directories have been created/verified")
    return dirs

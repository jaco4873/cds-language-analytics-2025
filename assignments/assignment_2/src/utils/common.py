"""
Common utility functions for the text classification project.
This module contains general purpose helper functions used across the project.
"""

import os
from settings import settings


def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_all_dirs():
    """Ensure all necessary directories exist based on configuration."""
    dirs = [
        "data",
        settings.vectorization.vectorized_dir,
        settings.output.models_dir,
        settings.output.output_dir,
        settings.output.results_dir,
    ]

    for directory in dirs:
        ensure_dir(directory)

    return dirs


def print_section_header(title):
    """
    Print a formatted section header.

    Parameters:
    -----------
    title : str
        Title for the section header
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

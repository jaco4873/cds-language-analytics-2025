"""Data Loading Module

This module provides functions to load the news dataset and pretrained embeddings.
"""

import jsonlines
import numpy as np
from pathlib import Path

from src.config.settings import settings


def load_jsonl_data(file_path: Path) -> list[dict[str, any]]:
    """Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the parsed JSON data
    """
    with jsonlines.open(file_path) as reader:
        return list(reader)


def load_embeddings(file_path: Path) -> np.ndarray:
    """Load pretrained embeddings from a numpy file.
    
    Args:
        file_path: Path to the numpy embeddings file
        
    Returns:
        NumPy array containing the embeddings
    """
    return np.load(file_path, allow_pickle=True)


def prepare_data() -> tuple[list[dict[str, any]], list[str], list[str], np.ndarray]:
    """Prepare the data for topic modeling.
    
    Returns:
        Tuple containing:
        - List of data entries
        - List of headlines
        - List of categories
        - NumPy array of embeddings
    """
    # Load data and embeddings
    data_path = settings.DATA_DIR / settings.JSONL_FILE
    embeddings_path = settings.DATA_DIR / settings.HEADLINES_EMBEDDINGS
    
    data = load_jsonl_data(data_path)
    embeddings = load_embeddings(embeddings_path)
    
    # Extract headlines and categories
    headlines = [entry["headline"] for entry in data]
    categories = [entry["category"] for entry in data]
    
    return data, headlines, categories, embeddings 
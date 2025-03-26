#!/usr/bin/env python3
"""
Script to vectorize the Fake News dataset and save the results for reuse.
This handles the data preprocessing and feature extraction step,
allowing other scripts to directly use the vectorized data.
"""

from utils.data_utils import (
    load_fake_news_data,
    preprocess_split_data,
)
from utils.vectorization_utils import (
    vectorize_text,
    save_vectorized_data,
)
from utils.common import ensure_dir
from utils.logger import logger
from settings import settings

def vectorization_pipeline():
    """Main function to vectorize and save data."""
    # Create output directory if it doesn't exist
    output_dir = settings.vectorization.vectorized_dir
    ensure_dir(output_dir)

    data_path = settings.data.csv_path
    logger.info(f"Loading Fake News dataset from {data_path}...")
    X, y = load_fake_news_data(data_path)

    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = preprocess_split_data(X, y)

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    logger.info("Saving vectorized data...")
    save_vectorized_data(
        X_train_vec, X_test_vec, y_train, y_test, vectorizer, output_dir=output_dir
    )

    logger.info(f"Vectorized data saved to: {output_dir}")


if __name__ == "__main__":
    vectorization_pipeline()

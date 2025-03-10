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
from settings import settings


def main():
    """Main function to vectorize and save data."""
    # Create output directory if it doesn't exist
    output_dir = settings.vectorization.vectorized_dir
    ensure_dir(output_dir)

    data_path = settings.data.csv_path
    print(f"Loading Fake News dataset from {data_path}...")
    X, y = load_fake_news_data(data_path)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = preprocess_split_data(X, y)

    vectorizer_type = settings.vectorization.vectorizer_type
    max_features = settings.vectorization.max_features
    print(f"Vectorizing text data using {vectorizer_type.upper()}...")
    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train,
        X_test,
        vectorizer_type=vectorizer_type,
        max_features=max_features,
    )

    print("Saving vectorized data...")
    save_vectorized_data(
        X_train_vec, X_test_vec, y_train, y_test, vectorizer, output_dir=output_dir
    )

    print(f"Vectorized data saved to: {output_dir}")


if __name__ == "__main__":
    main()

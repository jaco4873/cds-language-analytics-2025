"""
Vectorization utility functions for text classification tasks.
This module contains functions for vectorizing text data and handling vectorized data.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from utils.common import ensure_dir
from utils.logger import logger
from settings import settings


def vectorize_text(
    X_train, X_test
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer | CountVectorizer]:
    """
    Vectorize text data using either TF-IDF or Count vectorization.

    Parameters:
    -----------
    X_train : array-like
        Training text data
    X_test : array-like
        Testing text data
    config : dict, optional
        Configuration dictionary. If None, uses default config.
    **kwargs : dict
        Additional parameters to pass to the vectorizer

    Returns:
    --------
    X_train_vec : sparse matrix
        Vectorized training data
    X_test_vec : sparse matrix
        Vectorized testing data
    vectorizer : TfidfVectorizer or CountVectorizer
        The fitted vectorizer
    """

    vec_config = settings.vectorization

    # Set vectorizer parameters
    defaults = {
        "max_features": vec_config.max_features,
        "min_df": vec_config.min_df,
        "max_df": vec_config.max_df,
        "lowercase": vec_config.lowercase,
    }

    # Create the  vectorizer
    vectorizer_type = vec_config.vectorizer_type
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(**defaults)
    elif vectorizer_type.lower() == "count":
        vectorizer = CountVectorizer(**defaults)
    else:
        raise ValueError(
            f"Unknown vectorizer type: {vectorizer_type}. Use 'tfidf' or 'count'."
        )

    # Fit and transform the training data, then transform the test data
    logger.info("Vectorizing training data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    logger.info("Vectorizing testing data...")
    X_test_vec = vectorizer.transform(X_test)

    logger.info(
        f"Vectorization complete. Training shape: {X_train_vec.shape}, Testing shape: {X_test_vec.shape}"
    )

    return X_train_vec, X_test_vec, vectorizer


def save_vectorized_data(
    X_train_vec, X_test_vec, y_train, y_test, vectorizer, output_dir=None
) -> None:
    """
    Save vectorized data and labels to disk.

    Parameters:
    -----------
    X_train_vec : sparse matrix
        Vectorized training data
    X_test_vec : sparse matrix
        Vectorized testing data
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    vectorizer : TfidfVectorizer or CountVectorizer
        The fitted vectorizer
    output_dir : str, optional
        Directory to save the data
    """
    if output_dir is None:
        output_dir = settings.vectorization.vectorized_dir

    ensure_dir(output_dir)

    # Save sparse matrices and arrays
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    with open(os.path.join(output_dir, "X_train_vec.pkl"), "wb") as f:
        pickle.dump(X_train_vec, f)

    with open(os.path.join(output_dir, "X_test_vec.pkl"), "wb") as f:
        pickle.dump(X_test_vec, f)

    # Save the vectorizer
    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info(f"Vectorized data saved to {output_dir}")


def load_vectorized_data(
    input_dir,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load vectorized data and labels from disk.

    Parameters:
    -----------
    input_dir : str, optional
        Directory to load the data from

    Returns:
    --------
    X_train_vec : sparse matrix
        Vectorized training data
    X_test_vec : sparse matrix
        Vectorized testing data
    y_train : array-like
        Training labels
    y_test : array-like
        Testing labels
    vectorizer : TfidfVectorizer or CountVectorizer
        The fitted vectorizer

    Raises:
    -------
    FileNotFoundError:
        If the required vectorized data files are not found in the input directory
    """
    input_dir = settings.vectorization.vectorized_dir

    # Check if all required files exist
    required_files = [
        "y_train.npy",
        "y_test.npy",
        "X_train_vec.pkl",
        "X_test_vec.pkl",
        "vectorizer.pkl",
    ]

    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(input_dir, f))
    ]

    if missing_files:
        logger.error(
            f"Vectorized data not found in {input_dir}. Missing files: {', '.join(missing_files)}. "
            f"Please run vectorization first using vectorize_text() and save_vectorized_data()."
        )
        raise FileNotFoundError(
            f"Vectorized data not found in {input_dir}. Missing files: {', '.join(missing_files)}. "
            f"Please run vectorization first using vectorize_text() and save_vectorized_data()."
        )

    # Load arrays
    y_train = np.load(os.path.join(input_dir, "y_train.npy"))
    y_test = np.load(os.path.join(input_dir, "y_test.npy"))

    # Load sparse matrices
    with open(os.path.join(input_dir, "X_train_vec.pkl"), "rb") as f:
        X_train = pickle.load(f)

    with open(os.path.join(input_dir, "X_test_vec.pkl"), "rb") as f:
        X_test = pickle.load(f)

    logger.info(f"Vectorized data loaded from {input_dir}")

    return X_train, X_test, y_train, y_test

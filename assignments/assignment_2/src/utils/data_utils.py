"""
Data utility functions for text classification tasks.
This module contains functions for loading and preprocessing the Fake News dataset.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from settings import settings


def load_fake_news_data(csv_path=None):
    """
    Load the Fake News dataset from the specified CSV file.

    Parameters:
    -----------
    csv_path : str, optional
        Path to the Fake News CSV file. If None, uses the path from config.

    Returns:
    --------
    X : array-like
        Feature array (text content)
    y : array-like
        Target array (1 for FAKE, 0 for REAL)
    """
    if csv_path is None:
        csv_path = settings.data.csv_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Please ensure the dataset exists."
        )

    try:
        # Load the data
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded with shape: {df.shape}")

        # Extract features and labels
        X = df["text"].values
        y = (df["label"] == "FAKE").astype(
            int
        )  # Convert to binary (1 for FAKE, 0 for REAL)

        return X, y

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError(f"Failed to load the dataset: {e}")


def preprocess_split_data(X, y, test_size=None, random_state=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    -----------
    X : array-like
        Feature array (text content)
    y : array-like
        Target array
    test_size : float, optional
        Proportion of the dataset to include in the test split.
        If None, uses the value from config.
    random_state : int, optional
        Random seed for reproducibility.
        If None, uses the value from config.

    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split dataset
    """
    if test_size is None:
        test_size = settings.data.test_size

    if random_state is None:
        random_state = settings.data.random_state

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

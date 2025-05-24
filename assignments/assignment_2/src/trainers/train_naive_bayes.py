"""
Script to train a Naive Bayes classifier on the Fake News dataset.
This script uses settings from the central configuration.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from settings import settings
from utils.trainer_utils import train_and_evaluate_model


def train_naive_bayes(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> dict:
    """
    Function to train and evaluate the Naive Bayes model.

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels

    Returns:
    --------
    dict
        Dictionary of model performance metrics
    """
    # Get configuration from settings
    nb_config = settings.models.naive_bayes

    # Create model info string for logging
    model_info = f"Training Naive Bayes (alpha={nb_config.alpha}, fit_prior={nb_config.fit_prior})..."

    # Define model factory function
    def create_model():
        return MultinomialNB(**nb_config.dict(exclude={"name", "enabled"}))

    # Use the shared train and evaluate function
    metrics = train_and_evaluate_model(
        model_factory=create_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=nb_config,
        model_info=model_info,
    )

    return metrics


if __name__ == "__main__":
    train_naive_bayes()

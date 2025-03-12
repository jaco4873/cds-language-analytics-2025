#!/usr/bin/env python3
"""
Script to train a Naive Bayes classifier on the Fake News dataset.
This script loads pre-vectorized data created by the vectorize_data.py script
and uses settings from the central configuration.
"""

import time
from sklearn.naive_bayes import MultinomialNB
from settings import settings
from utils.model_utils import evaluate_model
from utils.common import ensure_dir


def train_naive_bayes(X_train, X_test, y_train, y_test) -> dict:
    """Function to train and evaluate the Naive Bayes model."""
    # Get configuration from settings
    nb_config = settings.models.naive_bayes

    models_dir = settings.output.models_dir
    output_dir = settings.output.output_dir
    model_name = nb_config.name

    # Ensure output directories exist
    ensure_dir(models_dir)
    ensure_dir(output_dir)
    # Create the model directly
    print(
        f"Training Naive Bayes (alpha={nb_config.alpha}, fit_prior={nb_config.fit_prior})..."
    )
    model = MultinomialNB(alpha=nb_config.alpha, fit_prior=nb_config.fit_prior)

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds")

    # Evaluate and save the model and results
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_name,
        training_time,
        models_dir,
        output_dir,
    )

    return metrics


if __name__ == "__main__":
    train_naive_bayes()

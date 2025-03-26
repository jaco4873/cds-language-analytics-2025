#!/usr/bin/env python3
"""
Script to train a logistic regression classifier on the Fake News dataset.
This script uses settings from the central configuration.
"""

import time
from sklearn.linear_model import LogisticRegression
from settings import settings
from utils.model_utils import evaluate_model
from utils.common import ensure_dir
from utils.logger import logger


def train_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    """Function to train and evaluate the logistic regression model."""
    # Get configuration from settings
    lr_config = settings.models.logistic_regression

    # Set up directories from settings
    models_dir = settings.output.models_dir
    output_dir = settings.output.output_dir
    model_name = lr_config.name

    # Ensure output directories exist
    ensure_dir(models_dir)
    ensure_dir(output_dir)

    # Create the model directly
    logger.info(
        f"Training Logistic Regression (C={lr_config.c_value}, max_iter={lr_config.max_iter}, solver={lr_config.solver})..."
    )

    model = LogisticRegression(
        C=lr_config.c_value,
        max_iter=lr_config.max_iter,
        random_state=settings.models.random_state,
        solver=lr_config.solver,
    )

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

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
    train_logistic_regression()

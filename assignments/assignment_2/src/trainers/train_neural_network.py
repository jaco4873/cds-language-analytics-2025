#!/usr/bin/env python3
"""
Script to train a neural network classifier on the Fake News dataset.
This script loads pre-vectorized data created by the vectorize_data.py script
and uses settings from the central configuration.
"""

import time
from sklearn.neural_network import MLPClassifier
from settings import settings
from utils.model_utils import evaluate_model
from utils.common import ensure_dir


def train_neural_network(X_train, X_test, y_train, y_test) -> dict:
    """Function to train and evaluate the neural network model."""
    # Get configuration from settings
    nn_config = settings.models.neural_network

    # Set up directories from settings
    models_dir = settings.output.models_dir
    output_dir = settings.output.output_dir
    model_name = nn_config.name

    # Get the hidden layer sizes directly from settings
    hidden_layer_sizes = tuple(nn_config.hidden_layer_sizes)

    # Ensure output directories exist
    ensure_dir(models_dir)
    ensure_dir(output_dir)

    # Create the model directly
    print(f"Training neural network with architecture: {hidden_layer_sizes}")
    print(
        f"Parameters: alpha={nn_config.alpha}, max_iter={nn_config.max_iter}, learning_rate_init={nn_config.learning_rate_init}"
    )

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=nn_config.activation,
        solver=nn_config.solver,
        alpha=nn_config.alpha,
        batch_size=nn_config.batch_size,
        learning_rate=nn_config.learning_rate,
        learning_rate_init=nn_config.learning_rate_init,
        max_iter=nn_config.max_iter,
        random_state=settings.models.random_state,
        early_stopping=nn_config.early_stopping,
        validation_fraction=nn_config.validation_fraction,
    )

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds")

    # Print information about convergence
    if model.n_iter_ < nn_config.max_iter:
        print(f"Convergence achieved after {model.n_iter_} iterations")
    else:
        print(
            f"Warning: Maximum iterations ({nn_config.max_iter}) reached without convergence"
        )

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
    train_neural_network()

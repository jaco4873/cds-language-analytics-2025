#!/usr/bin/env python3
"""
Script to train a neural network classifier on the Fake News dataset.
This script loads pre-vectorized data created by the vectorize_data.py script.
"""

import os
import argparse
from sklearn.neural_network import MLPClassifier
from utils.model_utils import (
    save_model,
    save_classification_report,
)
from utils.common import ensure_dir
from utils.vectorization_utils import load_vectorized_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a neural network classifier on the Fake News dataset."
    )

    parser.add_argument(
        "--vectorized-data-dir",
        type=str,
        default="data/vectorized",
        help="Directory containing vectorized data.",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save the trained model.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the classification report.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="neural_network",
        help="Name for the saved model file.",
    )

    parser.add_argument(
        "--hidden-layer-sizes",
        type=str,
        default="100,50",
        help='Comma-separated list of hidden layer sizes (e.g., "100,50" for two hidden layers).',
    )

    parser.add_argument(
        "--alpha", type=float, default=0.0001, help="L2 regularization parameter."
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum number of iterations for the solver.",
    )

    parser.add_argument(
        "--learning-rate-init", type=float, default=0.001, help="Initial learning rate."
    )

    return parser.parse_args()


def train_neural_network(
    X_train,
    y_train,
    hidden_layer_sizes=(100, 50),
    alpha=0.0001,
    max_iter=200,
    learning_rate_init=0.001,
):
    """
    Train a neural network model using MLPClassifier.

    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training data features
    y_train : array-like
        Training data labels
    hidden_layer_sizes : tuple, optional (default=(100, 50))
        Number of neurons in each hidden layer
    alpha : float, optional (default=0.0001)
        L2 regularization parameter
    max_iter : int, optional (default=200)
        Maximum number of iterations for the solver
    learning_rate_init : float, optional (default=0.001)
        Initial learning rate

    Returns:
    --------
    model : MLPClassifier
        Trained neural network model
    """
    print(f"Training neural network model with architecture: {hidden_layer_sizes}...")
    print(
        f"Parameters: alpha={alpha}, max_iter={max_iter}, learning_rate_init={learning_rate_init}"
    )

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",  # ReLU activation function
        solver="adam",  # Adam optimizer
        alpha=alpha,  # L2 regularization
        batch_size="auto",  # Automatically determine batch size
        learning_rate="adaptive",  # Adaptive learning rate
        learning_rate_init=learning_rate_init,  # Initial learning rate
        max_iter=max_iter,  # Maximum iterations
        early_stopping=True,  # Use early stopping
        validation_fraction=0.1,  # Use 10% of training data for validation
        random_state=42,  # For reproducibility
        verbose=True,  # Print progress
    )

    model.fit(X_train, y_train)
    print("Model training complete!")

    # Print information about convergence
    if model.n_iter_ < max_iter:
        print(f"Convergence achieved after {model.n_iter_} iterations")
    else:
        print(f"Warning: Maximum iterations ({max_iter}) reached without convergence")

    return model


def main():
    """Main function to train and evaluate the neural network model."""
    args = parse_args()

    # Parse hidden layer sizes from string
    hidden_layer_sizes = tuple(int(x) for x in args.hidden_layer_sizes.split(","))

    # Ensure output directories exist
    ensure_dir(args.models_dir)
    ensure_dir(args.output_dir)

    # Load pre-vectorized data
    print(f"Loading pre-vectorized data from {args.vectorized_data_dir}...")
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = load_vectorized_data(
        args.vectorized_data_dir
    )

    # Train the model
    model = train_neural_network(
        X_train_vec,
        y_train,
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=args.alpha,
        max_iter=args.max_iter,
        learning_rate_init=args.learning_rate_init,
    )

    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_vec)

    # Save classification report
    report_name = f"{args.model_name}_report"
    save_classification_report(y_test, y_pred, report_name, args.output_dir)

    # Save the model
    save_model(model, args.model_name, args.models_dir)

    print("\nNeural network model training and evaluation complete!")
    print(f"Model saved to: {os.path.join(args.models_dir, args.model_name + '.pkl')}")
    print(
        f"Classification report saved to: {os.path.join(args.output_dir, report_name + '.txt')}"
    )


if __name__ == "__main__":
    main()

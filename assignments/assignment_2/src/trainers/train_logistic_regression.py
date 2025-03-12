#!/usr/bin/env python3
"""
Script to train a logistic regression classifier on the Fake News dataset.
This script loads pre-vectorized data created by the vectorize_data.py script.
"""

import os
import argparse
from sklearn.linear_model import LogisticRegression
from utils.model_utils import (
    save_model,
    save_classification_report,
    ensure_dir,
)
from utils.vectorization_utils import load_vectorized_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a logistic regression classifier on the Fake News dataset."
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
        default="logistic_regression",
        help="Name for the saved model file.",
    )

    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Regularization parameter for logistic regression.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for the solver.",
    )

    return parser.parse_args()


def train_logistic_regression(X_train, y_train, c_value=1.0, max_iter=1000):
    """
    Train a logistic regression model.

    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training data features
    y_train : array-like
        Training data labels
    c_value : float, optional (default=1.0)
        Regularization parameter
    max_iter : int, optional (default=1000)
        Maximum number of iterations for the solver

    Returns:
    --------
    model : LogisticRegression
        Trained logistic regression model
    """
    print(f"Training logistic regression model (C={c_value}, max_iter={max_iter})...")
    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        random_state=42,
        solver="liblinear",
    )

    model.fit(X_train, y_train)
    print("Model training complete!")

    return model


def main():
    """Main function to train and evaluate the logistic regression model."""
    args = parse_args()

    # Ensure output directories exist
    ensure_dir(args.models_dir)
    ensure_dir(args.output_dir)

    # Load pre-vectorized data
    print(f"Loading pre-vectorized data from {args.vectorized_data_dir}...")
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = load_vectorized_data(
        args.vectorized_data_dir
    )

    # Train the model
    model = train_logistic_regression(
        X_train_vec, y_train, c_value=args.c_value, max_iter=args.max_iter
    )

    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_vec)

    # Save classification report
    report_name = f"{args.model_name}_report"
    save_classification_report(y_test, y_pred, report_name, args.output_dir)

    # Save the model
    save_model(model, args.model_name, args.models_dir)

    print("\nLogistic regression model training and evaluation complete!")
    print(f"Model saved to: {os.path.join(args.models_dir, args.model_name + '.pkl')}")
    print(
        f"Classification report saved to: {os.path.join(args.output_dir, report_name + '.txt')}"
    )


if __name__ == "__main__":
    main()

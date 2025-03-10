#!/usr/bin/env python3
"""
Script to train a Naive Bayes classifier on the Fake News dataset.
This script loads pre-vectorized data created by the vectorize_data.py script.
"""

import os
import argparse
from sklearn.naive_bayes import MultinomialNB
from utils.model_utils import (
    save_model,
    save_classification_report,
)
from utils.common import ensure_dir
from utils.vectorization_utils import load_vectorized_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Naive Bayes classifier on the Fake News dataset."
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
        default="naive_bayes",
        help="Name for the saved model file.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).",
    )

    parser.add_argument(
        "--fit-prior",
        action="store_true",
        default=True,
        help="Whether to learn class prior probabilities or not.",
    )

    return parser.parse_args()


def train_naive_bayes(X_train, y_train, alpha=1.0, fit_prior=True):
    """
    Train a Multinomial Naive Bayes model.

    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training data features
    y_train : array-like
        Training data labels
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
    fit_prior : bool, optional (default=True)
        Whether to learn class prior probabilities

    Returns:
    --------
    model : MultinomialNB
        Trained Naive Bayes model
    """
    print(
        f"Training Multinomial Naive Bayes model (alpha={alpha}, fit_prior={fit_prior})..."
    )
    model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)

    model.fit(X_train, y_train)
    print("Model training complete!")

    return model


def main():
    """Main function to train and evaluate the Naive Bayes model."""
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
    model = train_naive_bayes(
        X_train_vec, y_train, alpha=args.alpha, fit_prior=args.fit_prior
    )

    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_vec)

    # Save classification report
    report_name = f"{args.model_name}_report"
    save_classification_report(y_test, y_pred, report_name, args.output_dir)

    # Save the model
    save_model(model, args.model_name, args.models_dir)

    print("\nNaive Bayes model training and evaluation complete!")
    print(f"Model saved to: {os.path.join(args.models_dir, args.model_name + '.pkl')}")
    print(
        f"Classification report saved to: {os.path.join(args.output_dir, report_name + '.txt')}"
    )


if __name__ == "__main__":
    main()

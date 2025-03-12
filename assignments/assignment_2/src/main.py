#!/usr/bin/env python3
"""
Orchestration script for text classification benchmarks on the Fake News Dataset.
This script runs the entire pipeline:
1. Data loading and preprocessing
2. Vectorization
3. Training of multiple classifiers
4. Evaluation and comparison of results
5. Visualization of performance metrics
"""

import pandas as pd
import seaborn as sns

from trainers.train_logistic_regression import train_logistic_regression
from trainers.train_neural_network import train_neural_network
from trainers.train_naive_bayes import train_naive_bayes
from utils.data_utils import load_fake_news_data, preprocess_split_data
from utils.vectorization_utils import (
    vectorize_text,
    save_vectorized_data,
    load_vectorized_data,
)
from utils.common import print_section_header, ensure_all_dirs
from utils.visualization_utils import create_visualizations
from utils.result_utils import create_comparison_table

# Import settings
from settings import settings

# Set style for plots
sns.set_style("whitegrid")


def train_evaluate_all_models(X_train, X_test, y_train, y_test) -> list[dict]:
    """
    Train and evaluate all enabled classification models from settings.

    Parameters:
    -----------
    X_train, X_test, y_train, y_test : training and testing data

    Returns:
    --------
    results : list of dict
        List of dictionaries containing model metrics
    """

    print(
        f"Loading pre-vectorized data from {settings.vectorization.vectorized_dir}..."
    )
    X_train_vec, X_test_vec, y_train, y_test = load_vectorized_data(
        settings.vectorization.vectorized_dir
    )

    results = []

    # Process each enabled model
    if settings.models.logistic_regression.enabled:
        print_section_header(f"Training {settings.models.logistic_regression.name}")
        logistic_regression_metrics = train_logistic_regression(
            X_train_vec, X_test_vec, y_train, y_test
        )
        results.append(logistic_regression_metrics)
    if settings.models.neural_network.enabled:
        print_section_header(f"Training {settings.models.neural_network.name}")
        neural_network_metrics = train_neural_network(X_train, X_test, y_train, y_test)
        results.append(neural_network_metrics)
    if settings.models.naive_bayes.enabled:
        print_section_header(f"Training {settings.models.naive_bayes.name}")
        naive_bayes_metrics = train_naive_bayes(X_train, X_test, y_train, y_test)
        results.append(naive_bayes_metrics)

    return results


def main():
    """Main function to run the entire pipeline."""

    print("Evaluating multiple classifiers on the Fake News Dataset.")
    print("Using settings from config.yaml")

    # Print a summary of key settings
    print(f"Data source: {settings.data.csv_path}")
    print(f"Test size: {settings.data.test_size}")
    print(
        f"Vectorizer: {settings.vectorization.vectorizer_type.upper()} with {settings.vectorization.max_features} features"
    )

    # List enabled models
    enabled_models = []
    if settings.models.logistic_regression.enabled:
        enabled_models.append("Logistic Regression")
    if settings.models.neural_network.enabled:
        enabled_models.append("Neural Network")
    if settings.models.naive_bayes.enabled:
        enabled_models.append("Naive Bayes")

    print(f"Enabled models: {', '.join(enabled_models)}")
    print(f"Results will be saved to: {settings.output.results_dir}")

    # Ensure all directories exist
    ensure_all_dirs()

    # Override vectorization skip from command line
    skip_vectorization = settings.vectorization.skip_vectorization

    # Step 1: Data loading and vectorization
    if not skip_vectorization:
        print_section_header("Data Loading and Preprocessing")
        csv_path = settings.data.csv_path
        print(f"Loading data from {csv_path}...")
        X, y = load_fake_news_data(csv_path)

        # Split data
        test_size = settings.data.test_size
        random_state = settings.data.random_state
        print(f"Splitting data into train and test sets (test_size={test_size})...")
        X_train, X_test, y_train, y_test = preprocess_split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        # Vectorize text
        vectorizer_type = settings.vectorization.vectorizer_type
        print(f"Vectorizing text data using {vectorizer_type.upper()}...")
        X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

        # Save vectorized data if configured
        save_vectorized = settings.vectorization.save_vectorized
        if save_vectorized:
            vectorized_dir = settings.vectorization.vectorized_dir
            print(f"Saving vectorized data to {vectorized_dir}...")
            save_vectorized_data(
                X_train_vec,
                X_test_vec,
                y_train,
                y_test,
                vectorizer,
                output_dir=vectorized_dir,
            )
    else:
        # Load previously vectorized data
        print_section_header("Loading Pre-vectorized Data")
        vectorized_dir = settings.vectorization.vectorized_dir
        print(f"Loading vectorized data from {vectorized_dir}...")
        X_train_vec, X_test_vec, y_train, y_test, vectorizer = load_vectorized_data(
            input_dir=vectorized_dir
        )

    # Step 2: Train and evaluate models
    results = train_evaluate_all_models(X_train_vec, X_test_vec, y_train, y_test)

    # Step 3: Create comparison table
    print_section_header("Creating Result Comparison")
    results_dir = settings.output.results_dir
    create_comparison_table(results, results_dir)

    # Step 4: Create visualizations
    if settings.output.visualize_results:
        print_section_header("Creating Visualizations")
        print(f"Saving visualizations to {results_dir}...")
        create_visualizations(pd.DataFrame(results), results_dir)

    print("Analysis completed successfully.")

    return results


if __name__ == "__main__":
    main()

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

import os
import argparse
import time
import pandas as pd
import seaborn as sns

from utils.data_utils import load_fake_news_data, preprocess_split_data
from utils.vectorization_utils import (
    vectorize_text,
    save_vectorized_data,
    load_vectorized_data,
)
from utils.model_utils import (
    train_model,
    evaluate_model,
)
from utils.common import print_section_header, ensure_all_dirs
from utils.visualization import create_visualizations

# Import settings
from settings import settings

# Set style for plots
sns.set_style("whitegrid")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run text classification benchmarks on the Fake News Dataset."
    )
    parser.add_argument(
        "--skip-vectorization",
        action="store_true",
        help="Skip data loading and vectorization, use existing vectorized data.",
    )
    return parser.parse_args()


def train_evaluate_all_models(X_train, X_test, y_train, y_test):
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
    models_dir = settings.output.models_dir
    output_dir = settings.output.output_dir
    results = []

    # Get all enabled model configurations
    model_configs = [
        (settings.models.logistic_regression, "logistic_regression"),
        (settings.models.neural_network, "neural_network"),
        (settings.models.naive_bayes, "naive_bayes"),
    ]

    # Process each enabled model
    for config, model_type in model_configs:
        if not config.enabled:
            continue

        print_section_header(f"Training {config.name}")

        # Train model
        start_time = time.time()
        model = train_model(model_type, config, X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(
            model, X_test, y_test, config.name, training_time, models_dir, output_dir
        )

        # Store results
        results.append(metrics)

        # Print results summary
        print("Model evaluation complete!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (REAL): {metrics['real_f1']:.4f}")
        print(f"F1-Score (FAKE): {metrics['fake_f1']:.4f}")

    return results


def create_comparison_table(results, results_dir):
    """
    Create and save a table comparing model performance.

    Parameters:
    -----------
    results : list of dict
        List of dictionaries containing model metrics
    results_dir : str
        Directory to save the comparison table

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the comparison table
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Reorder columns for better display
    display_cols = [
        "model",
        "description",
        "accuracy",
        "training_time",
        "real_precision",
        "real_recall",
        "real_f1",
        "fake_precision",
        "fake_recall",
        "fake_f1",
    ]

    # Rename columns for better display
    rename_map = {
        "model": "Model",
        "description": "Description",
        "real_precision": "REAL Precision",
        "real_recall": "REAL Recall",
        "real_f1": "REAL F1",
        "fake_precision": "FAKE Precision",
        "fake_recall": "FAKE Recall",
        "fake_f1": "FAKE F1",
        "accuracy": "Accuracy",
        "training_time": "Training Time (s)",
    }

    # Create and display comparison table
    comparison_table = df[display_cols].rename(columns=rename_map).set_index("Model")

    # Print the table
    print_section_header("Model Performance Comparison")
    print(comparison_table.round(4))

    # Save the table to CSV
    comparison_table.to_csv(os.path.join(results_dir, "model_comparison.csv"))
    print(f"\nComparison saved to {os.path.join(results_dir, 'model_comparison.csv')}")

    return df


def main():
    """Main function to run the entire pipeline."""
    # Parse command line arguments
    args = parse_args()

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
    skip_vectorization = args.skip_vectorization

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

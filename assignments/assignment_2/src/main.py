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

from trainers.train_logistic_regression import train_logistic_regression
from trainers.train_neural_network import train_neural_network
from trainers.train_naive_bayes import train_naive_bayes
from utils.vectorization_utils import (
    load_vectorized_data,
)
from utils.common import ensure_all_dirs
from utils.visualization_utils import create_visualizations
from utils.result_utils import create_comparison_table
from utils.logger import logger
from data_processing.vectorize_data import vectorization_pipeline
from settings import settings


def train_test_evaluate_all_models(X_train, X_test, y_train, y_test) -> list[dict]:
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

    results = []

    # Process each enabled model
    if settings.models.logistic_regression.enabled:
        logger.info(f"Training {settings.models.logistic_regression.name}")
        logistic_regression_metrics = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )
        results.append(logistic_regression_metrics)
    if settings.models.neural_network.enabled:
        logger.info(f"Training {settings.models.neural_network.name}")
        neural_network_metrics = train_neural_network(
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(neural_network_metrics)
    if settings.models.naive_bayes.enabled:
        logger.info(f"Training {settings.models.naive_bayes.name}")
        naive_bayes_metrics = train_naive_bayes(
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(naive_bayes_metrics)

    return results


def main():
    """Main function to run the entire pipeline."""

    logger.info("Evaluating multiple classifiers on the Fake News Dataset.")
    logger.info("Using settings from config.yaml")
    logger.info(f"Log level: {settings.log_level}")

    # Log a summary of key settings
    logger.info(f"Data source: {settings.data.csv_path}")
    logger.info(f"Test size: {settings.data.test_size}")

    # Ensure all directories exist
    ensure_all_dirs()

    # Step 1: Data loading and vectorization (if needed)
    skip_vectorization = settings.vectorization.skip_vectorization
    if not skip_vectorization:
        logger.info("Starting Data Loading and Preprocessing")
        vectorization_pipeline()

    # Step 2: Load the vectorized data
    logger.info(
        f"Loading pre-vectorized data from {settings.vectorization.vectorized_dir}..."
    )
    X_train, X_test, y_train, y_test = load_vectorized_data(
        settings.vectorization.vectorized_dir
    )

    # Step 3: Train, test and evaluate models
    results = train_test_evaluate_all_models(X_train, X_test, y_train, y_test)

    # Step 4: Create comparison table
    logger.info("Creating Result Comparison")
    results_dir = settings.output.results_dir
    create_comparison_table(results, results_dir)

    # Step 5: Create visualizations
    if settings.output.visualize_results:
        logger.info("Creating Visualizations")
        logger.info(f"Saving visualizations to {results_dir}...")
        create_visualizations(pd.DataFrame(results), results_dir)

    logger.info("Analysis completed successfully.")

    return results


if __name__ == "__main__":
    main()

"""
Baseline logistic regression for IMDb sentiment analysis.
"""

import time
from typing import Any

from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from ..settings import settings
from ..utils.logger import logger


def optimize_logistic_regression(
    train_dataset: Dataset,
    val_dataset: Dataset,
    random_state: int = 42,
    n_iter: int = settings.BAYES_SEARCH_ITERATIONS,
) -> tuple[dict[str, Any], Pipeline]:
    """
    Optimize TF-IDF + Logistic Regression hyperparameters using Bayesian optimization.

    Args:
        train_dataset: Training data
        val_dataset: Validation data
        random_state: Random seed
        n_iter: Number of iterations for Bayesian optimization

    Returns:
        Best hyperparameters and the best pipeline
    """
    logger.info("Optimizing hyperparameters using Bayesian optimization...")

    # Create base pipeline
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3))),
            (
                "classifier",
                LogisticRegression(random_state=random_state, n_jobs=-1, max_iter=1000),
            ),
        ]
    )

    # Define search space
    search_space = {
        "tfidf__max_features": Integer(5000, 20000),
        "tfidf__min_df": Integer(2, 15),
        "tfidf__max_df": Real(0.6, 0.95),
        "tfidf__sublinear_tf": Categorical([True, False]),
        "classifier__C": Real(0.01, 100, prior="log-uniform"),
    }

    # Create Bayesian search
    search = BayesSearchCV(
        pipeline,
        search_space,
        n_iter=n_iter,
        cv=settings.BAYES_CV_FOLDS,  # Cross-validation within training set
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )

    # Get data
    X_train = train_dataset["review"]
    y_train = train_dataset["sentiment"]

    # Run optimization
    start_time = time.time()
    search.fit(X_train, y_train)
    optimization_time = time.time() - start_time

    logger.info(f"Bayesian optimization completed in {optimization_time:.2f} seconds")
    logger.info(f"Best hyperparameters: {search.best_params_}")

    # Evaluate best model on validation set
    X_val = val_dataset["review"]
    y_val = val_dataset["sentiment"]
    val_accuracy = search.score(X_val, y_val)

    logger.info(f"Best model validation accuracy: {val_accuracy:.4f}")

    return search.best_params_, search.best_estimator_


def train_test_logistic_regression(
    train_dataset: Dataset, val_dataset: Dataset, random_state: int = 42
) -> tuple[Pipeline, dict[str, float]]:
    """
    Train TF-IDF + Logistic Regression with Bayesian-optimized hyperparameters.

    Args:
        train_dataset: Training data
        val_dataset: Validation data
        random_state: Random seed

    Returns:
        Trained model and validation metrics
    """
    # Find optimal hyperparameters
    _, best_model = optimize_logistic_regression(
        train_dataset, val_dataset, random_state
    )

    # Evaluate on validation set
    val_metrics = evaluate_model_metrics(best_model, val_dataset)

    logger.info(f"Optimized model validation metrics: {val_metrics}")

    return best_model, val_metrics


def evaluate_model_metrics(
    model: Pipeline,
    dataset: Dataset,
) -> dict[str, float]:
    """
    Calculate model metrics on a dataset.

    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        dataset_type: Type of dataset (test/validation) for logging

    Returns:
        Dictionary with performance metrics
    """
    reviews = dataset["review"]
    sentiments = dataset["sentiment"]

    # Get predictions
    predictions = model.predict(reviews)
    predictions_list = predictions.tolist()

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        sentiments, predictions_list, average="weighted"
    )
    acc = accuracy_score(sentiments, predictions_list)

    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    return metrics


def evaluate_logistic_regression_model(
    model: Pipeline, test_dataset: Dataset
) -> tuple[dict[str, float], list[int]]:
    """
    Evaluate baseline model on test data.

    Args:
        model: Trained baseline model
        test_dataset: Test data

    Returns:
        Metrics dictionary and prediction list
    """
    logger.info("Evaluating model on test data...")

    test_reviews = test_dataset["review"]

    # Get predictions
    predictions = model.predict(test_reviews)
    predictions_list = predictions.tolist()

    # Calculate metrics
    metrics = evaluate_model_metrics(model, test_dataset)

    logger.info(f"Test metrics: {metrics}")

    return metrics, predictions_list
